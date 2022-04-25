import copy
import itertools
import math
import numpy as np
import logging
import time
import torch
import torch.nn as nn
import torchvision

from agents.base import Agent
from agents.nil_global import nil_global_agent
from utils.utils import get_device, move_cuda, plot_candidates
from utils.meters import AverageMeter, accuracy

from bisect import bisect


class nil_agent(Agent):
    def __init__(
        self,
        model,
        config,
        optimizer,
        criterion,
        cuda,
        buffer_cuda,
        log_interval,
        state_dict=None,
    ):
        super(nil_agent, self).__init__(
            model,
            config,
            optimizer,
            criterion,
            cuda,
            buffer_cuda,
            log_interval,
            state_dict,
        )

        self.device = "cuda" if self.buffer_cuda else 'cpu'
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.num_representatives = config.get("num_representatives", 60)
        self.num_candidates = config.get("num_candidates", 20)
        self.num_samples = config.get("num_samples", 20)
        self.batch_size = config.get("batch_size")

        self.num_reps = 0
        self.candidates = None
        self.counts = [0 for _ in range(model.num_classes)]

        self.representatives_x = [None for _ in range(model.num_classes)]
        self.representatives_y = None
        self.representatives_w = None

        self.mask = torch.as_tensor(
            [0.0 for _ in range(self.model.num_classes)],
            device=torch.device(get_device(self.cuda)),
        )

        self.acc_get_time = 0
        self.acc_cat_time = 0
        self.acc_acc_time = 0

    def init_candidates(self, size):
        size.insert(0, 0)
        self.candidates = [torch.empty(*size, device=self.device)
                           for _ in range(self.model.num_classes)]

    def get_samples(self):
        """Select or retrieve the representatives from the data

        :return: a list of num_samples representatives.
        """
        if self.num_reps == 0:
            return [], [], []

        repr_list = torch.randperm(self.num_reps)
        while len(repr_list) < self.num_samples:
            repr_list = torch.cat((repr_list, torch.randperm(self.num_reps)))
        repr_list = repr_list[: self.num_samples]

        # Accumulated sum of representative list lengths
        def len_cumsum():
            counts = np.clip(copy.deepcopy(self.counts),
                             0, self.num_representatives)
            return list(itertools.accumulate(counts))

        # Find 2D index from accumulated list of lengths
        def find_2d_idx(c, idx):
            i1 = bisect(c, idx)
            i2 = (idx - c[i1 - 1]) if i1 > 0 else idx
            return (i1, i2)
        idx_2d = [find_2d_idx(len_cumsum(), i.item()) for i in repr_list]

        return (
            torch.stack([self.representatives_x[i1][i2] for i1, i2 in idx_2d]),
            self.representatives_y[repr_list],
            self.representatives_w[repr_list],
        )

    def get_num_representatives(self):
        return self.num_reps

    def get_memory_size(self):
        for reps in self.representatives_x:
            if reps is not None:
                for rep in reps:
                    return rep.element_size() * rep.nelement() * self.num_reps
        return -1

    def accumulate(self, x, y):
        """Modify the representatives list by selecting candidates randomly from
        the incoming data x and the current list of representatives.

        In this version, we permute indices on r+c and we discard the c last
        samples.
        """
        self.init_candidates(list(x.size())[1:])

        previous_counts = np.clip(copy.deepcopy(self.counts),
                                  0, self.num_representatives)

        i = min(self.num_candidates, len(x))
        rand_indices = torch.randperm(len(y))
        x = x.clone()[rand_indices][:i]
        y = y.clone()[rand_indices][:i]
        labels, c = y.unique(return_counts=True)
        for i in range(len(labels)):
            self.counts[labels[i]] += c[i].item()
            self.candidates[labels[i]] = x[(
                y == labels[i]).nonzero(as_tuple=True)[0]]
        new_reps = torch.cat(self.candidates)

        offsets = [0]
        for i in range(self.model.num_classes):
            lower = offsets[-1]
            upper = lower + len(self.candidates[i])
            offsets.append(upper)
            if lower == upper:
                continue

            rand_indices = torch.randperm(
                previous_counts[i] + len(self.candidates[i])
            )[: self.num_representatives]
            rm = [
                j for j in range(previous_counts[i])
                if j not in rand_indices
            ]
            add = [
                j for j in range(len(self.candidates[i]))
                if j + previous_counts[i] in rand_indices
            ]

            if previous_counts[i] == 0 and len(add) > 0:
                size = list(new_reps[lower:upper][add][0].size())
                size.insert(0, self.num_representatives)
                self.representatives_x[i] = torch.empty(*size)

            # If not enough samples stored in the buffer, mark the next ones as
            # "to be removed"
            if len(rm) < len(add):
                rm = [j for j in range(previous_counts[i], min(
                    self.counts[i], self.num_representatives))]

            for r, a in zip(rm, add):
                self.representatives_x[i][r] = new_reps[lower:upper][a]

        self.representatives_y = torch.tensor(
            [
                i
                for i in range(self.model.num_classes)
                for _ in range(min(self.counts[i], self.num_representatives))
            ]
        )
        self.num_reps = len(self.representatives_y)

        self.recalculate_weights()

    def recalculate_weights(self):
        """Reassign the weights of the representatives
        """
        total_count = np.sum(self.counts)
        # This version proposes that the total weight of representatives is
        # calculated from the proportion of candidates with respect to the batch.
        # E.g. a batch of 100 images and 10 are num_candidates selected,
        # total_weight = 10
        total_weight = (self.batch_size * 1.0) / self.num_candidates
        # The total_weight is adjusted to the proportion between candidates
        # and representatives.
        total_weight *= total_count / len(self.representatives_y)
        probs = self.counts / total_count
        ws = []
        for y in self.representatives_y.unique():
            y = y.item()
            w = max(math.log(probs[y].item() * total_weight), 1.0)
            ws.append((self.representatives_y == y).float() * w)
        self.representatives_w = torch.sum(torch.stack(ws), 0)

    def before_every_task(self, task_id, train_data_regime):
        self.steps = 0

        # Distribute the data
        torch.cuda.nvtx.range_push("Distribute dataset")
        train_data_regime.get_loader(True)
        torch.cuda.nvtx.range_pop()

        if self.best_model is not None:
            logging.debug(
                f"Loading best model with minimal eval loss ({self.minimal_eval_loss}).."
            )
            self.model.load_state_dict(self.best_model)
            self.minimal_eval_loss = float("inf")

        if task_id > 0:
            if self.config.get("reset_state_dict", False):
                logging.debug("Resetting model internal state..")
                self.model.load_state_dict(
                    copy.deepcopy(self.initial_snapshot))
            self.optimizer.reset(self.model.parameters())

        # Add the new classes to the mask
        torch.cuda.nvtx.range_push("Create mask")
        nc = set([data[1] for data in train_data_regime.get_data()])
        for y in nc:
            self.mask[y] = 1.0
        torch.cuda.nvtx.range_pop()

        self.criterion = nn.CrossEntropyLoss(
            weight=self.mask, reduction="none")

    """
    Forward pass for the current epoch
    """

    def loop(self, data_regime, average_output=False, training=False):
        prefix = "train" if training else "val"
        meters = {
            metric: AverageMeter(f"{prefix}_{metric}")
            for metric in ["loss", "prec1", "prec5"]
        }
        start = time.time()
        step_count = 0

        for i_batch, (x, y, t) in enumerate(data_regime.get_loader()):
            torch.cuda.nvtx.range_push(f"Batch {i_batch}")

            output, loss = self._step(
                i_batch, x, y, training=training, average_output=average_output
            )

            # measure accuracy and record loss
            prec1, prec5 = accuracy(
                output[: y.size(0)],
                y,
                topk=(1, min(self.model.num_classes, 5)),
            )
            meters["loss"].update(loss)
            meters["prec1"].update(prec1, x.size(0))
            meters["prec5"].update(prec5, x.size(0))

            if i_batch % self.log_interval == 0 or i_batch == len(
                data_regime.get_loader()
            ):
                logging.info(
                    "{phase}: epoch: {0} [{1}/{2}]\t"
                    "Loss {meters[loss].val:.4f} ({meters[loss].avg:.4f})\t"
                    "Prec@1 {meters[prec1].val:.3f} ({meters[prec1].avg:.3f})\t"
                    "Prec@5 {meters[prec5].val:.3f} ({meters[prec5].avg:.3f})\t".format(
                        self.epoch + 1,
                        i_batch,
                        len(data_regime.get_loader()),
                        phase="TRAINING" if training else "EVALUATING",
                        meters=meters,
                    )
                )

                if self.writer is not None:
                    self.writer.add_scalar(
                        f"{prefix}_loss", meters["loss"].avg, self.global_steps
                    )
                    self.writer.add_scalar(
                        f"{prefix}_prec1",
                        meters["prec1"].avg,
                        self.global_steps,
                    )
                    self.writer.add_scalar(
                        f"{prefix}_prec5",
                        meters["prec5"].avg,
                        self.global_steps,
                    )
                    if training:
                        self.writer.add_scalar(
                            "lr", self.optimizer.get_lr()[0], self.global_steps
                        )
                        self.writer.add_scalar(
                            "num_representatives",
                            self.get_num_representatives(),
                            self.global_steps,
                        )
                        self.writer.add_scalar(
                            "memory_size",
                            self.get_memory_size(),
                            self.global_steps,
                        )
                    self.writer.flush()
                if self.watcher is not None:
                    self.observe(
                        trainer=self,
                        model=self.model,
                        optimizer=self.optimizer,
                        data=(x, y),
                    )
                    self.stream_meters(meters, prefix=prefix)
                    if training:
                        self.write_stream(
                            "lr",
                            (self.global_steps, self.optimizer.get_lr()[0]),
                        )
            torch.cuda.nvtx.range_pop()
            step_count += 1
        end = time.time()

        logging.info(f"epoch time {end - start}")
        logging.info(f"\tnum_representatives {self.get_num_representatives()}")
        logging.info(f"\tget time {self.acc_get_time}")
        logging.info(f"\tcat time {self.acc_cat_time}")
        logging.info(f"\tacc time {self.acc_acc_time}")
        self.acc_get_time = 0
        self.acc_cat_time = 0
        self.acc_acc_time = 0

        meters = {name: meter.avg.item() for name, meter in meters.items()}
        meters["error1"] = 100.0 - meters["prec1"]
        meters["error5"] = 100.0 - meters["prec5"]
        meters["time"] = end - start
        meters["step_count"] = step_count

        return meters

    def _step(
        self,
        i_batch,
        inputs_batch,
        target_batch,
        training=False,
        average_output=False,
        chunk_batch=1,
    ):
        outputs = []
        total_loss = 0

        if training:
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.steps)

        for i, (x, y) in enumerate(
            zip(
                inputs_batch.chunk(chunk_batch, dim=0),
                target_batch.chunk(chunk_batch, dim=0),
            )
        ):
            torch.cuda.nvtx.range_push(f"Chunk {i}")

            if training:
                # Get the representatives
                start_get_time = time.time()
                (
                    rep_values,
                    rep_labels,
                    rep_weights,
                ) = self.get_samples()
                self.acc_get_time += time.time() - start_get_time
                num_reps = len(rep_values)
                if self.writer is not None and self.writer_images and num_reps > 0:
                    fig = plot_candidates(rep_values, rep_labels, 5)
                    self.writer.add_figure(
                        "candidates", fig, self.global_steps)

            # Create batch weights
            w = torch.ones(len(x), device=torch.device(get_device(self.cuda)))
            torch.cuda.nvtx.range_push("Copy to device")
            x, y = move_cuda(x, self.cuda), move_cuda(y, self.cuda)
            torch.cuda.nvtx.range_pop()

            if training and num_reps > 0:
                torch.cuda.nvtx.range_push("Combine batches")
                rep_values, rep_labels, rep_weights = (
                    move_cuda(rep_values, self.cuda),
                    move_cuda(rep_labels, self.cuda),
                    move_cuda(rep_weights, self.cuda),
                )
                # Concatenate the training samples with the representatives
                start_cat_time = time.time()
                w = torch.cat((w, rep_weights))
                x = torch.cat((x, rep_values))
                y = torch.cat((y, rep_labels))
                self.acc_cat_time += time.time() - start_cat_time
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Forward pass")
            output = self.model(x)
            if training:
                loss = self.criterion(output, y)
            else:
                loss = nn.CrossEntropyLoss()(output, y)
            torch.cuda.nvtx.range_pop()

            if training:
                # Leads to decreased accuracy
                # total_weight = hvd.allreduce(torch.sum(w), name='total_weight', op=hvd.Sum)
                dw = w / torch.sum(w)
                # Faster to provide the derivative of L wrt {l}^n than letting
                # pytorch computing it by itself
                loss.backward(dw)
                # SGD step
                torch.cuda.nvtx.range_push("Optimizer step")
                self.optimizer.step()
                torch.cuda.nvtx.range_pop()
                self.global_steps += 1
                self.steps += 1

                start_acc_time = time.time()
                if num_reps == 0:
                    if self.buffer_cuda:
                        self.accumulate(x, y)
                    else:
                        self.accumulate(x.cpu(), y.cpu())
                else:
                    if self.buffer_cuda:
                        self.accumulate(x[:-num_reps], y[:-num_reps])
                    else:
                        self.accumulate(
                            x[:-num_reps].cpu(), y[:-num_reps].cpu())
                self.acc_acc_time += time.time() - start_acc_time

            outputs.append(output.detach())
            total_loss += torch.mean(loss).item()

            torch.cuda.nvtx.range_pop()

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss


def nil(model, config, optimizer, criterion, cuda, buffer_cuda, log_interval):
    implementation = config.get("implementation", "")
    agent = nil_agent
    if implementation == "global":
        agent = nil_global_agent
    return agent(model, config, optimizer, criterion, cuda, buffer_cuda, log_interval)
