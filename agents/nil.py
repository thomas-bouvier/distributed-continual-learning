import copy
import horovod.torch as hvd
import itertools
import math
import numpy as np
import logging
import time
import torch
import torch.nn as nn
import torchvision
import wandb

from agents.base import Agent
from agents.nil_global import nil_global_agent
from agents.nil_cpp import nil_cpp_agent
from utils.utils import get_device, move_cuda, plot_representatives, find_2d_idx
from utils.meters import AverageMeter, accuracy


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
        self.counts = {}

        self.representatives_x = {}
        self.representatives_y = None
        self.representatives_w = None

        self.mask = torch.as_tensor(
            [0.0 for _ in range(self.model.num_classes)],
            device=torch.device(get_device(self.cuda)),
        )

        self.acc_get_time = 0
        self.acc_cat_time = 0
        self.acc_acc_time = 0

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
            counts = copy.deepcopy(self.counts)
            all_classes = [0 for _ in range(max(counts)+1)]
            for k, v in counts.items():
                all_classes[k] = min(v, self.num_representatives)
            return list(itertools.accumulate(all_classes))

        idx_2d = [find_2d_idx(len_cumsum(), i.item()) for i in repr_list]
        return (
            torch.stack([self.representatives_x[i1][i2] for i1, i2 in idx_2d]),
            self.representatives_y[repr_list],
            self.representatives_w[repr_list],
        )

    def accumulate(self, x, y):
        """Modify the representatives list by selecting candidates randomly from
        the incoming data x and the current list of representatives.

        In this version, we permute indices on r+c and we discard the c last
        samples.
        """
        size = list(x.size())[1:]
        size.insert(0, 0)
        candidates = [torch.empty(*size, device=self.device)
                      for _ in range(self.model.num_classes)]

        previous_counts = copy.deepcopy(self.counts)
        for k, v in previous_counts.items():
            v = min(v, self.num_representatives)

        i = min(self.num_candidates, len(x))
        rand_candidates = torch.randperm(len(y))
        x = x.clone()[rand_candidates][:i]
        y = y.clone()[rand_candidates][:i]
        labels, c = y.unique(return_counts=True)

        for i in range(len(labels)):
            label = labels[i].item()
            self.counts[label] = self.counts.get(label, 0) + c[i].item()
            candidates = x[(y == label).nonzero(as_tuple=True)[0]]

            representatives_count = previous_counts.get(label, 0)
            rand_indices = torch.randperm(
                representatives_count + len(candidates)
            )[: self.num_representatives]
            rm = [
                j for j in range(representatives_count)
                if j not in rand_indices
            ]
            add = [
                j for j in range(len(candidates))
                if j + representatives_count in rand_indices
            ]

            # If not the buffer for current label is not full yet, mark the next
            # ones as "to be removed"
            for j in range(max(0, len(add) - len(rm) + 1)):
                if representatives_count+j < min(self.counts[label], self.num_representatives):
                    rm += [representatives_count+j]
            if len(rm) > len(add):
                rm = rm[:len(add)]
            assert len(rm) == len(add), f"{len(rm)} == {len(add)}"

            if representatives_count == 0 and len(add) > 0:
                size = list(candidates[add][0].size())
                size.insert(0, self.num_representatives)
                self.representatives_x[label] = torch.empty(*size)

            for r, a in zip(rm, add):
                self.representatives_x[label][r] = candidates[a]

        self.representatives_y = torch.tensor(
            [
                i
                for i in range(max(self.counts)+1)
                for _ in range(min(self.counts.get(i, 0), self.num_representatives))
            ]
        )
        self.num_reps = len(self.representatives_y)

        self.recalculate_weights()
        # for k, v in self.representatives_x.items():
        #    print(f"label {k} v {v.shape}")
        #    print(f"counts {self.counts[k]}")
        # input()

    def recalculate_weights(self):
        """Reassign the weights of the representatives
        """
        total_count = 0
        for i in range(max(self.counts)+1):
            total_count += self.counts.get(i, 0)

        # This version proposes that the total weight of representatives is
        # calculated from the proportion of samples to augment the batch with.
        # E.g. a batch of 100 images and 10 are num_samples selected,
        # weight = 10
        weight = (self.batch_size * 1.0) / \
            (self.num_samples * len(self.representatives_y))
        # The weight is adjusted to the proportion between historical candidates
        # and representatives.
        ws = []
        for y in self.representatives_y.unique():
            y = y.item()
            w = max(math.log(self.counts[y] * weight), 1.0)
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
                        self.epoch,
                        i_batch,
                        len(data_regime.get_loader()),
                        phase="TRAINING" if training else "EVALUATING",
                        meters=meters,
                    )
                )

                if hvd.rank() == 0 and hvd.local_rank() == 0:
                    wandb.log({f"{prefix}_loss": meters["loss"].avg,
                        "step": self.global_steps,
                        "epoch": self.global_epoch,
                        "batch": i_batch,
                        f"{prefix}_prec1": meters["prec1"].avg,
                        f"{prefix}_prec5": meters["prec5"].avg})
                    if training:
                        wandb.log({"lr": self.optimizer.get_lr()[0],
                            "step": self.global_steps,
                            "epoch": self.global_epoch})
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

        if training:
            self.global_epoch += 1

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

            # Create batch weights
            w = torch.ones(len(x), device=torch.device(get_device(self.cuda)))
            torch.cuda.nvtx.range_push("Copy to device")
            x, y = move_cuda(x, self.cuda), move_cuda(y, self.cuda)
            torch.cuda.nvtx.range_pop()

            if training:
                start_acc_time = time.time()
                if self.buffer_cuda:
                    self.accumulate(x, y)
                else:
                    self.accumulate(x.cpu(), y.cpu())
                self.acc_acc_time += time.time() - start_acc_time

                # Get the representatives
                start_get_time = time.time()
                (
                    rep_values,
                    rep_labels,
                    rep_weights,
                ) = self.get_samples()
                self.acc_get_time += time.time() - start_get_time
                num_reps = len(rep_values)

                if num_reps > 0:
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

                # Log representatives
                if self.writer is not None and self.writer_images and num_reps > 0:
                    fig = plot_representatives(rep_values, rep_labels, 5)
                    self.writer.add_figure(
                        "representatives", fig, self.global_steps)

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

            outputs.append(output.detach())
            total_loss += torch.mean(loss).item()

            torch.cuda.nvtx.range_pop()

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss

    def get_num_representatives(self):
        return self.num_reps

    def get_memory_size(self):
        for label, reps in self.representatives_x.items():
            for rep in reps:
                return rep.element_size() * rep.numel() * self.num_reps
        return -1


def nil(model, config, optimizer, criterion, cuda, buffer_cuda, log_interval):
    implementation = config.get("implementation", "")
    agent = nil_agent
    if implementation == "global":
        agent = nil_global_agent
    elif implementation == "cpp":
        agent = nil_cpp_agent
    return agent(model, config, optimizer, criterion, cuda, buffer_cuda, log_interval)
