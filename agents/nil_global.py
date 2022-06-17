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

from apex import amp

from agents.base import Agent
from utils.utils import get_device, move_cuda, plot_representatives, find_2d_idx
from utils.meters import AverageMeter, accuracy


class nil_global_agent(Agent):
    def __init__(
        self,
        model,
        use_amp,
        config,
        optimizer_regime,
        criterion,
        cuda,
        buffer_cuda,
        log_interval,
        state_dict=None,
    ):
        super(nil_global_agent, self).__init__(
            model,
            use_amp,
            config,
            optimizer_regime,
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
        self.counts = torch.tensor([0 for _ in range(model.num_classes)])

        self.global_representatives_x = [
            [torch.Tensor() for _ in range(model.num_classes)]
            for _ in range(hvd.size())
        ]
        self.global_representatives_y = None
        self.global_representatives_w = None
        self.global_counts = None

        self.mask = torch.as_tensor(
            [0.0 for _ in range(self.model.num_classes)],
            device=torch.device(get_device(self.cuda)),
        )

        self.epoch_acc_get_time = 0
        self.epoch_acc_cat_time = 0
        self.epoch_acc_acc_time = 0

    def get_samples(self):
        """Select or retrieves the representatives from the data

        :return: a list of num_samples representatives.
        """
        if self.num_reps == 0:
            return [], [], []

        # self.num_reps counts the number of representatives stored in all
        # workers. This values is bounded and is a factor of num_workers *
        # num_representatives.
        # repr_list is a flat list of indices targeting all representatives of
        # all workers.
        repr_list = torch.randperm(self.num_reps)
        while len(repr_list) < self.num_samples:
            repr_list = torch.cat((repr_list, torch.randperm(self.num_reps)))
        repr_list = repr_list[: self.num_samples]

        # Accumulated sum of representative list lengths
        def len_cumsum(counts, max_clip):
            clipped_counts = torch.clamp(counts, min=0, max=max_clip)
            return list(itertools.accumulate(clipped_counts.tolist()))

        # self.global_counts has shape torch.Size([w, num_classes]) and counts
        # the number of representatives per class.
        # global_counts is a flatten list containing the number of
        # representatives for each worker. Indices are ranks.
        global_counts = torch.sum(self.global_counts, dim=1)
        accumulated_counts = len_cumsum(
            global_counts, self.num_representatives*self.model.num_classes)

        # [(worker, representative count)]
        idx_worker = [find_2d_idx(accumulated_counts, i.item())
                      for i in repr_list]
        idx_3d = []
        for w, i in idx_worker:
            idx_2d = find_2d_idx(len_cumsum(
                self.global_counts[w], self.num_representatives), i)
            idx_3d.append((w, idx_2d[0], idx_2d[1]))

        return (
            torch.stack([self.global_representatives_x[i1][i2][i3]
                        for i1, i2, i3 in idx_3d]),
            self.global_representatives_y.view(-1)[repr_list].squeeze(0),
            self.global_representatives_w.view(-1)[repr_list].squeeze(0),
        )

    def accumulate(self, x, y):
        """Create a bufferbased in random sampling

        :param image_batch: the list of images of a batch
        :param target_batch: the list of one hot labels of a batch
        :param iteration: current iteration of training
        :param megabatch: current megabatch
        :return: None
        """
        size = list(x.size())[1:]
        size.insert(0, 0)
        candidates = [torch.empty(*size, device=self.device)
                      for _ in range(self.model.num_classes)]

        previous_counts = copy.deepcopy(self.counts)
        for i in range(len(previous_counts)):
            previous_counts[i] = min(
                previous_counts[i], self.num_representatives)

        i = min(self.num_candidates, len(x))
        rand_candidates = torch.randperm(len(y))
        x = x.clone()[rand_candidates][:i]
        y = y.clone()[rand_candidates][:i]
        labels, c = y.unique(return_counts=True)

        for i in range(len(labels)):
            label = labels[i].item()
            self.counts[label] += c[i].item()
            candidates[label] = x[(y == label).nonzero(as_tuple=True)[0]]

        # For the current worker, for each class, we want:
        #   - a list of candidates (cumulated sum is num_candidates)
        #   - indices to add in that list
        #   - indices to rm in representatives
        rm_targets = []
        add_targets = []
        offsets = [0]
        for label in range(self.model.num_classes):
            offsets.append(offsets[-1] + len(candidates[label]))

            representatives_count = previous_counts[label]
            rand_indices = torch.randperm(
                representatives_count + len(candidates[label])
            )[: self.num_representatives]
            rm_targets += [
                j for j in range(representatives_count)
                if j not in rand_indices
            ]
            add_targets += [
                j for j in range(len(candidates[label]))
                if j + representatives_count in rand_indices
            ]

            # If not the buffer for current label is not full yet, mark the next
            # ones as "to be removed"
            for j in range(max(0, len(add_targets) - len(rm_targets) + 1)):
                if representatives_count+j < min(self.counts[label], self.num_representatives):
                    rm_targets += [representatives_count+j]
            if len(rm_targets) > len(add_targets):
                rm_targets = rm_targets[:len(add_targets)]
            assert len(rm_targets) == len(
                add_targets), f"{len(rm_targets)} == {len(add_targets)}"

        while len(rm_targets) < self.num_candidates:
            rm_targets.append(-1)
        while len(add_targets) < self.num_candidates:
            add_targets.append(-1)

        self.share_representatives(
            candidates, offsets, rm_targets, add_targets)
        self.recalculate_weights()

    def share_representatives(self, candidates, offsets, rm_targets, add_targets):
        #torch.cuda.nvtx.range_push("Share representatives")

        self.global_counts = hvd.allgather(self.counts.unsqueeze(0))
        global_candidates = hvd.allgather(torch.cat(candidates).unsqueeze(0))
        global_offsets = hvd.allgather(torch.tensor([offsets]))
        global_rm_targets = hvd.allgather(torch.tensor([rm_targets]))
        global_add_targets = hvd.allgather(torch.tensor([add_targets]))

        for w in range(hvd.size()):
            candidates = global_candidates[w]
            offsets = global_offsets[w]
            rm_targets = global_rm_targets[w]
            rm_targets = rm_targets[rm_targets != -1]
            add_targets = global_add_targets[w]
            add_targets = add_targets[add_targets != -1]

            for label in range(self.model.num_classes):
                lower = offsets[label]
                upper = offsets[label + 1]
                if lower == upper:
                    continue

                rm = rm_targets[lower:upper]
                add = add_targets[lower:upper]

                if len(self.global_representatives_x[w][label]) == 0 and len(add) > 0:
                    size = list(candidates[lower:upper][0].size())
                    size.insert(0, self.num_representatives)
                    self.global_representatives_x[w][label] = torch.empty(
                        *size)

                for r, a in zip(rm, add):
                    self.global_representatives_x[w][label][r] = candidates[lower:upper][a]

        self.global_representatives_y = hvd.allgather(torch.tensor(
            [
                i
                for i in range(self.model.num_classes)
                for _ in range(min(self.counts[i], self.num_representatives))
            ]
        ).unsqueeze(0))
        self.num_reps = self.global_representatives_y.numel()
        # torch.cuda.nvtx.range_pop()

    def recalculate_weights(self):
        """Reassign the weights of the representatives
        """
        total_count = torch.sum(self.global_counts, dim=1).sum().item()

        # This version proposes that the total weight of representatives is
        # calculated from the proportion of samples to augment the batch with.
        # E.g. a batch of 100 images and 10 are num_samples selected,
        # total_weight = 10
        total_weight = (self.batch_size * 1.0) / \
            (self.num_samples * len(self.global_representatives_y))
        # The total_weight is adjusted to the proportion between candidates
        # and actual representatives.
        ws = []
        for y in self.global_representatives_y.unique():
            y = y.item()
            weight = max(math.log(self.global_counts.sum(
                dim=0)[y].item() * total_weight), 1.0)
            ws.append((self.global_representatives_y == y).float() * weight)
        self.global_representatives_w = torch.sum(torch.stack(ws), 0)

    def before_every_task(self, task_id, train_data_regime):
        self.steps = 0

        # Distribute the data
        #torch.cuda.nvtx.range_push("Distribute dataset")
        train_data_regime.get_loader(True)
        # torch.cuda.nvtx.range_pop()

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
            self.optimizer_regime.reset(self.model.parameters())

        # Add the new classes to the mask
        #torch.cuda.nvtx.range_push("Create mask")
        nc = set([data[1] for data in train_data_regime.get_data()])
        for y in nc:
            self.mask[y] = 1.0
        # torch.cuda.nvtx.range_pop()

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
            #torch.cuda.nvtx.range_push(f"Batch {i_batch}")

            batch_start = time.time()
            output, loss = self._step(
                i_batch, x, y, training=training, average_output=average_output
            )
            batch_end = time.time()

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
                logging.info(f"Time taken for batch {i_batch} is {batch_end - batch_start} sec")

                if hvd.rank() == 0 and hvd.local_rank() == 0:
                    wandb.log({f"{prefix}_loss": meters["loss"].avg,
                               "step": self.global_steps,
                               "epoch": self.global_epoch,
                               "batch": i_batch,
                               f"{prefix}_prec1": meters["prec1"].avg,
                               f"{prefix}_prec5": meters["prec5"].avg})
                    if training:
                        wandb.log({"lr": self.optimizer_regime.get_lr()[0],
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
                            "lr", self.optimizer_regime.get_lr()[0], self.global_steps
                        )
                        self.writer.add_scalar(
                            "num_representatives",
                            self.get_num_representatives(),
                            self.global_steps,
                        )
                        self.writer.add_scalar(
                            "num_representatives",
                            self.get_num_representatives(),
                            self.global_steps,
                        )
                    self.writer.flush()
                if self.watcher is not None:
                    self.observe(
                        trainer=self,
                        model=self.model,
                        optimizer=self.optimizer_regime.optimizer,
                        data=(x, y),
                    )
                    self.stream_meters(meters, prefix=prefix)
                    if training:
                        self.write_stream(
                            "lr",
                            (self.global_steps, self.optimizer_regime.get_lr()[0]),
                        )
            # torch.cuda.nvtx.range_pop()
            step_count += 1
        end = time.time()

        if training:
            self.global_epoch += 1

        logging.info(f"epoch time {end - start}")
        logging.info(f"\tnum_representatives {self.get_num_representatives()}")
        logging.info(f"\tepoch get time {self.epoch_acc_get_time}")
        logging.info(f"\tepoch cat time {self.epoch_acc_cat_time}")
        logging.info(f"\tepoch acc time {self.epoch_acc_acc_time}")
        self.epoch_acc_get_time = 0
        self.epoch_acc_cat_time = 0
        self.epoch_acc_acc_time = 0

        meters = {name: meter.avg.item() for name, meter in meters.items()}
        meters["error1"] = 100.0 - meters["prec1"]
        meters["error5"] = 100.0 - meters["prec5"]
        meters["time"] = end - start
        meters["step_count"] = step_count

        return meters

    def _step(
        self,
        i_batch,
        inputs,
        target,
        training=False,
        average_output=False,
        chunk_batch=1,
    ):
        outputs = []
        total_loss = 0

        if training:
            self.optimizer_regime.zero_grad()
            self.optimizer_regime.update(self.epoch, self.steps)

        for i, (x, y) in enumerate(zip(inputs.chunk(chunk_batch, dim=0),
                                       target.chunk(chunk_batch, dim=0))):
            w = torch.ones(len(x), device=torch.device(get_device(self.cuda)))
            #torch.cuda.nvtx.range_push("Copy to device")
            x, y = move_cuda(x, self.cuda and self.buffer_cuda), move_cuda(
                y, self.cuda and self.buffer_cuda)            # torch.cuda.nvtx.range_pop()
            
            if training:
                start_acc_time = time.time()
                if self.buffer_cuda:
                    self.accumulate(x, y)
                else:
                    self.accumulate(x.cpu(), y.cpu())
                self.epoch_acc_acc_time += time.time() - start_acc_time

                # Get the representatives
                start_get_time = time.time()
                (
                    rep_values,
                    rep_labels,
                    rep_weights,
                ) = self.get_samples()
                self.epoch_acc_get_time += time.time() - start_get_time
                num_reps = len(rep_values)

                if num_reps > 0:
                    #torch.cuda.nvtx.range_push("Combine batches")
                    rep_values, rep_labels, rep_weights = (
                        move_cuda(rep_values, self.cuda),
                        move_cuda(rep_labels, self.cuda),
                        move_cuda(rep_weights, self.cuda),
                    )
                    # Concatenates the training samples with the representatives
                    start_cat_time = time.time()
                    w = torch.cat((w, rep_weights))
                    x = torch.cat((x, rep_values))
                    y = torch.cat((y, rep_labels))
                    self.epoch_acc_cat_time += time.time() - start_cat_time
                    # torch.cuda.nvtx.range_pop()

                # Log representatives
                if self.writer is not None and self.writer_images and num_reps > 0:
                    fig = plot_representatives(rep_values, rep_labels, 5)
                    self.writer.add_figure(
                        "representatives", fig, self.global_steps)

            #torch.cuda.nvtx.range_push("Forward pass")
            output = self.model(x)
            if training:
                loss = self.criterion(output, y)
            else:
                loss = nn.CrossEntropyLoss()(output, y)
            # torch.cuda.nvtx.range_pop()

            if training:
                total_weight = hvd.allreduce(
                    torch.sum(w), name="total_weight", op=hvd.Sum
                )
                dw = w / total_weight
                #torch.cuda.nvtx.range_push("Optimizer step")
                if self.use_amp:
                    with amp.scale_loss(loss, self.optimizer_regime.optimizer) as scaled_loss:
                        scaled_loss.backward(dw)
                        self.optimizer_regime.optimizer.synchronize()
                    with self.optimizer_regime.optimizer.skip_synchronize():
                        self.optimizer_regime.step()
                else:
                    # Faster to provide the derivative of L wrt {l}^n than letting
                    # pytorch computing it by itself
                    loss.backward(dw)
                    self.optimizer_regime.step()
                # torch.cuda.nvtx.range_pop()
                self.global_steps += 1
                self.steps += 1

            outputs.append(output.detach())
            total_loss += torch.mean(loss).item()

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss

    def get_num_representatives(self):
        return self.num_reps

    def get_memory_size(self):
        return sum(x.element_size() * x.numel() for x in self.representatives)
