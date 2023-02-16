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
from torch.cuda.amp import GradScaler, autocast
from utils.utils import get_device, move_cuda, plot_representatives, synchronize_cuda, find_2d_idx
from utils.meters import AverageMeter, accuracy


class nil_agent(Agent):
    def __init__(
        self,
        model,
        use_amp,
        config,
        optimizer_regime,
        batch_size,
        cuda,
        log_level,
        log_buffer,
        log_interval,
        batch_metrics=None,
        state_dict=None,
    ):
        super(nil_agent, self).__init__(
            model,
            use_amp,
            config,
            optimizer_regime,
            batch_size,
            cuda,
            log_level,
            log_buffer,
            log_interval,
            batch_metrics,
            state_dict,
        )

        self.device = "cuda" if self.buffer_cuda else 'cpu'
        self.rehearsal_size = config.get("rehearsal_size", 60)
        self.num_candidates = config.get("num_candidates", 20)
        self.num_representatives = config.get("num_representatives", 20)

        self.num_reps = 0
        self.highest_task_so_far = 0

        self.epoch_load_time = 0
        self.epoch_move_time = 0
        self.epoch_wait_time = 0
        self.epoch_cat_time = 0
        self.epoch_acc_time = 0
        self.last_batch_time = 0
        self.last_batch_load_time = 0
        self.last_batch_move_time = 0
        self.last_batch_wait_time = 0
        self.last_batch_cat_time = 0
        self.last_batch_acc_time = 0

    def before_all_tasks(self, train_data_regime):
        super().before_all_tasks(train_data_regime)
        self.num_classes = train_data_regime.total_num_classes
        self.counts = torch.zeros(self.num_classes, dtype=int)

        self.global_representatives_x = [
            [torch.Tensor() for _ in range(self.num_classes)]
            for _ in range(hvd.size())
        ]
        self.global_representatives_y = None
        self.global_representatives_w = None
        self.global_counts = None

    def before_every_task(self, task_id, train_data_regime):
        self.task_id = task_id
        self.batch = 0
        if task_id > self.highest_task_so_far:
            self.highest_task_so_far = task_id

        # Distribute the data
        train_data_regime.get_loader(True)

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
        
        # Create mask so the loss is only used for classes learnt during this task
        self.mask = torch.tensor(train_data_regime.classes_mask, device=self.device).float()
        self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')

    def get_samples(self):
        """Select or retrieves the representatives from the data

        :return: a list of num_representatives representatives.
        """
        if self.num_reps == 0:
            return [], [], []

        # self.num_reps counts the number of representatives stored in all
        # workers. This values is bounded and is a factor of num_workers *
        # rehearsal_size.
        # repr_list is a flat list of indices targeting all representatives of
        # all workers.
        repr_list = torch.randperm(self.num_reps)
        while len(repr_list) < self.num_representatives:
            repr_list = torch.cat((repr_list, torch.randperm(self.num_reps)))
        repr_list = repr_list[: self.num_representatives]

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
            global_counts, self.rehearsal_size * self.num_classes)

        # [(worker, representative count)]
        idx_worker = [find_2d_idx(accumulated_counts, i.item())
                      for i in repr_list]
        idx_3d = []
        for w, i in idx_worker:
            idx_2d = find_2d_idx(len_cumsum(
                self.global_counts[w], self.rehearsal_size), i)
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
                      for _ in range(self.num_classes)]

        previous_counts = copy.deepcopy(self.counts)
        for i in range(len(previous_counts)):
            previous_counts[i] = min(
                previous_counts[i], self.rehearsal_size)

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
        for label in range(self.num_classes):
            offsets.append(offsets[-1] + len(candidates[label]))

            representatives_count = previous_counts[label]
            rand_indices = torch.randperm(
                representatives_count + len(candidates[label])
            )[: self.rehearsal_size]
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
                if representatives_count+j < min(self.counts[label], self.rehearsal_size):
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

            for label in range(self.num_classes):
                lower = offsets[label]
                upper = offsets[label + 1]
                if lower == upper:
                    continue

                rm = rm_targets[lower:upper]
                add = add_targets[lower:upper]

                if len(self.global_representatives_x[w][label]) == 0 and len(add) > 0:
                    size = list(candidates[lower:upper][0].size())
                    size.insert(0, self.rehearsal_size)
                    self.global_representatives_x[w][label] = torch.empty(
                        *size)

                for r, a in zip(rm, add):
                    self.global_representatives_x[w][label][r] = candidates[lower:upper][a]

        self.global_representatives_y = hvd.allgather(torch.tensor(
            [
                i
                for i in range(self.num_classes)
                for _ in range(min(self.counts[i], self.rehearsal_size))
            ]
        ).unsqueeze(0))
        self.num_reps = self.global_representatives_y.numel()

    def recalculate_weights(self):
        """Reassign the weights of the representatives
        """
        total_count = torch.sum(self.global_counts, dim=1).sum().item()

        # This version proposes that the total weight of representatives is
        # calculated from the proportion of samples to augment the batch with.
        # E.g. a batch of 100 images and 10 are num_representatives selected,
        # total_weight = 10
        total_weight = (self.batch_size * 1.0) / \
            (self.num_representatives * len(self.global_representatives_y))
        # The total_weight is adjusted to the proportion between candidates
        # and actual representatives.
        ws = []
        for y in self.global_representatives_y.unique():
            y = y.item()
            weight = max(math.log(self.global_counts.sum(
                dim=0)[y].item() * total_weight), 1.0)
            ws.append((self.global_representatives_y == y).float() * weight)
        self.global_representatives_w = torch.sum(torch.stack(ws), 0)

    """
    Forward pass for the current epoch
    """
    def loop(self, data_regime, training=False):
        prefix = "train" if training else "val"
        meters = {
            metric: AverageMeter(f"{prefix}_{metric}")
            for metric in ["loss", "prec1", "prec5"]
        }
        epoch_time = 0

        start_batch_time = time.time()
        start_load_time = start_batch_time

        with tqdm(total=len(data_regime.get_loader()),
              desc=f"Task #{self.task_id + 1} {prefix} epoch #{self.epoch + 1}",
              disable=self.log_level not in ('info')) as progress:
            for i_batch, (x, y, t) in enumerate(data_regime.get_loader()):
                synchronize_cuda(self.cuda)
                self.last_batch_load_time = time.time() - start_load_time
                self.epoch_load_time += self.last_batch_load_time

                self._step(i_batch, x, y, meters, training=training)

                synchronize_cuda(self.cuda)
                self.last_batch_time = time.time() - start_batch_time
                epoch_time += self.last_batch_time

                if i_batch % self.log_interval == 0 or i_batch == len(
                    data_regime.get_loader()
                ):
                    logging.debug(
                        "{0}: epoch: {1} [{2}/{3}]\t"
                        "Loss {meters[loss].avg:.4f}\t"
                        "Prec@1 {meters[prec1].avg:.3f}\t"
                        "Prec@5 {meters[prec5].avg:.3f}\t".format(
                            prefix,
                            self.epoch + 1,
                            i_batch,
                            len(data_regime.get_loader()),
                            meters=meters,
                        )
                    )

                    logging.debug(f"batch {i_batch} time {self.last_batch_time} sec")
                    logging.debug(
                        f"\tbatch load time {self.last_batch_load_time} sec ({self.last_batch_load_time*100/self.last_batch_time}%)")
                    logging.debug(
                        f"\tbatch move time {self.last_batch_move_time} sec ({self.last_batch_move_time*100/self.last_batch_time}%)")
                    logging.debug(
                        f"\tbatch wait time {self.last_batch_wait_time} sec ({self.last_batch_wait_time*100/self.last_batch_time}%)")
                    logging.debug(
                        f"\tbatch cat time {self.last_batch_cat_time} sec ({self.last_batch_cat_time*100/self.last_batch_time}%)")
                    logging.debug(
                        f"\tbatch acc time {self.last_batch_acc_time} sec ({self.last_batch_acc_time*100/self.last_batch_time}%)")
                    logging.debug(
                        f"\trehearsal_size {self.get_rehearsal_size()}")

                    if hvd.rank() == 0:
                        if training and self.epoch < 5 and self.batch_metrics is not None:
                            batch_metrics_values = dict(
                                epoch=self.epoch,
                                batch=i_batch,
                                time=self.last_batch_time,
                                load_time=self.last_batch_load_time,
                                move_time=self.last_batch_move_time,
                                wait_time=self.last_batch_wait_time,
                                cat_time=self.last_batch_cat_time,
                                acc_time=self.last_batch_acc_time,
                                num_reps=self.num_reps,
                            )
                            self.batch_metrics.add(**batch_metrics_values)
                            self.batch_metrics.save()

                        if self.task_id == self.highest_task_so_far:
                            wandb.log({f"{prefix}_loss": meters["loss"].avg,
                                    "batch": self.global_batch,
                                    "epoch": self.global_epoch,
                                    "batch": i_batch,
                                    f"{prefix}_prec1": meters["prec1"].avg,
                                    f"{prefix}_prec5": meters["prec5"].avg})
                            if training:
                                wandb.log({"lr": self.optimizer_regime.get_lr()[0],
                                        "batch": self.global_batch,
                                        "epoch": self.global_epoch})
                    """
                    if self.writer is not None:
                        self.writer.add_scalar(
                            f"{prefix}_loss", meters["loss"].avg, self.global_batch
                        )
                        self.writer.add_scalar(
                            f"{prefix}_prec1",
                            meters["prec1"].avg,
                            self.global_batch,
                        )
                        self.writer.add_scalar(
                            f"{prefix}_prec5",
                            meters["prec5"].avg,
                            self.global_batch,
                        )
                        if training:
                            self.writer.add_scalar(
                                "lr", self.optimizer_regime.get_lr()[
                                    0], self.global_batch
                            )
                            self.writer.add_scalar(
                                "rehearsal_size",
                                self.get_rehearsal_size(),
                                self.global_batch,
                            )
                            self.writer.add_scalar(
                                "rehearsal_size",
                                self.get_rehearsal_size(),
                                self.global_batch,
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
                                (self.global_batch,
                                self.optimizer_regime.get_lr()[0]),
                            )
                    """
                progress.set_postfix({'loss': meters["loss"].avg.item(),
                            'accuracy': meters["prec1"].avg.item(),
                            'representatives': self.num_reps})
                progress.update(1)

                if self.task_id == self.highest_task_so_far:
                    self.global_batch += 1
                    self.batch += 1
        
                synchronize_cuda(self.cuda)
                start_batch_time = time.time()
                start_load_time = start_batch_time

        if training:
            self.global_epoch += 1
            logging.info(f"\nCUMULATED VALUES:")
            logging.info(
                f"\trehearsal_size {self.get_rehearsal_size()}")
            logging.info(f"epoch time {epoch_time} sec")
            logging.info(
                f"\tepoch load time {self.epoch_load_time} sec ({self.epoch_load_time*100/epoch_time}%)")
            logging.info(
                f"\tepoch move time {self.epoch_move_time} sec ({self.epoch_move_time*100/epoch_time}%)")
            logging.info(
                f"\tepoch wait time {self.epoch_wait_time} sec ({self.epoch_wait_time*100/epoch_time}%)")
            logging.info(
                f"\tepoch cat time {self.epoch_cat_time} sec ({self.epoch_cat_time*100/epoch_time}%)")
            logging.info(
                f"\tepoch acc time {self.epoch_acc_time} sec ({self.epoch_acc_time*100/epoch_time}%)")
        self.epoch_load_time = 0
        self.epoch_move_time = 0
        self.epoch_wait_time = 0
        self.epoch_cat_time = 0
        self.epoch_acc_time = 0

        meters = {name: meter.avg.item() for name, meter in meters.items()}
        meters["error1"] = 100.0 - meters["prec1"]
        meters["error5"] = 100.0 - meters["prec5"]
        meters["time"] = epoch_time
        meters["batch"] = self.batch

        return meters

    def _step(
        self,
        i_batch,
        x,
        y,
        meters,
        training=False,
    ):
        if training:
            self.optimizer_regime.zero_grad()
            self.optimizer_regime.update(self.epoch, self.batch)

        w = torch.ones(len(x), device=torch.device(get_device(self.cuda)))
        start_move_time = time.time()
        x, y = move_cuda(x, self.cuda and self.buffer_cuda), move_cuda(
            y, self.cuda and self.buffer_cuda)
        self.last_batch_move_time = time.time() - start_move_time
        self.epoch_move_time += self.last_batch_move_time

        if training:
            start_acc_time = time.time()
            if self.buffer_cuda:
                self.accumulate(x, y)
            else:
                self.accumulate(x.cpu(), y.cpu())
            self.last_batch_acc_time = time.time() - start_acc_time
            self.epoch_acc_time += self.last_batch_acc_time

            # Get the representatives
            start_wait_time = time.time()
            (
                rep_values,
                rep_labels,
                rep_weights,
            ) = self.get_samples()
            self.last_batch_wait_time = time.time() - start_wait_time
            self.epoch_wait_time += self.last_batch_wait_time
            num_reps = len(rep_values)

            if num_reps > 0:
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
                self.last_batch_cat_time = time.time() - start_cat_time
                self.epoch_cat_time += self.last_batch_cat_time

            # Log representatives
            if self.writer is not None and self.writer_images and num_reps > 0:
                fig = plot_representatives(rep_values, rep_labels, 5)
                self.writer.add_figure(
                    "representatives", fig, self.global_batch)

        if self.use_amp:
            with autocast():
                output = self.model(x)
                loss = self.criterion(output, y)
        else:
            output = self.model(x)
            loss = self.criterion(output, y)

        if training:
            total_weight = hvd.allreduce(torch.sum(w), name='total_weight', op=hvd.Sum)
            dw = w / total_weight

            if self.use_amp:
                self.scaler.scale(loss).backward(dw)
                self.optimizer_regime.optimizer.synchronize()
                with self.optimizer_regime.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer_regime.optimizer)
                    self.scaler.update()
            else:
                loss.backward(dw)
                self.optimizer_regime.step()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output[: y.size(0)], y, topk=(1, 5))
            meters["loss"].update(loss.sum() / self.mask[y].sum(), x.size(0))
            meters["prec1"].update(prec1, x.size(0))
            meters["prec5"].update(prec5, x.size(0))

    def get_rehearsal_size(self):
        return self.num_reps

    def get_memory_size(self):
        return sum(x.element_size() * x.numel() for x in self.representatives)
