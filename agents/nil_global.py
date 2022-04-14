import copy
import horovod.torch as hvd
import math
import logging
import time
import torch
import torch.nn as nn
import torchvision

from agents.base import Agent
from utils.utils import get_device, move_cuda
from utils.meters import AverageMeter, accuracy


class nil_global_agent(Agent):
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
        super(nil_global_agent, self).__init__(
            model,
            config,
            optimizer,
            criterion,
            cuda,
            buffer_cuda,
            log_interval,
            state_dict,
        )

        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.num_representatives = config.get("num_representatives", 60)
        self.num_candidates = config.get("num_candidates", 20)
        self.batch_size = config.get("batch_size")

        self.num_reps = 0
        self.representatives = [torch.Tensor()
                                for _ in range(model.num_classes)]
        self.candidates = [torch.Tensor() for _ in range(model.num_classes)]
        self.reps_by_worker = [
            [torch.Tensor() for _ in range(model.num_classes)]
            for _ in range(hvd.size())
        ]
        self.class_count = torch.tensor([0 for _ in range(model.num_classes)])

        self.global_x_reps = None
        self.global_y_reps = None
        self.global_w_reps = None

        self.mask = torch.as_tensor(
            [0.0 for _ in range(self.model.num_classes)],
            device=torch.device(get_device(self.cuda)),
        )

    def get_representatives(self):
        """Select or retrieves the representatives from the data

        :return: a list of num_candidates representatives.
        """
        if self.num_reps == 0:
            return [], [], []

        repr_list = torch.randperm(self.num_reps)
        while len(repr_list) < self.num_candidates:
            repr_list += torch.randperm(self.num_reps)
        repr_list = repr_list[: self.num_candidates]

        return (
            self.global_x_reps[repr_list],
            self.global_y_reps[repr_list],
            self.global_w_reps[repr_list],
        )

    def get_num_representatives(self):
        return sum(x.size(dim=0) for x in self.representatives)

    def get_memory_size(self):
        return sum(x.element_size() * x.nelement() for x in self.representatives)

    def pick_candidates(self, x, y):
        """Create a bufferbased in random sampling

        :param image_batch: the list of images of a batch
        :param target_batch: the list of one hot labels of a batch
        :param iteration: current iteration of training
        :param megabatch: current megabatch
        :return: None
        """
        i = min(self.num_candidates, len(x))
        rand_indices = torch.randperm(len(y))
        x = x.clone()[rand_indices][:i]
        y = y.clone()[rand_indices][:i]
        labels, counts = y.unique(return_counts=True)
        for i in range(len(labels)):
            self.class_count[labels[i]] += counts[i]
            self.candidates[labels[i]] = x[(
                y == labels[i]).nonzero(as_tuple=True)[0]]

        rm = []
        add = []
        offsets = [0]
        for i in range(self.model.num_classes):
            offsets.append(offsets[-1] + len(self.candidates[i]))
            rand_indices = torch.randperm(
                len(self.representatives[i]) + len(self.candidates[i])
            )[: self.num_representatives]
            rm += [
                j + self.num_representatives * i
                for j in range(len(self.representatives[i]))
                if j not in rand_indices
            ]
            add += [
                j + self.num_candidates * i
                for j in range(len(self.candidates[i]))
                if j + len(self.representatives[i]) in rand_indices
            ]

        while len(add) < self.model.num_classes * self.num_candidates:
            add.append(-1)
        while len(rm) < self.model.num_classes * self.num_candidates:
            rm.append(-1)

        self.share_reps(rm, add, offsets)
        self.recalculate_weights()
        self.num_reps = len(self.global_y_reps)

    def share_reps(self, rm, add, offsets):
        """Share representatives accross workers"""
        if sum(self.class_count) == 0:
            return
        self.global_class_count = hvd.allreduce(self.class_count, op=hvd.Sum)
        rm = torch.tensor([rm])
        rm = hvd.allgather(rm)
        add = torch.tensor([add])
        add = hvd.allgather(add)
        offsets = torch.tensor([offsets])
        offsets = hvd.allgather(offsets)
        new_reps = hvd.allgather(torch.cat(self.candidates).unsqueeze(0))

        for worker in range(hvd.size()):
            add_w = add[worker]
            add_w = add_w[add_w != -1]
            rm_w = rm[worker]
            rm_w = rm_w[rm_w != -1]

            for c in range(self.model.num_classes):
                lower = offsets[worker][c]
                upper = offsets[worker][c + 1]
                if lower == upper:
                    continue

                to_add = add_w
                to_add = to_add[
                    (to_add >= self.num_candidates * c)
                    & (to_add < self.num_candidates * (c + 1))
                ]
                to_add -= self.num_candidates * c
                if len(rm_w) > 0:
                    to_rm = rm_w
                    to_rm = to_rm[
                        (to_rm >= self.num_representatives * c)
                        & (to_rm < self.num_representatives * (c + 1))
                    ]
                    to_rm -= self.num_representatives * c

                if len(torch.flatten(self.reps_by_worker[worker][c])) == 0:
                    self.reps_by_worker[worker][c] = new_reps[worker][lower:upper][
                        to_add
                    ]
                else:
                    if len(rm_w) > 0 and len(to_rm) > 0:
                        selected = [
                            i
                            for i in range(len(self.reps_by_worker[worker][c]))
                            if i not in to_rm
                        ]
                        self.reps_by_worker[worker][c] = self.reps_by_worker[worker][c][
                            selected
                        ]
                    self.reps_by_worker[worker][c] = torch.cat(
                        (
                            self.reps_by_worker[worker][c],
                            new_reps[worker][lower:upper][to_add],
                        )
                    )

        self.global_x_reps = torch.cat(
            [torch.cat(per_worker) for per_worker in self.reps_by_worker]
        )
        self.candidates = [torch.Tensor()
                           for _ in range(self.model.num_classes)]

        for i in range(self.model.num_classes):
            self.representatives[i] = self.reps_by_worker[hvd.rank()][i]

        y = torch.tensor(
            [
                i
                for i in range(self.model.num_classes)
                for _ in range(len(self.representatives[i]))
            ]
        )
        self.global_y_reps = hvd.allgather(y)

    def recalculate_weights(self):
        """Reassign the weights of the representatives"""
        total_count = torch.sum(self.global_class_count).item()
        if total_count == 0:
            return
        # This version proposes that the total weight of representatives is calculated from the proportion of candidate
        # representatives respect to the batch. E.g. a batch of 100 images and 10 are num_candidates selected, total_weight = 10
        total_weight = (self.batch_size * 1.0) / self.num_candidates
        # The total_weight is adjusted to the proportion between candidate representatives and actual representatives
        total_weight *= total_count / len(self.global_y_reps)
        probs = self.global_class_count.float() / total_count
        # probs = hvd.allreduce(probs, name='probs')
        ws = []
        for y in self.global_y_reps.unique():
            y = y.item()
            weight = max(math.log(probs[y].item() * total_weight), 1.0)
            ws.append((self.global_y_reps == y).float() * weight)
        self.global_w_reps = torch.sum(torch.stack(ws), 0)

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
                            "num_representatives",
                            self.get_num_representatives(),
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
                (
                    rep_values,
                    rep_labels,
                    rep_weights,
                ) = self.get_representatives()
                num_reps = len(rep_values)
                if self.writer is not None and self.writer_images and num_reps > 0:
                    grid = torchvision.utils.make_grid(rep_values)
                    caption = ', '.join([str(label.item())
                                        for label in rep_labels])
                    self.writer.add_image(caption, grid, self.global_steps)

            # Get the respective weights
            w = torch.ones(len(x))
            torch.cuda.nvtx.range_push("Copy to device")
            x, y, w = (
                move_cuda(x, self.cuda),
                move_cuda(y, self.cuda),
                move_cuda(w, self.cuda),
            )
            torch.cuda.nvtx.range_pop()

            if training and num_reps > 0:
                torch.cuda.nvtx.range_push("Combine batches")
                rep_values, rep_labels, rep_weights = (
                    move_cuda(rep_values, self.cuda),
                    move_cuda(rep_labels, self.cuda),
                    move_cuda(rep_weights, self.cuda),
                )
                # Concatenates the training samples with the representatives
                w = torch.cat((w, rep_weights))
                x = torch.cat((x, rep_values))
                y = torch.cat((y, rep_labels))
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Forward pass")
            output = self.model(x)
            if training:
                loss = self.criterion(output, y)
            else:
                loss = nn.CrossEntropyLoss()(output, y)
            torch.cuda.nvtx.range_pop()

            if training:
                total_weight = hvd.allreduce(
                    torch.sum(w), name="total_weight", op=hvd.Sum
                )
                dw = w / total_weight
                # Faster to provide the derivative of L wrt {l}^b than letting pytorch computing it by itself
                loss.backward(dw)
                # SGD step
                torch.cuda.nvtx.range_push("Optimizer step")
                self.optimizer.step()
                torch.cuda.nvtx.range_pop()
                self.global_steps += 1
                self.steps += 1

                if num_reps == 0:
                    if self.buffer_cuda:
                        self.pick_candidates(x, y)
                    else:
                        self.pick_candidates(x.cpu(), y.cpu())
                else:
                    if self.buffer_cuda:
                        self.pick_candidates(x[:-num_reps], y[:-num_reps])
                    else:
                        self.pick_candidates(
                            x[:-num_reps].cpu(), y[:-num_reps].cpu())

            outputs.append(output.detach())
            total_loss += torch.mean(loss).item()

            torch.cuda.nvtx.range_pop()

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss
