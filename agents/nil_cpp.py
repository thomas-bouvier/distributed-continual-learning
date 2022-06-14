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

import ctypes
from cpp_loader import rehearsal

from agents.base import Agent
from utils.utils import get_device, move_cuda, plot_representatives
from utils.meters import AverageMeter, accuracy

from bisect import bisect


class nil_cpp_agent(Agent):
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
        super(nil_cpp_agent, self).__init__(
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

        self.sl = rehearsal.StreamLoader(
            model.num_classes, self.num_representatives, self.num_candidates, ctypes.c_int64(torch.random.initial_seed()).value)

        self.aug_x = None
        self.aug_y = torch.randint(high=model.num_classes, size=(
            self.batch_size + self.num_samples,), device=self.device)
        self.aug_w = torch.zeros(
            self.batch_size + self.num_samples, device=self.device)

        self.mask = torch.as_tensor(
            [0.0 for _ in range(self.model.num_classes)],
            device=torch.device(get_device(self.cuda)),
        )

        self.acc_get_time = 0
        self.acc_cat_time = 0
        self.acc_acc_time = 0

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
            self.optimizer.reset(self.model.parameters())

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
            # torch.cuda.nvtx.range_pop()
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
        inputs,
        target,
        training=False,
        average_output=False,
        chunk_batch=1,
    ):
        outputs = []
        total_loss = 0

        if training:
            self.optimizer.zero_grad()
            self.optimizer.update(self.epoch, self.steps)

        for i, (x, y) in enumerate(zip(inputs.chunk(chunk_batch, dim=0),
                                       target.chunk(chunk_batch, dim=0))):
            w = torch.ones(len(x), device=torch.device(get_device(self.cuda)))
            #torch.cuda.nvtx.range_push("Copy to device")
            x, y = move_cuda(x, self.cuda and self.buffer_cuda), move_cuda(
                y, self.cuda and self.buffer_cuda)
            # torch.cuda.nvtx.range_pop()

            if self.aug_x is None:
                size = list(x.size())[1:]
                self.aug_x = torch.zeros(
                    self.batch_size + self.num_samples, *size, device=self.device)

            if training:
                start_acc_time = time.time()
                self.sl.accumulate(x.detach().clone(), y.detach(
                ).clone(), self.aug_x, self.aug_y, self.aug_w)
                self.acc_acc_time += time.time() - start_acc_time

                # Get the representatives
                start_get_time = time.time()
                self.sl.wait()
                self.acc_get_time += time.time() - start_get_time
                self.num_reps = self.sl.get_rehearsal_size()
                history_count = self.sl.get_history_count()

                # Log representatives
                if self.writer is not None and self.writer_images and self.num_reps > 0:
                    fig = plot_representatives(
                        self.aug_x[len(x):], self.aug_y[len(x):], 5)
                    self.writer.add_figure(
                        "representatives", fig, self.global_steps)

            #torch.cuda.nvtx.range_push("Forward pass")
            output = self.model(self.aug_x)
            if training:
                loss = self.criterion(output, self.aug_y)
            else:
                loss = nn.CrossEntropyLoss()(output, self.aug_y)
            # torch.cuda.nvtx.range_pop()

            if training:
                # Leads to decreased accuracy
                # total_weight = hvd.allreduce(torch.sum(w), name='total_weight', op=hvd.Sum)
                dw = self.aug_w / torch.sum(self.aug_w)
                # Faster to provide the derivative of L wrt {l}^n than letting
                # pytorch computing it by itself
                loss.backward(dw)
                # SGD step
                #torch.cuda.nvtx.range_push("Optimizer step")
                self.optimizer.step()
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
        return -1
