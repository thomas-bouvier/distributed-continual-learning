import copy
import horovod.torch as hvd
import itertools
import math
import numpy as np
import logging
import time
import torch
import torchvision
import wandb

import ctypes
from cpp_loader import rehearsal

from agents.base import Agent
from torch.cuda.amp import GradScaler, autocast
from utils.utils import get_device, move_cuda, plot_representatives, synchronize_cuda, display
from utils.meters import AverageMeter, accuracy

from bisect import bisect


class nil_cpp_global_agent(Agent):
    def __init__(
        self,
        model,
        use_amp,
        config,
        optimizer_regime,
        cuda,
        buffer_cuda,
        log_buffer,
        log_interval,
        batch_metrics=None,
        state_dict=None,
    ):
        super(nil_cpp_global_agent, self).__init__(
            model,
            use_amp,
            config,
            optimizer_regime,
            cuda,
            buffer_cuda,
            log_buffer,
            log_interval,
            batch_metrics,
            state_dict,
        )

        self.device = 'cuda' if self.cuda else 'cpu'
        self.num_representatives = config.get("num_representatives", 60)
        self.num_candidates = config.get("num_candidates", 20)
        self.num_samples = config.get("num_samples", 20)
        self.batch_size = config.get("batch_size")

        self.num_reps = 0

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

        shape = next(iter(train_data_regime.get_loader()))[0][0].size()

        self.dsl = rehearsal.DistributedStreamLoader(
            rehearsal.Classification,
            self.num_classes, self.num_representatives, self.num_candidates,
            ctypes.c_int64(torch.random.initial_seed() + hvd.rank()).value,
            ctypes.c_uint16(hvd.rank()).value,
            "na+sm", 1, list(shape), True
        )

        self.aug_x = torch.zeros(
            self.batch_size + self.num_samples, *shape, device=self.device)
        self.aug_y = torch.randint(high=self.num_classes, size=(
            self.batch_size + self.num_samples,), device=self.device)
        self.aug_w = torch.zeros(
            self.batch_size + self.num_samples, device=self.device)

    def before_every_task(self, task_id, train_data_regime):
        self.steps = 0

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
        step_count = 0

        dataloader_iter = enumerate(data_regime.get_loader())

        start_batch_time = time.time()
        start_load_time = start_batch_time
        for i_batch, (x, y, t) in dataloader_iter:
            synchronize_cuda(self.cuda)
            self.last_batch_load_time = time.time() - start_load_time
            self.epoch_load_time += self.last_batch_load_time

            output, loss = self._step(
                i_batch, x, y, training=training
            )
            synchronize_cuda(self.cuda)
            self.last_batch_time = time.time() - start_batch_time
            epoch_time += self.last_batch_time

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output[: y.size(0)], y, topk=(1, 5))
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

                logging.info(f"batch {i_batch} time {self.last_batch_time} sec")
                logging.info(
                    f"\tbatch load time {self.last_batch_load_time} sec ({self.last_batch_load_time*100/self.last_batch_time}%)")
                logging.info(
                    f"\tbatch move time {self.last_batch_move_time} sec ({self.last_batch_move_time*100/self.last_batch_time}%)")
                logging.info(
                    f"\tbatch wait time {self.last_batch_wait_time} sec ({self.last_batch_wait_time*100/self.last_batch_time}%)")
                logging.info(
                    f"\tbatch cat time {self.last_batch_cat_time} sec ({self.last_batch_cat_time*100/self.last_batch_time}%)")
                logging.info(
                    f"\tbatch acc time {self.last_batch_acc_time} sec ({self.last_batch_acc_time*100/self.last_batch_time}%)")
                logging.info(
                    f"\tnum_representatives {self.get_num_representatives()}")

                if hvd.rank() == 0 and hvd.local_rank() == 0:
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

                    wandb.log({f"{prefix}_loss": meters["loss"].avg,
                               "step": self.global_steps,
                               "epoch": self.global_epoch,
                               "batch": i_batch,
                               f"{prefix}_prec1": meters["prec1"].avg,
                               f"{prefix}_prec5": meters["prec5"].avg})
                    if training:
                        wandb.log({"lr": self.optimizer_regime.get_lr()[0],
                                   "num_reps": self.get_num_representatives(),
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
                            "lr", self.optimizer_regime.get_lr()[
                                0], self.global_steps
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
                        optimizer=self.optimizer_regime.optimizer,
                        data=(x, y),
                    )
                    self.stream_meters(meters, prefix=prefix)
                    if training:
                        self.write_stream(
                            "lr",
                            (self.global_steps,
                             self.optimizer_regime.get_lr()[0]),
                        )

            step_count += 1
            synchronize_cuda(self.cuda)
            start_batch_time = time.time()
            start_load_time = start_batch_time

        if training:
            self.global_epoch += 1
            logging.info(f"\nCUMULATED VALUES:")
            logging.info(
                f"\tnum_representatives {self.get_num_representatives()}")
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
        meters["step_count"] = step_count

        return meters

    def _step(
        self,
        i_batch,
        x,
        y,
        training=False,
    ):
        if training:
            self.optimizer_regime.zero_grad()
            self.optimizer_regime.update(self.epoch, self.steps)

        w = torch.ones(len(x), device=torch.device(self.device))
        start_move_time = time.time()
        x, y = move_cuda(x, self.cuda), move_cuda(y, self.cuda)
        self.last_batch_move_time = time.time() - start_move_time
        self.epoch_move_time += self.last_batch_move_time

        if self.epoch == 0 and i_batch == 0:
            self.dsl.accumulate(x, y, self.aug_x, self.aug_y, self.aug_w)

        if training:
            # Get the representatives
            start_wait_time = time.time()
            aug_size = self.dsl.wait()
            self.last_batch_wait_time = time.time() - start_wait_time
            self.epoch_wait_time += self.last_batch_wait_time

            logging.info(f"Received {aug_size - self.batch_size} samples from other nodes")
            if self.log_buffer and i_batch % self.log_interval == 0 and self.num_reps > 0 and hvd.rank() == 0 and hvd.local_rank() == 0:
                display(f"aug_batch_{self.epoch}_{i_batch}", self.aug_x, aug_size, captions=self.aug_y, cuda=self.cuda)

            ############ slot 1
            #start_acc_time = time.time()
            #self.dsl.accumulate(x, y, self.aug_x, self.aug_y, self.aug_w)
            #self.last_batch_acc_time = time.time() - start_acc_time
            #self.epoch_acc_time += self.last_batch_acc_time

            if self.use_amp:
                with autocast():
                    output = self.model(self.aug_x)
                    loss = self.criterion(output, self.aug_y)
            else:
                output = self.model(self.aug_x)
                loss = self.criterion(output, self.aug_y)
        else:
            if self.use_amp:
                with autocast():
                    output = self.model(x)
                    loss = self.criterion(output, y)
            else:
                output = self.model(x)
                loss = self.criterion(output, y)

        if training:
            ############ slot 2
            #start_acc_time = time.time()
            #self.dsl.accumulate(x, y, self.aug_x, self.aug_y, self.aug_w)
            #self.last_batch_acc_time = time.time() - start_acc_time
            #self.epoch_acc_time += self.last_batch_acc_time

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.optimizer_regime.optimizer.synchronize()
                with self.optimizer_regime.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer_regime.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                self.optimizer_regime.step()

            ############ slot 3
            start_acc_time = time.time()
            self.dsl.accumulate(x, y, self.aug_x, self.aug_y, self.aug_w)
            self.last_batch_acc_time = time.time() - start_acc_time
            self.epoch_acc_time += self.last_batch_acc_time

            self.global_steps += 1
            self.steps += 1
            self.num_reps = self.dsl.get_rehearsal_size()

            # Log representatives
            if self.writer is not None and self.writer_images and self.num_reps > 0:
                fig = plot_representatives(
                    self.aug_x[len(x):], self.aug_y[len(x):], 5)
                self.writer.add_figure(
                    "representatives", fig, self.global_steps)

        return output, loss

    def get_num_representatives(self):
        return self.num_reps

    def get_memory_size(self):
        return -1
