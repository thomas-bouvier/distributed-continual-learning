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


class nil_cpp_global_agent(Agent):
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
        super(nil_cpp_global_agent, self).__init__(
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

        if self.use_amp:
            try:
                global amp
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to run this app.")

        self.device = "cuda" if self.buffer_cuda else 'cpu'

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
        self.last_batch_load_time = 0
        self.last_batch_move_time = 0
        self.last_batch_wait_time = 0
        self.last_batch_cat_time = 0
        self.last_batch_acc_time = 0

    def before_all_tasks(self, train_data_regime):
        super().before_all_tasks(train_data_regime)

        port = 1234 + hvd.rank()
        remote_nodes = []
        remote_ips = self.config.get("remote_nodes", "")
        if remote_ips:
            for ip in remote_ips.split(','):
                address_and_port = ip.split(':')
                # No need to exclude current node!
                #if len(address_and_port) == 3 and int(address_and_port[2]) == port:
                #    continue
                remote_nodes.append((int(address_and_port[2]) - 1234, ip))
        logging.debug(f"Remote nodes to sample from: {remote_nodes}")
        self.dsl = rehearsal.DistributedStreamLoader(
            self.num_classes, self.num_representatives, self.num_candidates,
            ctypes.c_int64(torch.random.initial_seed()).value,
            ctypes.c_uint16(hvd.rank()).value,
            f"tcp://127.0.0.1:{port}",
            remote_nodes
        )

        self.aug_x = None
        self.aug_y = torch.randint(high=self.num_classes, size=(
            self.batch_size + self.num_samples,), device=self.device)
        self.aug_w = torch.zeros(
            self.batch_size + self.num_samples, device=self.device)

        self.mask = torch.as_tensor(
            [0.0 for _ in range(self.num_classes)],
            device=torch.device(get_device(self.cuda)),
        )

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

    def loop(self, data_regime, training=False):
        prefix = "train" if training else "val"
        meters = {
            metric: AverageMeter(f"{prefix}_{metric}")
            for metric in ["loss", "prec1", "prec5"]
        }
        epoch_time = 0
        step_count = 0

        start_batch_time = time.time()
        start_load_time = start_batch_time
        for i_batch, (x, y, t) in enumerate(data_regime.get_loader()):
            #torch.cuda.nvtx.range_push(f"Batch {i_batch}")
            synchronize_cuda(self.cuda)
            self.last_batch_load_time = time.time() - start_load_time
            self.epoch_load_time += self.last_batch_load_time

            output, loss = self._step(
                i_batch, x, y, training=training
            )
            synchronize_cuda(self.cuda)
            batch_time = time.time() - start_batch_time
            epoch_time += batch_time

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

                logging.info(f"batch {i_batch} time {batch_time} sec")
                logging.info(
                    f"\tbatch load time {self.last_batch_load_time} sec ({self.last_batch_load_time*100/batch_time}%)")
                logging.info(
                    f"\tbatch move time {self.last_batch_move_time} sec ({self.last_batch_move_time*100/batch_time}%)")
                logging.info(
                    f"\tbatch wait time {self.last_batch_wait_time} sec ({self.last_batch_wait_time*100/batch_time}%)")
                logging.info(
                    f"\tbatch cat time {self.last_batch_cat_time} sec ({self.last_batch_cat_time*100/batch_time}%)")
                logging.info(
                    f"\tbatch acc time {self.last_batch_acc_time} sec ({self.last_batch_acc_time*100/batch_time}%)")

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
            # torch.cuda.nvtx.range_pop()
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
        inputs,
        target,
        training=False,
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
            start_move_time = time.time()
            x, y = move_cuda(x, self.cuda and self.buffer_cuda), move_cuda(
                y, self.cuda and self.buffer_cuda)
            self.last_batch_move_time = time.time() - start_move_time
            self.epoch_move_time += self.last_batch_move_time
            # torch.cuda.nvtx.range_pop()

            if self.aug_x is None:
                size = list(x.size())[1:]
                self.aug_x = torch.zeros(
                    self.batch_size + self.num_samples, *size, device=self.device)
                self.dsl.accumulate(x, y, self.aug_x, self.aug_y, self.aug_w)

            if training:
                # Get the representatives
                start_wait_time = time.time()
                self.dsl.wait()
                self.last_batch_wait_time = time.time() - start_wait_time
                self.epoch_wait_time += self.last_batch_wait_time

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

            outputs.append(output.detach())
            total_loss += torch.mean(loss).item()

        outputs = torch.cat(outputs, dim=0)
        return outputs, total_loss

    def get_num_representatives(self):
        return self.num_reps

    def get_memory_size(self):
        return -1
