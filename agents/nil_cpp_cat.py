import copy
import horovod.torch as hvd
import numpy as np
import logging
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import wandb

import ctypes
import neomem

from agents.base import Agent
from torch.cuda.amp import GradScaler, autocast
from utils.meters import AverageMeter, MeasureTime, accuracy
from utils.utils import move_cuda, display


class AugmentedMinibatch:
    def __init__(self, num_representatives, shape, device):
        self.x = torch.zeros(num_representatives, *shape, device=device)
        self.y = torch.randint(high=1000, size=(num_representatives,), device=device)
        self.w = torch.ones(num_representatives, device=device)


class nil_cpp_cat_agent(Agent):
    def __init__(
        self,
        model,
        use_mask,
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
        super(nil_cpp_cat_agent, self).__init__(
            model,
            use_mask,
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

        self.rehearsal_size = config.get("rehearsal_size", 16)
        self.num_candidates = config.get("num_candidates", 8)
        self.num_representatives = config.get("num_representatives", 8)
        self.provider = config.get('provider', 'na+sm')
        self.discover_endpoints = config.get('discover_endpoints', True)
        self.cuda_rdma = config.get('cuda_rdma', False)

        self.current_rehearsal_size = 0

    def measure_performance(self):
        #assert self.current_rehearsal_size > 0
        return self.task_id == 1 and self.epoch == 10

    def before_all_tasks(self, train_data_regime):
        super().before_all_tasks(train_data_regime)

        x, y, _ = next(iter(train_data_regime.get_loader()))
        shape = x[0].size()
        self.next_minibatch = AugmentedMinibatch(self.num_representatives, shape, self.device)

        engine = neomem.EngineLoader(self.provider,
            ctypes.c_uint16(hvd.rank()).value, self.cuda_rdma
        )
        self.dsl = neomem.DistributedStreamLoader(
            engine,
            neomem.Classification,
            train_data_regime.total_num_classes, self.rehearsal_size, self.num_representatives, self.num_candidates,
            ctypes.c_int64(torch.random.initial_seed() + hvd.rank()).value,
            1, list(shape), neomem.CPUBuffer, self.discover_endpoints, self.log_level not in ('info')
        )
        #self.dsl.enable_augmentation(True)
        self.dsl.use_these_allocated_variables(self.next_minibatch.x, self.next_minibatch.y, self.next_minibatch.w)
        self.dsl.start()

        self.dsl.accumulate(x, y)

    def before_every_task(self, task_id, train_data_regime):
        super().before_every_task(task_id, train_data_regime)

        if task_id > 0:
            self.dsl.enable_augmentation(True)

    """Forward pass for the current epoch"""
    def train_one_epoch(self, data_regime):
        prefix = "train"
        meters = {
            metric: AverageMeter(f"{prefix}_{metric}")
            for metric in ["loss", "prec1", "prec5", "num_samples"]
        }
        epoch_time = 0
        last_batch_time = 0
        loader = data_regime.get_loader()

        enable_tqdm = self.log_level in ('info') and hvd.rank() == 0
        with tqdm(total=len(loader),
              desc=f"Task #{self.task_id + 1} {prefix} epoch #{self.epoch + 1}",
              disable=not enable_tqdm
        ) as progress:
            timer = self.get_timer('load', previous_iteration=True)
            timer.__enter__()
            start_batch_time = time.perf_counter()

            for x, y, t in loader:
                x, y = move_cuda(x, self.cuda), move_cuda(y.long(), self.cuda)
                timer.__exit__(None, None, None)

                self.train_one_step(x, y, meters)

                last_batch_time = time.perf_counter() - start_batch_time
                epoch_time += last_batch_time

                if hvd.rank() == 0:
                    # Performance metrics
                    if self.measure_performance() and self.batch_metrics is not None:
                        metrics = self.perf_metrics.get(self.batch-1)
                        batch_metrics_values = dict(
                            epoch=self.epoch,
                            batch=self.batch,
                            time=last_batch_time,
                            load_time=metrics.get('load', 0),
                            wait_time=metrics.get('wait', 0),
                            assemble_time=metrics.get('assemble', 0),
                            train_time=metrics.get('train', 0),
                            accumulate_time=metrics[0],
                            copy_time=metrics[1],
                            bulk_prepare_time=metrics[2],
                            rpcs_resolve_time=metrics[3],
                            representatives_copy_time=metrics[4],
                            buffer_update_time=metrics[5],
                            aug_size=self.repr_size,
                            local_rehearsal_size=self.current_rehearsal_size,
                        )
                        self.batch_metrics.add(**batch_metrics_values)
                        self.batch_metrics.save()

                    # Logging
                    if self.batch % self.log_interval == 0 or self.batch == len(loader):
                        logging.debug(
                            "{0}: epoch: {1} [{2}/{3}]\t"
                            "Loss {meters[loss].avg:.4f}\t"
                            "Prec@1 {meters[prec1].avg:.3f}\t"
                            "Prec@5 {meters[prec5].avg:.3f}\t".format(
                                prefix,
                                self.epoch + 1,
                                self.batch % len(loader) + 1,
                                len(loader),
                                meters=meters,
                            )
                        )

                        if self.measure_performance():
                            metrics = self.perf_metrics.get(self.batch-1)
                            logging.debug(f"batch {self.batch} time {last_batch_time} sec")
                            logging.debug(
                                f"\t[C++] local_rehearsal_size {self.current_rehearsal_size}")
                            logging.debug(
                                f"\t[Python] batch wait time {metrics.get('wait', 0)} sec ({metrics.get('wait', 0)*100/last_batch_time}%)")
                            logging.debug(
                                f"\t[Python] batch load time {metrics.get('load', 0)} sec ({metrics.get('load', 0)*100/last_batch_time}%)")
                            logging.debug(
                                f"\t[Python] batch assemble time {metrics.get('assemble', 0)} sec ({metrics.get('assemble', 0)*100/last_batch_time}%)")
                            logging.debug(
                                f"\t[Python] batch train time {metrics.get('train', 0)} sec ({metrics.get('train', 0)*100/last_batch_time}%)")

                            logging.debug(
                                f"\t[C++] batch accumulate time {metrics[0]} sec")
                            logging.debug(
                                f"\t[C++] batch copy time {metrics[1]} sec")
                            logging.debug(
                                f"\t[C++] bulk prepare time {metrics[2]} sec")
                            logging.debug(
                                f"\t[C++] rpcs resolve time {metrics[3]} sec")
                            logging.debug(
                                f"\t[C++] representatives copy time {metrics[4]} sec")
                            logging.debug(
                                f"\t[C++] buffer update time {metrics[5]} sec")

                progress.set_postfix({'loss': meters["loss"].avg.item(),
                           'accuracy': meters["prec1"].avg.item(),
                           'augmented': self.repr_size,
                           'local_rehearsal_size': self.current_rehearsal_size})
                progress.update(1)

                self.global_batch += 1
                self.batch += 1

                timer = self.get_timer('load', previous_iteration=True)
                timer.__enter__()
                start_batch_time = time.perf_counter()

        if hvd.rank() == 0:
            wandb.log({"epoch": self.global_epoch,
                    f"{prefix}_loss": meters["loss"].avg,
                    f"{prefix}_prec1": meters["prec1"].avg,
                    f"{prefix}_prec5": meters["prec5"].avg,
                    "lr": self.optimizer_regime.get_lr()[0],
                    "local_rehearsal_size": self.current_rehearsal_size})

        self.global_epoch += 1
        self.epoch += 1

        logging.info(f"\nCUMULATED VALUES:")
        logging.info(
            f"\tlocal_rehearsal_size {self.current_rehearsal_size}")
        logging.info(f"epoch time {epoch_time} sec")
        """
        logging.info(
            f"\tepoch load time {self.epoch_load_time} sec ({self.epoch_load_time*100/epoch_time}%)")
        logging.info(
            f"\tepoch train time {self.epoch_train_time} sec ({self.epoch_train_time*100/epoch_time}%)")
        logging.info(
            f"\tepoch wait time {self.epoch_wait_time} sec ({self.epoch_wait_time*100/epoch_time}%)")
        logging.info(
            f"\tepoch assemble time {self.epoch_assemble_time} sec ({self.epoch_assemble_time*100/epoch_time}%)")
        """

        num_samples = meters["num_samples"].sum.item()
        meters = {name: meter.avg.item() for name, meter in meters.items()}
        meters["num_samples"] = num_samples
        meters["error1"] = 100.0 - meters["prec1"]
        meters["error5"] = 100.0 - meters["prec5"]
        meters["time"] = epoch_time
        meters["batch"] = self.batch

        return meters

    def train_one_step(
        self,
        x,
        y,
        meters,
    ):
        # Get the representatives
        with self.get_timer('wait', previous_iteration=True):
            self.repr_size = self.dsl.wait()
            logging.debug(f"Received {self.repr_size} samples from other nodes")

            if self.measure_performance():
                cpp_metrics = self.dsl.get_metrics(self.batch)
                self.perf_metrics.add(self.batch-1, cpp_metrics)

        # Assemble the minibatch
        with self.get_timer('assemble'):
            w = torch.ones(self.batch_size, device=self.device)
            new_x = torch.cat((x, self.next_minibatch.x[:self.repr_size]))
            new_y = torch.cat((y, self.next_minibatch.y[:self.repr_size]))
            new_w = torch.cat((w, self.next_minibatch.w[:self.repr_size]))

        if self.log_buffer and self.batch % self.log_interval == 0 and hvd.rank() == 0:
            if self.repr_size > 0:
                captions = []
                for y, w in zip(self.next_minibatch.y[-self.repr_size:], self.next_minibatch.w[-self.repr_size:]):
                    captions.append(f"y={y.item()} w={w.item()}")
                display(f"aug_batch_{self.task_id}_{self.epoch}_{self.batch}", self.next_minibatch.x[-self.repr_size:], captions=captions)

        # In-advance preparation of next minibatch
        with self.get_timer('accumulate'):
            self.dsl.accumulate(x, y)

        # Train
        with self.get_timer('train'):
            self.optimizer_regime.update(self.epoch, self.batch)
            self.optimizer_regime.zero_grad()

            if self.use_amp:
                with autocast():
                    output = self.model(new_x)
                    loss = self.criterion(output, new_y)
            else:
                output = self.model(new_x)
                loss = self.criterion(output, new_y)

            assert not torch.isnan(loss).any(), "Loss is NaN, stopping training"

            # https://stackoverflow.com/questions/43451125/pytorch-what-a-re-the-gradient-arguments
            total_weight = hvd.allreduce(torch.sum(new_w), name='total_weight', op=hvd.Sum)
            dw = new_w / total_weight * self.batch_size * hvd.size() / self.batch_size

            if self.use_amp:
                self.scaler.scale(loss).backward(dw)
                self.optimizer_regime.optimizer.synchronize()
                with self.optimizer_regime.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer_regime.optimizer)
                    self.scaler.update()
            else:
                loss.backward(dw)
                self.optimizer_regime.step()

            self.current_rehearsal_size = self.dsl.get_rehearsal_size()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, new_y, topk=(1, 5))
            # https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/10
            meters["loss"].update(loss.sum() / self.mask[new_y].sum())
            meters["prec1"].update(prec1, new_x.size(0))
            meters["prec5"].update(prec5, new_x.size(0))
            meters["num_samples"].update(self.batch_size + self.repr_size)

    def validate_one_epoch(self, data_regime, task_id):
        previous_task = task_id != self.task_id

        prefix = "val"
        meters = {
            metric: AverageMeter(f"{prefix}_{metric}")
            for metric in ["loss", "prec1", "prec5", "num_samples"]
        }
        epoch_time = 0
        last_batch_time = 0
        loader = data_regime.get_loader(task_id)

        criterion = torch.nn.CrossEntropyLoss()

        enable_tqdm = self.log_level in ('info') and hvd.rank() == 0
        with tqdm(total=len(loader),
              desc=f"Task #{self.task_id + 1} {prefix} epoch #{self.epoch}",
              disable=not enable_tqdm
        ) as progress:
            start_batch_time = time.perf_counter()

            for x, y, t in loader:
                x, y = move_cuda(x, self.cuda), move_cuda(y, self.cuda)

                if self.use_amp:
                    with autocast():
                        output = self.model(x)
                        loss = criterion(output, y)
                else:
                    output = self.model(x)
                    loss = criterion(output, y)

                prec1, prec5 = accuracy(output, y, topk=(1, 5))
                meters["loss"].update(loss, x.size(0))
                meters["prec1"].update(prec1, x.size(0))
                meters["prec5"].update(prec5, x.size(0))

                last_batch_time = time.perf_counter() - start_batch_time
                epoch_time += last_batch_time

                progress.set_postfix({'loss': meters["loss"].avg.item(),
                           'accuracy': meters["prec1"].avg.item(),
                           'augmented': self.repr_size,
                           'local_rehearsal_size': self.current_rehearsal_size})
                progress.update(1)

        if hvd.rank() == 0 and not previous_task:
            wandb.log({"epoch": self.global_epoch,
                    f"{prefix}_loss": meters["loss"].avg,
                    f"{prefix}_prec1": meters["prec1"].avg,
                    f"{prefix}_prec5": meters["prec5"].avg})

        logging.info(f"epoch time {epoch_time} sec")

        num_samples = meters["num_samples"].sum.item()
        meters = {name: meter.avg.item() for name, meter in meters.items()}
        meters["num_samples"] = num_samples
        meters["error1"] = 100.0 - meters["prec1"]
        meters["error5"] = 100.0 - meters["prec5"]
        meters["time"] = epoch_time
        meters["batch"] = self.batch

        return meters

    def get_timer(self, name, previous_iteration=False):
        batch = self.batch
        if previous_iteration:
            batch -= 1
        return MeasureTime(batch, name, self.perf_metrics, self.cuda, not self.measure_performance())