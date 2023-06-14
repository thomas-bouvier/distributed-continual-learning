import copy
import horovod.torch as hvd
import numpy as np
import logging
import time
from tqdm import tqdm
import torch
import torch.nn as nn
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

    def train_one_step(
        self,
        x,
        y,
        meters,
    ):
        # Get the representatives
        with self.get_timer('wait', previous_iteration=True):
            self.repr_size = self.dsl.wait()
            n = self.repr_size
            if n > 0:
                logging.debug(f"Received {n} samples from other nodes")

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


    def get_timer(self, name, previous_iteration=False):
        batch = self.batch
        if previous_iteration:
            batch -= 1
        return MeasureTime(batch, name, self.perf_metrics, self.cuda, not self.measure_performance())