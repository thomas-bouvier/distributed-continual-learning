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
from utils.utils import get_device, move_cuda, display
from utils.log import PerformanceResultsLog
from utils.meters import AverageMeter, MeasureTime, accuracy


class AugmentedMinibatch:
    def __init__(self, batch_size, num_representatives, shape, device):
        self.x = torch.zeros(
            batch_size + num_representatives, *shape, device=device)
        self.y = torch.randint(high=1000, size=(
            batch_size + num_representatives,), device=device)
        self.w = torch.zeros(
            batch_size + num_representatives, device=device)


class nil_cpp_agent(Agent):
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
        super(nil_cpp_agent, self).__init__(
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

        self.rehearsal_size = config.get("rehearsal_size", 100)
        self.num_candidates = config.get("num_candidates", 20)
        self.num_representatives = config.get("num_representatives", 20)
        self.provider = config.get('provider', 'na+sm')
        self.discover_endpoints = config.get('discover_endpoints', True)
        self.cuda_rdma = config.get('cuda_rdma', False)

        self.current_rehearsal_size = 0
        self.perf_metrics = PerformanceResultsLog()

    def measure_performance(self):
        assert self.current_rehearsal_size > 0
        return self.task_id == 0 and self.global_epoch == 0

    def before_all_tasks(self, train_data_regime):
        super().before_all_tasks(train_data_regime)

        x, y, _ = next(iter(train_data_regime.get_loader()))
        shape = x[0].size()

        engine = neomem.EngineLoader(self.provider,
            ctypes.c_uint16(hvd.rank()).value, self.cuda_rdma
        )
        self.dsl = neomem.DistributedStreamLoader(
            engine,
            neomem.Classification,
            train_data_regime.total_num_classes, self.rehearsal_size, self.num_representatives, self.num_candidates,
            ctypes.c_int64(torch.random.initial_seed() + hvd.rank()).value,
            1, list(shape), self.discover_endpoints, self.log_level not in ('info')
        )
        #self.dsl.enable_augmentation(True)
        self.dsl.start()

        self.minibatches_ahead = 2
        self.next_minibatches = []
        for i in range(self.minibatches_ahead):
            self.next_minibatches.append(AugmentedMinibatch(self.batch_size, self.num_representatives, shape, self.device))

        self.dsl.accumulate(x, y, self.next_minibatches[1].x,
                                  self.next_minibatches[1].y,
                                  self.next_minibatches[1].w)

    def before_every_task(self, task_id, train_data_regime):
        self.task_id = task_id
        self.batch = 0

        # Distribute the data
        train_data_regime.get_loader(True)

        if self.best_model is not None:
            logging.debug(
                f"Loading best model with minimal eval loss ({self.minimal_eval_loss}).."
            )
            self.model.load_state_dict(self.best_model)
            self.minimal_eval_loss = float("inf")

        if task_id > 0:
            self.dsl.enable_augmentation(True)

            if self.config.get("reset_state_dict", False):
                logging.debug("Resetting model internal state..")
                self.model.load_state_dict(
                    copy.deepcopy(self.initial_snapshot))
            self.optimizer_regime.reset(self.model.parameters())

        # Create mask so the loss is only used for classes learnt during this task
        self.mask = torch.tensor(train_data_regime.previous_classes_mask, device=self.device).float()
        self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')

    """Forward pass for the current epoch"""
    def loop(self, data_regime, training=False, previous_task=False):
        prefix = "train" if training else "val"
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
                x, y = move_cuda(x, self.cuda), move_cuda(y, self.cuda)
                timer.__exit__(None, None, None)

                self._step(x, y, meters, training=training, previous_task=previous_task)

                last_batch_time = time.perf_counter() - start_batch_time
                epoch_time += last_batch_time

                if hvd.rank() == 0:
                    # Performance metrics
                    if not previous_task:
                        wandb.log({f"{prefix}_loss": meters["loss"].avg,
                                "batch": self.global_batch,
                                "epoch": self.global_epoch,
                                f"{prefix}_prec1": meters["prec1"].avg,
                                f"{prefix}_prec5": meters["prec5"].avg})

                        if training:
                            wandb.log({"lr": self.optimizer_regime.get_lr()[0],
                                    "rehearsal_size": self.current_rehearsal_size,
                                    "batch": self.global_batch,
                                    "epoch": self.global_epoch})

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
                                    aug_size=self.aug_size,
                                    rehearsal_size=self.current_rehearsal_size,
                                )
                                self.batch_metrics.add(**batch_metrics_values)
                                self.batch_metrics.save()

                # Logging
                if self.batch % self.log_interval == 0 or self.batch == len(
                    data_regime.get_loader()
                ):
                    logging.debug(
                        "{0}: epoch: {1} [{2}/{3}]\t"
                        "Loss {meters[loss].avg:.4f}\t"
                        "Prec@1 {meters[prec1].avg:.3f}\t"
                        "Prec@5 {meters[prec5].avg:.3f}\t".format(
                            prefix,
                            self.epoch + 1,
                            self.batch + 1,
                            len(data_regime.get_loader()),
                            meters=meters,
                        )
                    )

                    if training and self.measure_performance():
                        metrics = self.perf_metrics.get(self.batch-1)
                        logging.debug(f"batch {self.batch} time {last_batch_time} sec")
                        logging.debug(
                            f"\t[C++] rehearsal_size {self.current_rehearsal_size}")
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
                           'augmented': self.aug_size - self.batch_size,
                           'rehearsal_size': self.current_rehearsal_size})
                progress.update(1)

                if not previous_task:
                    self.global_batch += 1
                    self.batch += 1

                timer = self.get_timer('load', previous_iteration=True)
                timer.__enter__()
                start_batch_time = time.perf_counter()

        if training:
            self.global_epoch += 1
            self.epoch += 1

            logging.info(f"\nCUMULATED VALUES:")
            logging.info(
                f"\trehearsal_size {self.current_rehearsal_size}")
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

    def _step(
        self,
        x,
        y,
        meters,
        training=False,
        previous_task=False,
    ):
        if training:
            # Get the representatives
            with self.get_timer('wait', previous_iteration=True):
                self.aug_size = self.dsl.wait()
                logging.debug(f"Received {self.aug_size - self.batch_size} samples from other nodes")

                if self.measure_performance():
                    cpp_metrics = self.dsl.get_metrics(self.batch)
                    self.perf_metrics.add(self.batch-1, cpp_metrics)

            # Assemble the minibatch
            with self.get_timer('assemble'):
                current_minibatch = self.get_current_augmented_minibatch()
                new_x = current_minibatch.x[:self.aug_size]
                new_y = current_minibatch.y[:self.aug_size]
                new_w = current_minibatch.w[:self.aug_size]

            if self.log_buffer and self.batch % self.log_interval == 0 and hvd.rank() == 0:
                repr_size = self.aug_size - self.batch_size
                if repr_size > 0:
                    captions = []
                    for label, weight in zip(current_minibatch.y[-repr_size:], current_minibatch.w[-repr_size:]):
                        captions.append(f"y={label.item()} w={weight.item()}")
                    display(f"aug_batch_{self.epoch}_{self.batch}", current_minibatch.x[-repr_size:], captions=captions)

            # In-advance preparation of next minibatch
            with self.get_timer('accumulate'):
                next_minibatch = self.get_next_augmented_minibatch()
                self.dsl.accumulate(x, y, next_minibatch.x, next_minibatch.y, next_minibatch.w)

            # Train
            with self.get_timer('train'):
                self.optimizer_regime.zero_grad()
                self.optimizer_regime.update(self.epoch, self.batch)

                if self.use_amp:
                    with autocast():
                        output = self.model(new_x)
                        loss = self.criterion(output, new_y)
                else:
                    output = self.model(new_x)
                    loss = self.criterion(output, new_y)

                # https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments
                total_weight = hvd.allreduce(torch.sum(new_w), name='total_weight', op=hvd.Sum)
                dw = new_w / total_weight

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
                meters["num_samples"].update(self.aug_size)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output, new_y, topk=(1, 5))
                # https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/10
                meters["loss"].update(loss.sum() / self.mask[new_y].sum(), new_x.size(0))

        else:
            if self.use_amp:
                with autocast():
                    output = self.model(x)
                    loss = nn.CrossEntropyLoss()(output, y)
            else:
                output = self.model(x)
                loss = nn.CrossEntropyLoss()(output, y)

            prec1, prec5 = accuracy(output, y, topk=(1, 5))
            meters["loss"].update(loss, x.size(0))

        meters["prec1"].update(prec1, x.size(0))
        meters["prec5"].update(prec5, x.size(0))

    def get_current_augmented_minibatch(self):
        return self.next_minibatches[self.global_batch % self.minibatches_ahead]

    def get_next_augmented_minibatch(self):
        return self.next_minibatches[(self.global_batch + 1) % self.minibatches_ahead]

    def get_timer(self, name, previous_iteration=False):
        batch = self.batch
        if previous_iteration:
            batch -= 1
        return MeasureTime(batch, name, self.perf_metrics, self.cuda, not self.measure_performance())