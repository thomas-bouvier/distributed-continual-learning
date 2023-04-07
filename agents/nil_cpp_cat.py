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
from utils.utils import get_device, move_cuda, plot_representatives, synchronize_cuda, display
from utils.meters import AverageMeter, accuracy


class AugmentedMinibatch:
    def __init__(self, num_representatives, shape, device):
        self.x = torch.zeros(num_representatives, *shape, device=device)
        self.y = torch.randint(high=1000, size=(num_representatives,), device=device)
        self.w = torch.zeros(num_representatives, device=device)


class nil_cpp_cat_agent(Agent):
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
        super(nil_cpp_cat_agent, self).__init__(
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

        self.device = 'cuda' if self.cuda else 'cpu'
        self.rehearsal_size = config.get("rehearsal_size", 100)
        self.num_candidates = config.get("num_candidates", 20)
        self.num_representatives = config.get("num_representatives", 20)
        self.provider = config.get('provider', 'na+sm')
        self.discover_endpoints = config.get('discover_endpoints', True)
        self.cuda_rdma = config.get('cuda_rdma', False)

        self.num_reps = 0

        self.epoch_wait_time = 0
        self.epoch_assemble_time = 0
        self.last_batch_wait_time = 0
        self.last_batch_assemble_time = 0

    def before_all_tasks(self, train_data_regime):
        super().before_all_tasks(train_data_regime)

        shape = next(iter(train_data_regime.get_loader()))[0][0].size()
        self.next_minibatch = AugmentedMinibatch(self.num_representatives, shape, self.device)

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
        self.dsl.use_these_allocated_variables(self.next_minibatch.x, self.next_minibatch.y, self.next_minibatch.w)
        self.dsl.start()

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
        dataloader_iter = enumerate(data_regime.get_loader())

        start_batch_time = time.time()
        start_load_time = start_batch_time

        enable_tqdm = self.log_level in ('info') and hvd.rank() == 0
        with tqdm(total=len(data_regime.get_loader()),
              desc=f"Task #{self.task_id + 1} {prefix} epoch #{self.epoch + 1}",
              disable=not enable_tqdm) as progress:
            for i_batch, (x, y, t) in dataloader_iter:
                x, y = move_cuda(x, self.cuda), move_cuda(y, self.cuda)

                synchronize_cuda(self.cuda)
                self.last_batch_load_time = time.time() - start_load_time
                self.epoch_load_time += self.last_batch_load_time

                self._step(i_batch, x, y, meters, training=training, previous_task=previous_task)

                synchronize_cuda(self.cuda)
                self.last_batch_time = time.time() - start_batch_time
                epoch_time += self.last_batch_time

                self.last_batch_train_time = self.last_batch_time - self.last_batch_load_time - self.last_batch_wait_time - self.last_batch_assemble_time
                self.epoch_train_time += self.last_batch_train_time

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
                        f"\t[Python] batch load time {self.last_batch_load_time} sec ({self.last_batch_load_time*100/self.last_batch_time}%)")
                    logging.debug(
                        f"\t[Python] batch train time {self.last_batch_train_time} sec ({self.last_batch_train_time*100/self.last_batch_time}%)")
                    logging.debug(
                        f"\t[Python] batch wait time {self.last_batch_wait_time} sec ({self.last_batch_wait_time*100/self.last_batch_time}%)")
                    logging.debug(
                        f"\t[Python] batch assemble time {self.last_batch_assemble_time} sec ({self.last_batch_assemble_time*100/self.last_batch_time}%)")
                    perf_metrics = self.dsl.get_metrics(i_batch)
                    logging.debug(
                        f"\t[C++] batch accumulate time {perf_metrics[0]} sec")
                    logging.debug(
                        f"\t[C++] batch copy time {perf_metrics[1]} sec")
                    logging.debug(
                        f"\t[C++] bulk prepare time {perf_metrics[2]} sec")
                    logging.debug(
                        f"\t[C++] rpcs resolve time {perf_metrics[3]} sec")
                    logging.debug(
                        f"\t[C++] representatives copy time {perf_metrics[4]} sec")
                    logging.debug(
                        f"\t[C++] buffer update time {perf_metrics[5]} sec")
                    logging.debug(
                        f"\t[C++] rehearsal_size {self.get_rehearsal_size()}")

                    if hvd.rank() == 0:
                        if training and self.epoch < 25 and self.batch_metrics is not None:
                            batch_metrics_values = dict(
                                epoch=self.epoch,
                                batch=i_batch,
                                time=self.last_batch_time,
                                load_time=self.last_batch_load_time,
                                train_time=self.last_batch_train_time,
                                wait_time=self.last_batch_wait_time,
                                assemble_time=self.last_batch_assemble_time,
                                accumulate_time=perf_metrics[0],
                                copy_time=perf_metrics[1],
                                bulk_prepare_time=perf_metrics[2],
                                rpcs_resolve_time=perf_metrics[3],
                                representatives_copy_time=perf_metrics[4],
                                buffer_update_time=perf_metrics[5],
                                aug_size=self.aug_size,
                                num_reps=self.num_reps,
                            )
                            self.batch_metrics.add(**batch_metrics_values)
                            self.batch_metrics.save()

                        if not previous_task:
                            wandb.log({f"{prefix}_loss": meters["loss"].avg,
                                    "batch": self.global_batch,
                                    "epoch": self.global_epoch,
                                    "batch": i_batch,
                                    f"{prefix}_prec1": meters["prec1"].avg,
                                    f"{prefix}_prec5": meters["prec5"].avg})
                            if training:
                                wandb.log({"lr": self.optimizer_regime.get_lr()[0],
                                        "num_reps": self.get_rehearsal_size(),
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

                if not previous_task:
                    self.global_batch += 1
                    self.batch += 1

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
                f"\tepoch train time {self.epoch_train_time} sec ({self.epoch_train_time*100/epoch_time}%)")
            logging.info(
                f"\tepoch wait time {self.epoch_wait_time} sec ({self.epoch_wait_time*100/epoch_time}%)")
            logging.info(
                f"\tepoch assemble time {self.epoch_assemble_time} sec ({self.epoch_assemble_time*100/epoch_time}%)")
        self.epoch_load_time = 0
        self.epoch_train_time = 0
        self.epoch_wait_time = 0
        self.epoch_assemble_time = 0

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
        i_batch,
        x,
        y,
        meters,
        training=False,
        previous_task=False,
    ):
        new_x = x
        new_y = y
        w = torch.ones(self.batch_size, device=self.device)

        if training:
            self.optimizer_regime.zero_grad()
            self.optimizer_regime.update(self.epoch, self.batch)

            if self.global_batch == 0 and self.num_reps == 0:
                self.dsl.accumulate(x, y)

            # Get the representatives
            start_wait_time = time.time()
            self.aug_size = self.dsl.wait()
            self.last_batch_wait_time = time.time() - start_wait_time
            self.epoch_wait_time += self.last_batch_wait_time

            # Assemble the minibatch
            start_assemble_time = time.time()
            new_x = torch.cat((x, self.next_minibatch.x[:self.aug_size]))
            new_y = torch.cat((y, self.next_minibatch.y[:self.aug_size]))
            new_w = torch.cat((w, self.next_minibatch.w[:self.aug_size]))
            self.last_batch_assemble_time = time.time() - start_assemble_time
            self.epoch_assemble_time += self.last_batch_assemble_time

            logging.debug(f"Received {self.aug_size} samples from other nodes")
            if self.log_buffer and i_batch % self.log_interval == 0 and hvd.rank() == 0:
                num_reps = self.aug_size
                if num_reps > 0:
                    captions = []
                    for label, weight in zip(self.next_minibatch.y[-num_reps:], self.next_minibatch.w[-num_reps:]):
                        captions.append(f"y={label.item()} w={weight.item()}")
                    display(f"aug_batch_{self.epoch}_{i_batch}", self.next_minibatch.x[-num_reps:], captions=captions)

            # In-advance preparation of next minibatch
            self.dsl.accumulate(x, y)

            if self.use_amp:
                with autocast():
                    output = self.model(new_x)
                    loss = self.criterion(output, new_y)
            else:
                output = self.model(new_x)
                loss = self.criterion(output, new_y)
        else:
            if self.use_amp:
                with autocast():
                    output = self.model(new_x)
                    loss = nn.CrossEntropyLoss()(output, new_y)
            else:
                output = self.model(new_x)
                loss = nn.CrossEntropyLoss()(output, new_y)

        if training:
            # https://stackoverflow.com/questions/43451125/pytorch-what-a-re-the-gradient-arguments
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

            self.num_reps = self.dsl.get_rehearsal_size()
            meters["num_samples"].update(self.aug_size)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, new_y, topk=(1, 5))
            # https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/10
            meters["loss"].update(loss.sum() / self.mask[new_y].sum(), new_x.size(0))
        else:
            prec1, prec5 = accuracy(output, new_y, topk=(1, 5))
            meters["loss"].update(loss, new_x.size(0))

        meters["prec1"].update(prec1, new_x.size(0))
        meters["prec5"].update(prec5, new_x.size(0))

    def get_rehearsal_size(self):
        return self.num_reps