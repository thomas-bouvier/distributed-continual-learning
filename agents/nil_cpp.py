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
from cpp_loader import rehearsal

from agents.base import Agent
from torch.cuda.amp import GradScaler, autocast
from utils.utils import get_device, move_cuda, plot_representatives, synchronize_cuda, display
from utils.meters import AverageMeter, accuracy


class AugmentedMinibatch:
    def __init__(self, batch_size, num_samples, shape, device):
        self.x = torch.zeros(
            batch_size + num_samples, *shape, device=device)
        self.y = torch.randint(high=1000, size=(
            batch_size + num_samples,), device=device)
        self.w = torch.zeros(
            batch_size + num_samples, device=device)


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

        self.device = 'cuda' if self.cuda else 'cpu'
        self.num_representatives = config.get("num_representatives", 60)
        self.num_candidates = config.get("num_candidates", 20)
        self.num_samples = config.get("num_samples", 20)
        self.provider = config.get('provider', 'na+sm://')
        self.discover_endpoints = config.get('discover_endpoints', True)
        self.cuda_rdma = config.get('cuda_rdma', False)

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
            train_data_regime.total_num_classes, self.num_representatives, self.num_candidates,
            ctypes.c_int64(torch.random.initial_seed() + hvd.rank()).value,
            ctypes.c_uint16(hvd.rank()).value, self.provider,
            1, list(shape), self.cuda_rdma, self.discover_endpoints, self.log_level not in ('info')
        )

        self.minibatches_ahead = 2
        self.next_minibatches = []
        for i in range(self.minibatches_ahead):
            self.next_minibatches.append(AugmentedMinibatch(self.batch_size, self.num_samples, shape, self.device))

    def before_every_task(self, task_id, train_data_regime):
        self.task_id = task_id
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

        # Create mask so the loss is only used for classes learnt during this task
        self.mask = torch.tensor(train_data_regime.classes_mask, device=self.device).float()
        self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')

    """Forward pass for the current epoch"""
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
        with tqdm(total=len(data_regime.get_loader()),
              desc=f"Task #{self.task_id + 1} {prefix} epoch #{self.epoch + 1}",
              disable=self.log_level not in ('info')) as progress:
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
                if training:
                    # https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/10
                    meters["loss"].update(loss.sum() / self.mask[y].sum())
                else:
                    meters["loss"].update(loss)
                meters["prec1"].update(prec1, x.size(0))
                meters["prec5"].update(prec5, x.size(0))

                if i_batch % self.log_interval == 0 or i_batch == len(data_regime.get_loader()):
                    logging.debug(
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
                        f"\tnum_representatives {self.get_num_representatives()}")

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
                progress.set_postfix({'loss': meters["loss"].avg.item(),
                           'accuracy': 100. * meters["prec1"].avg.item(),
                           'representatives': self.num_reps})
                progress.update(1)

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

        if training:
            if self.global_steps == 0 and self.num_reps == 0:
                current_minibatch = self.get_current_augmented_minibatch()
                self.dsl.accumulate(x, y, current_minibatch.x, current_minibatch.y, current_minibatch.w)

            # Get the representatives
            start_wait_time = time.time()
            aug_size = self.dsl.wait()
            self.last_batch_wait_time = time.time() - start_wait_time
            self.epoch_wait_time += self.last_batch_wait_time

            current_minibatch = self.get_current_augmented_minibatch()
            self.aug_x = current_minibatch.x
            self.aug_y = current_minibatch.y
            self.aug_w = current_minibatch.w

            logging.debug(f"Received {aug_size - self.batch_size} samples from other nodes")
            if self.log_buffer and i_batch % self.log_interval == 0 and hvd.rank() == 0:
                num_reps = aug_size - self.batch_size
                if num_reps > 0:
                    display(f"aug_batch_{self.epoch}_{i_batch}", self.aug_x[-num_reps:], captions=self.aug_y[-num_reps:], cuda=self.cuda)

            # In-advance preparation of next minibatch
            start_acc_time = time.time()
            next_minibatch = self.get_next_augmented_minibatch()
            self.dsl.accumulate(x, y, next_minibatch.x, next_minibatch.y, next_minibatch.w)
            self.last_batch_acc_time = time.time() - start_acc_time
            self.epoch_acc_time += self.last_batch_acc_time

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
                    loss = nn.CrossEntropyLoss()(output, y)
            else:
                output = self.model(x)
                loss = nn.CrossEntropyLoss()(output, y)

        if training:
            # https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments
            total_weight = hvd.allreduce(torch.sum(self.aug_w), name='total_weight', op=hvd.Sum)
            dw = self.aug_w / total_weight

            if self.use_amp:
                self.scaler.scale(loss).backward(dw)
                self.optimizer_regime.optimizer.synchronize()
                with self.optimizer_regime.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer_regime.optimizer)
                    self.scaler.update()
            else:
                loss.backward(dw)
                self.optimizer_regime.step()

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

    def get_current_augmented_minibatch(self):
        return self.next_minibatches[self.global_steps % self.minibatches_ahead]

    def get_next_augmented_minibatch(self):
        return self.next_minibatches[(self.global_steps + 1) % self.minibatches_ahead]