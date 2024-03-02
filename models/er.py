import os

import horovod.torch as hvd
import torch

from torch import nn
from torch.cuda.amp import autocast

from modules import ContinualLearner, Buffer
from train.train import measure_performance
from utils.meters import get_timer, accuracy

__all__ = ["Er"]


class Er(ContinualLearner):
    """Model for classifying images, "enriched" as ContinualLearner object."""

    def __init__(
        self,
        backbone: nn.Module,
        optimizer_regime,
        use_amp,
        nsys_run,
        batch_size,
        config,
        buffer_config,
        batch_metrics=None,
    ):
        super().__init__(
            backbone,
            optimizer_regime,
            use_amp,
            nsys_run,
            batch_size,
            config,
            buffer_config,
            batch_metrics,
        )

        self.use_memory_buffer = True

    def before_all_tasks(self, train_data_regime):
        self.buffer = Buffer(
            train_data_regime.total_num_classes,
            train_data_regime.sample_shape,
            self.batch_size,
            cuda=self._is_on_cuda(),
            **self.buffer_config,
        )

    def before_every_task(self, task_id, train_data_regime):
        super().before_every_task(task_id, train_data_regime)

        if task_id >= self.buffer.augmentations_offset or self.nsys_run:
            self.buffer.enable_augmentations()
        if task_id > 0 and self.nsys_run:
            # todo: this function doesn't exist, so the app will be killed.
            os.exit()

    def train_one_step(self, data, meters, step):
        """
        step: dict containing `task_id`, `epoch` and `batch` keys for logging purposes only
        """
        x, y, _ = data
        x, y = x.to(self._device()), y.long().to(self._device())

        # If making multiple backward passes per step, we need to cut the
        # current effective batch into local mini-batches.
        for i in range(0, len(x), self.batch_size):
            x_batch = x[i : i + self.batch_size]
            y_batch = y[i : i + self.batch_size]

            # Get data from the last iteration (blocking)
            aug_x, aug_y, _, _, _ = self.buffer.update(
                [x_batch],
                y_batch,
                step,
                batch_metrics=self.batch_metrics,
                activations=[],
            )
            aug_x = aug_x[0]

            with get_timer(
                "train",
                step,
                batch_metrics=self.batch_metrics,
                dummy=not measure_performance(step),
            ):
                # If performing multiple passes, update the optimizer only once.
                if i == 0:
                    self.optimizer_regime.update(step)
                    self.optimizer_regime.zero_grad()

                # Forward pass
                with autocast(enabled=self.use_amp):
                    output = self.backbone(aug_x)
                    loss = self.criterion(output, aug_y)

                # https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments
                # The aug_w variabled can be obtained via the update() primitive above.
                # total_weight = hvd.allreduce(
                #    torch.sum(aug_w), name="total_weight", op=hvd.Sum
                # )
                # dw = (
                #    aug_w
                #    / total_weight
                #    * self.batch_size
                #    * hvd.size()
                #    / self.batch_size
                # )

                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss.sum() / loss.size(0)).backward()
                    self.optimizer_regime.optimizer.synchronize()
                    with self.optimizer_regime.optimizer.skip_synchronize():
                        self.scaler.step(self.optimizer_regime.optimizer)
                        self.scaler.update()
                else:
                    (loss.sum() / loss.size(0)).backward()
                    self.optimizer_regime.step()

                # Measure accuracy and record metrics
                prec1, prec5 = accuracy(output, aug_y, topk=(1, 5))
                meters["loss"].update(loss.sum() / loss.size(0))
                meters["prec1"].update(prec1, aug_x.size(0))
                meters["prec5"].update(prec5, aug_x.size(0))
                meters["num_samples"].update(aug_x.size(0))
                meters["local_rehearsal_size"].update(self.buffer.get_size())

    def evaluate_one_step(self, data, meters, step):
        x, y, _ = data
        x, y = x.to(self._device()), y.long().to(self._device())

        with autocast(enabled=self.use_amp):
            output = self.backbone(x)
            loss = self.criterion(output, y)

        # TODO: if true for multiple iterations, trigger this
        # assert not torch.isnan(loss).any(), "Loss is NaN, stopping training"

        prec1, prec5 = accuracy(output, y, topk=(1, 5))
        meters["loss"].update(loss.sum() / loss.size(0))
        meters["prec1"].update(prec1, x.size(0))
        meters["prec5"].update(prec5, x.size(0))
        meters["num_samples"].update(x.size(0))

    def train_recon_one_step(self, data, meters, step):
        """
        step: dict containing `task_id`, `epoch` and `batch` keys for logging purposes only
        """
        x, y, amp, ph, _ = data
        x, y, amp, ph = (
            x.to(self._device()),
            y.to(self._device()),
            amp.to(self._device()),
            ph.to(self._device()),
        )

        # If making multiple backward passes per step, we need to cut the
        # current effective batch into local mini-batches.
        for i in range(0, len(x), self.batch_size):
            x_batch = x[i : i + self.batch_size]
            y_batch = y[i : i + self.batch_size]
            amp_batch = amp[i : i + self.batch_size]
            ph_batch = ph[i : i + self.batch_size]

            # Get data from the last iteration (blocking)
            aug_x, _, _, _, _ = self.buffer.update(
                [x_batch, amp_batch, ph_batch],
                y_batch,
                step,
                batch_metrics=self.batch_metrics,
            )

            with get_timer(
                "train",
                step,
                batch_metrics=self.batch_metrics,
                dummy=not measure_performance(step),
            ):
                # If performing multiple passes, update the optimizer only once.
                if i == 0:
                    self.optimizer_regime.update(step)
                    self.optimizer_regime.zero_grad()

                # Forward pass
                with autocast(enabled=self.use_amp):
                    amp_output, ph_output = self.backbone(aug_x[0])
                    amp_loss = self.criterion(amp_output, aug_x[1])
                    ph_loss = self.criterion(ph_output, aug_x[2])
                    loss = amp_loss + ph_loss

                # Backward pass
                self.scaler.scale(loss.sum() / loss.size(0)).backward()
                self.optimizer_regime.optimizer.synchronize()
                with self.optimizer_regime.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer_regime.optimizer)
                    self.scaler.update()

                meters["loss"].update(loss.sum() / aug_x[0].size(0))
                meters["loss_amp"].update(amp_loss.sum() / aug_x[0].size(0))
                meters["loss_ph"].update(ph_loss.sum() / aug_x[0].size(0))
                meters["num_samples"].update(aug_x[0].size(0))
                meters["local_rehearsal_size"].update(self.buffer.get_size())

    def evaluate_recon_one_step(self, data, meters, step):
        x, _, amp, ph, _ = data
        x, amp, ph = x.to(self._device()), amp.to(self._device()), ph.to(self._device())

        with autocast(enabled=self.use_amp):
            amp_output, ph_output = self.backbone(x)
            amp_loss = self.criterion(amp_output, amp)
            ph_loss = self.criterion(ph_output, ph)
            loss = amp_loss + ph_loss

        assert not torch.isnan(loss).any(), "Validate loss is NaN, stopping training"

        meters["loss"].update(loss.sum() / loss.size(0))
        meters["loss_amp"].update(amp_loss.sum() / amp_loss.size(0))
        meters["loss_ph"].update(ph_loss.sum() / ph_loss.size(0))
        meters["num_samples"].update(x.size(0))
