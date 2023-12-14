import copy
import horovod.torch as hvd
import logging
import torch
import torch.nn as nn

from torch.cuda.amp import autocast

from modules import ContinualLearner
from train.train import measure_performance
from utils.meters import get_timer, accuracy

__all__ = ["Vanilla"]


class Vanilla(ContinualLearner):
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
        super(Vanilla, self).__init__(
            backbone,
            optimizer_regime,
            use_amp,
            nsys_run,
            batch_size,
            config,
            buffer_config,
            batch_metrics,
        )

        self.use_memory_buffer = False

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
                    output = self.backbone(x_batch)
                    loss = self.criterion(output, y_batch)

                assert not torch.isnan(loss).any(), "Loss is NaN, stopping training"

                # Backward pass
                self.scaler.scale(loss.sum() / loss.size(0)).backward()
                self.optimizer_regime.optimizer.synchronize()
                with self.optimizer_regime.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer_regime.optimizer)
                    self.scaler.update()

                # Measure accuracy and record metrics
                prec1, prec5 = accuracy(output, y_batch, topk=(1, 5))
                meters["loss"].update(loss.sum() / loss.size(0))
                meters["prec1"].update(prec1, x_batch.size(0))
                meters["prec5"].update(prec5, x_batch.size(0))
                meters["num_samples"].update(x_batch.size(0))

    def evaluate_one_step(self, data, meters, step):
        x, y, _ = data
        x, y = x.to(self._device()), y.long().to(self._device())

        with autocast(enabled=self.use_amp):
            output = self.backbone(x)
            loss = self.criterion(output, y)

        prec1, prec5 = accuracy(output, y, topk=(1, 5))
        meters["loss"].update(loss.sum() / loss.size(0))
        meters["prec1"].update(prec1, x.size(0))
        meters["prec5"].update(prec5, x.size(0))

    def train_recon_one_step(self, data, meters, step):
        """
        step: dict containing `task_id`, `epoch` and `batch` keys for logging purposes only
        """
        x, _, amp, ph, _ = data
        x, amp, ph = x.to(self._device()), amp.to(self._device()), ph.to(self._device())

        # If making multiple backward passes per step, we need to cut the
        # current effective batch into local mini-batches.
        for i in range(0, len(x), self.batch_size):
            x_batch = x[i : i + self.batch_size]
            amp_batch = amp[i : i + self.batch_size]
            ph_batch = ph[i : i + self.batch_size]

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
                    amp_output, ph_output = self.backbone(x_batch)
                    amp_loss = self.criterion(amp_output, amp_batch)
                    ph_loss = self.criterion(ph_output, ph_batch)
                    loss = amp_loss + ph_loss

                # Backward pass
                self.scaler.scale(loss.sum() / loss.size(0)).backward()
                self.optimizer_regime.optimizer.synchronize()
                with self.optimizer_regime.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer_regime.optimizer)
                    self.scaler.update()

                meters["loss"].update(loss.sum() / loss.size(0))
                meters["loss_amp"].update(amp_loss.sum() / amp_loss.size(0))
                meters["loss_ph"].update(ph_loss.sum() / ph_loss.size(0))
                meters["num_samples"].update(x_batch.size(0))

    def evaluate_recon_one_step(self, data, meters, step):
        x, _, amp, ph, _ = data
        x, amp, ph = x.to(self._device()), amp.to(self._device()), ph.to(self._device())

        with autocast(enabled=self.use_amp):
            amp_output, ph_output = self.backbone(x)
            amp_loss = self.criterion(amp_output, amp)
            ph_loss = self.criterion(ph_output, ph)
            loss = amp_loss + ph_loss

        meters["loss"].update(loss.sum() / loss.size(0))
        meters["loss_amp"].update(amp_loss.sum() / amp_loss.size(0))
        meters["loss_ph"].update(ph_loss.sum() / ph_loss.size(0))
        meters["num_samples"].update(x.size(0))
