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

    def train_one_step(self, x, y, meters, step):
        """
        step: dict containing `task_id`, `epoch` and `batch` keys for logging purposes only
        """
        with get_timer(
            "train",
            step,
            batch_metrics=self.batch_metrics,
            dummy=not measure_performance(step),
        ):
            self.optimizer_regime.update(step)
            self.optimizer_regime.zero_grad()

            # Forward pass
            with autocast(enabled=self.use_amp):
                output = self.backbone(x)
                loss = self.criterion(output, y)

            assert not torch.isnan(loss).any(), "Loss is NaN, stopping training"

            # Backward pass
            self.scaler.scale(loss.sum() / loss.size(0)).backward()
            self.optimizer_regime.optimizer.synchronize()
            with self.optimizer_regime.optimizer.skip_synchronize():
                self.scaler.step(self.optimizer_regime.optimizer)
                self.scaler.update()

            # Measure accuracy and record metrics
            prec1, prec5 = accuracy(output, y, topk=(1, 5))
            meters["loss"].update(loss.sum() / loss.size(0))
            meters["prec1"].update(prec1, x.size(0))
            meters["prec5"].update(prec5, x.size(0))
            meters["num_samples"].update(x.size(0))

    def evaluate_one_step(self, x, y, meters, step):
        with autocast(enabled=self.use_amp):
            output = self.backbone(x)
            loss = self.criterion(output, y)

        prec1, prec5 = accuracy(output, y, topk=(1, 5))
        meters["loss"].update(loss.sum() / loss.size(0))
        meters["prec1"].update(prec1, x.size(0))
        meters["prec5"].update(prec5, x.size(0))
