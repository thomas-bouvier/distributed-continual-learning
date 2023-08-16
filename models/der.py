import horovod.torch as hvd
import logging
import torch
import torch.nn as nn

from torch.cuda.amp import autocast

from modules import ContinualLearner, Buffer
from train.train import measure_performance
from utils.meters import get_timer, accuracy

from torch.nn import functional as F

__all__ = ["Der"]


class Der(ContinualLearner):
    """Model for classifying images, "enriched" as ContinualLearner object."""

    def __init__(
        self,
        backbone: nn.Module,
        optimizer_regime,
        use_amp,
        batch_size,
        config,
        buffer_config,
        batch_metrics=None,
    ):
        super(Der, self).__init__(
            backbone,
            optimizer_regime,
            use_amp,
            batch_size,
            config,
            buffer_config,
            batch_metrics,
        )

        self.use_memory_buffer = True
        self.alpha, _ = get_alpha_beta_parms()
        self.temp = False

    def before_all_tasks(self, train_data_regime):
        self.buffer = Buffer(
            train_data_regime.total_num_classes,
            train_data_regime.sample_shape,
            self.batch_size,
            cuda=self._is_on_cuda(),
            **self.buffer_config,
        )

        x, y, _ = next(iter(train_data_regime.get_loader(0)))
        self.buffer.add_data(x, y, dict(batch=-1))

    def before_every_task(self, task_id, train_data_regime):
        super().before_every_task(task_id, train_data_regime)

        if task_id > 0:
            self.buffer.enable_augmentations()

    def train_one_step(self, x, y, meters, step):
        """
        step: dict containing `task_id`, `epoch` and `batch` keys for logging purposes only
        """
        # Get data from the last iteration (blocking)

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

            # loss = loss.sum() / loss.size(0)

            if self.temp:
                buf = self.buffer._Buffer__get_current_augmented_minibatch(step)
                buf_inputs, buf_logits = buf.x, buf.logits
                buf_outputs = self.backbone(buf_inputs)
                loss += self.alpha * F.mse_loss(buf_outputs, buf_logits)

            elif step["task_id"] > 0:
                self.temp = True

            self.scaler.scale(loss.sum() / loss.size(0)).backward()

            self.optimizer_regime.optimizer.synchronize()
            with self.optimizer_regime.optimizer.skip_synchronize():
                self.scaler.step(self.optimizer_regime.optimizer)
                self.scaler.update()

            # If scaler doesn't work use this classic backward
            # loss.backward()

            self.buffer.update_with_logits(
                x, y, output.data, step, batch_metrics=self.batch_metrics
            )

            # If scaler doesn't work use this classic backward
            # self.optimizer_regime.step()

            # Measure accuracy and record metrics
            prec1, prec5 = accuracy(output, y, topk=(1, 5))
            meters["loss"].update(loss.mean())
            meters["prec1"].update(prec1, x.size(0))
            meters["prec5"].update(prec5, x.size(0))
            meters["num_samples"].update(x.size(0))
            meters["local_rehearsal_size"].update(self.buffer.get_size())

    def evaluate_one_step(self, x, y, meters, step):
        with autocast(enabled=self.use_amp):
            output = self.backbone(x)
            loss = self.criterion(output, y)

        prec1, prec5 = accuracy(output, y, topk=(1, 5))
        meters["loss"].update(loss.sum() / loss.size(0))  # loss IF SCALER DESACTIVATED
        meters["prec1"].update(prec1, x.size(0))
        meters["prec5"].update(prec5, x.size(0))
