import horovod.torch as hvd
import torch

from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F

from modules import ContinualLearner, Buffer, BufferMode
from train.train import measure_performance
from utils.meters import get_timer, accuracy

__all__ = ["Der"]


class Der(ContinualLearner):
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

        try:
            self.alpha = config["alpha"]
        except Exception as exc:
            raise Exception("Parameter alpha is required for Der") from exc

        self.use_memory_buffer = True
        self.first_iteration = True
        self.activations = None

    def before_all_tasks(self, train_data_regime):
        self.buffer = Buffer(
            train_data_regime.total_num_classes,
            train_data_regime.sample_shape,
            self.batch_size,
            cuda=self._is_on_cuda(),
            mode=BufferMode.KD,
            **self.buffer_config,
        )

    def before_every_task(self, task_id, train_data_regime):
        super().before_every_task(task_id, train_data_regime)

        if task_id > self.buffer.augmentations_offset or self.nsys_run:
            self.buffer.enable_augmentations()

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
            if self.activations is not None:
                if not self.first_iteration:
                    _, _, _, buf_activations, buf_activations_rep = self.buffer.update(
                        [x_batch],
                        y_batch,
                        step,
                        batch_metrics=self.batch_metrics,
                        activations=[self.activations],
                    )
                else:
                    self.buffer.add_data(
                        [x_batch],
                        y_batch,
                        step,
                        batch_metrics=self.batch_metrics,
                        activations=[self.activations],
                    )
                    buf_activations_rep = None
                    self.first_iteration = False

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

                # Knowledge distillation
                if self.activations is not None and buf_activations_rep is not None:
                    buf_outputs = self.backbone(buf_activations_rep)
                    loss += self.alpha * F.mse_loss(buf_outputs, buf_activations[0])

                self.activations = output.data

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
                prec1, prec5 = accuracy(output, y, topk=(1, 5))
                meters["loss"].update(loss.mean())
                meters["prec1"].update(prec1, x.size(0))
                meters["prec5"].update(prec5, x.size(0))
                meters["num_samples"].update(x.size(0))
                meters["local_rehearsal_size"].update(self.buffer.get_size())

    def evaluate_one_step(self, data, meters):
        x, y, _ = data
        x, y = x.to(self._device()), y.long().to(self._device())

        with autocast(enabled=self.use_amp):
            output = self.backbone(x)
            loss = self.criterion(output, y)

        prec1, prec5 = accuracy(output, y, topk=(1, 5))
        meters["loss"].update(loss.sum() / loss.size(0))
        meters["prec1"].update(prec1, x.size(0))
        meters["prec5"].update(prec5, x.size(0))
        meters["num_samples"].update(x.size(0))
