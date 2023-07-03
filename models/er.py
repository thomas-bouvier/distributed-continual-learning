import copy
import horovod.torch as hvd
import logging
import torch
import torch.nn as nn

from torch.cuda.amp import autocast

from modules import ContinualLearner, Buffer
from train.train import measure_performance
from utils.meters import get_timer, accuracy
from utils.log import PerformanceResultsLog


class Er(ContinualLearner):
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
        super(Er, self).__init__(
            backbone,
            optimizer_regime,
            use_amp,
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
            budget_per_class=self.buffer_config.get("rehearsal_size"),
            num_candidates=self.buffer_config.get("num_candidates"),
            num_representatives=self.buffer_config.get("num_representatives"),
            provider=self.buffer_config.get("provider"),
            discover_endpoints=self.buffer_config.get("discover_endpoints"),
            cuda=self._is_on_cuda(),
            cuda_rdma=self.buffer_config.get("cuda_rdma"),
            implementation=self.buffer_config.get("implementation"),
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
        w = torch.ones(self.batch_size, device=self._device())

        # Get data from the last iteration (blocking)
        aug_x, aug_y, aug_w = self.buffer.update(
            x, y, w, step, perf_metrics=self.perf_metrics
        )

        with get_timer(
            "train",
            step["batch"],
            perf_metrics=self.perf_metrics,
            dummy=not measure_performance(step),
        ):
            self.optimizer_regime.update(step["epoch"], step["batch"])
            self.optimizer_regime.zero_grad()

            # Forward pass
            with autocast(enabled=self.use_amp):
                output = self.backbone(aug_x)
                loss = self.criterion(output, aug_y)

            # TODO: if true for multiple iterations, trigger this
            # assert not torch.isnan(loss).any(), "Loss is NaN, stopping training"

            # https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments
            total_weight = hvd.allreduce(
                torch.sum(aug_w), name="total_weight", op=hvd.Sum
            )
            dw = aug_w / total_weight * self.batch_size * hvd.size() / self.batch_size

            # Backward pass
            self.scaler.scale(loss).backward(dw)
            self.optimizer_regime.optimizer.synchronize()
            with self.optimizer_regime.optimizer.skip_synchronize():
                self.scaler.step(self.optimizer_regime.optimizer)
                self.scaler.update()

            # Measure accuracy and record metrics
            prec1, prec5 = accuracy(output, aug_y, topk=(1, 5))
            meters["loss"].update(loss.sum() / aug_y.size(0))
            meters["prec1"].update(prec1, aug_x.size(0))
            meters["prec5"].update(prec5, aug_x.size(0))
            meters["num_samples"].update(aug_x.size(0))
            meters["local_rehearsal_size"].update(self.buffer.get_size())
