import horovod.torch as hvd
import logging
import torch
import torch.nn as nn

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
        aug_x, aug_y, aug_w = self.buffer.update(
            x, y, step, batch_metrics=self.batch_metrics
        )

        with get_timer(
            "train",
            step,
            batch_metrics=self.batch_metrics,
            dummy=not measure_performance(step),
        ):
            self.optimizer_regime.update(step)
            self.optimizer_regime.zero_grad()
            
            repr_size = 3
            captions = []
            for label in zip(y[-repr_size:]):
                captions.append(f"y={label}")
            #display(f"img/aug_batch_{step['task_id']}_{step['epoch']}_{step['batch']}", x[-repr_size:], captions=captions)
            #if self.repr_size > 0:
            #    captions = []
            #    for y, w in zip(self.next_minibatch.y[-self.repr_size:], self.next_minibatch.w[-self.repr_size:]):
            #        captions.append(f"y={y.item()} w={w.item()}")
            #    display(f"aug_batch_{step["task_id"]}_{step["epoch"]}_{step["batch"]}", self.next_minibatch.x[-self.repr_size:], captions=captions)


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

    def evaluate_one_step(self, x, y, meters, step):
        with autocast(enabled=self.use_amp):
            output = self.backbone(x)
            loss = self.criterion(output, y)

        prec1, prec5 = accuracy(output, y, topk=(1, 5))
        meters["loss"].update(loss.sum() / loss.size(0))
        meters["prec1"].update(prec1, x.size(0))
        meters["prec5"].update(prec5, x.size(0))
