import horovod.torch as hvd
import torch

from torch.cuda.amp import autocast
from torch import nn
from torch.nn import functional as F

from modules import ContinualLearner, Buffer, BufferMode
from train.train import measure_performance
from utils.meters import get_timer, accuracy


__all__ = ["Derpp"]


class Derpp(ContinualLearner):
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
            self.beta = config["beta"]
        except Exception as exc:
            raise Exception("Parameters alpha and beta are required for Derpp") from exc

        if self.beta != 1:
            raise Exception("As rehearsal is part of minibatch augmentations, parameter beta must be set to 1")

        self.use_memory_buffer = True
        self.first_iteration = True
        self.activations = None
        self.activations_amp = None
        self.activations_ph = None

    def before_all_tasks(self, train_data_regime):
        self.buffer = Buffer(
            train_data_regime.total_num_classes,
            train_data_regime.sample_shape,
            self.batch_size,
            half_precision=self.use_amp,
            cuda=self._is_on_cuda(),
            mode=BufferMode.REHEARSAL_KD,
            **self.buffer_config,
        )

    def before_every_task(self, task_id, train_data_regime):
        super().before_every_task(task_id, train_data_regime)

        if task_id >= self.buffer.augmentations_offset or self.nsys_run:
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
                    (
                        aug_x,
                        aug_y,
                        _,
                        buf_activations,
                        buf_activations_rep,
                    ) = self.buffer.update(
                        [self.save_x],
                        self.save_y,
                        step,
                        batch_metrics=self.batch_metrics,
                        activations=[self.activations],
                    )
                    aug_x = aug_x[0] # only one item in the tuple
                else:
                    self.buffer.add_data(
                        [self.save_x],
                        self.save_y,
                        step,
                        batch_metrics=self.batch_metrics,
                        activations=[self.activations],
                    )
                    aug_x = x_batch
                    aug_y = y_batch
                    buf_activations_rep = None
                    self.first_iteration = False
            else:
                aug_x = x_batch
                aug_y = y_batch
                buf_activations_rep = None
                self.first_iteration = True

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
                    output = self.backbone(aug_x)
                    loss = self.criterion(output, aug_y)

                # Knowledge distillation
                if step["task_id"] >= self.buffer.soft_augmentations_offset:
                    if self.activations is not None and buf_activations_rep is not None:
                        buf_outputs = self.backbone(buf_activations_rep)
                        loss += self.alpha * F.mse_loss(buf_outputs, buf_activations[0])

                self.save_x = aug_x[: self.batch_size]
                self.save_y = aug_y[: self.batch_size]
                self.activations = output.data[: self.batch_size]

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
                meters["loss"].update(loss.mean())
                meters["prec1"].update(prec1, aug_x.size(0))
                meters["prec5"].update(prec5, aug_x.size(0))
                meters["num_samples"].update(aug_x.size(0))
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

    def train_recon_one_step(self, data, meters, step):
        """
        The first forward pass must be done before interacting with the buffer,
        as the latter requires neural activations with Der++. A consequence of
        that is that a second forward pass has to be done on representatives
        only, alongside their activations.

        Params:
            step: dict containing `task_id`, `epoch` and `batch` keys for
            logging purposes only
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
            if self.activations_amp is not None and self.activations_ph is not None:
                if not self.first_iteration:
                    (
                        aug_x,
                        _,
                        _,
                        buf_activations,
                        buf_activations_rep,
                    ) = self.buffer.update(
                        [x_batch, amp_batch, ph_batch],
                        y_batch,
                        step,
                        batch_metrics=self.batch_metrics,
                        activations=[self.activations_amp, self.activations_ph],
                    )
                else:
                    self.buffer.add_data(
                        [x_batch, amp_batch, ph_batch],
                        y_batch,
                        step,
                        batch_metrics=self.batch_metrics,
                        activations=[self.activations_amp, self.activations_ph],
                    )
                    aug_x = [x_batch, amp_batch, ph_batch]
                    buf_activations_rep = None
                    self.first_iteration = False
            else:
                aug_x = [x_batch, amp_batch, ph_batch]
                buf_activations_rep = None
                self.first_iteration = True

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

                # Knowledge distillation
                if (
                    self.activations_amp is not None
                    and self.activations_ph is not None
                    and buf_activations_rep is not None
                ):
                    buf_output_amp, buf_output_ph = self.backbone(buf_activations_rep)
                    loss += self.alpha * (
                        F.mse_loss(buf_output_amp, buf_activations[0])
                        + F.mse_loss(buf_output_ph, buf_activations[1])
                    )

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

                meters["loss"].update(loss.sum() / loss.size(0))
                # meters["loss_amp"].update(amp_loss.sum() / aug_x.size(0))
                # meters["loss_ph"].update(ph_loss.sum() / aug_x.size(0))
                # meters["num_samples"].update(aug_x.size(0))
                meters["local_rehearsal_size"].update(self.buffer.get_size())

    def evaluate_recon_one_step(self, data, meters):
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

    def after_all_tasks(self):
        self.buffer.finalize()
