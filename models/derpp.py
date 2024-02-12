import horovod.torch as hvd
import torch
import torch.nn as nn

from torch.cuda.amp import autocast

from modules import ContinualLearner, Buffer
from train.train import measure_performance
from utils.meters import get_timer, accuracy

from torch.nn import functional as F

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
        except:
            raise Exception("Parameters alpha and beta are required for Derpp")

        self.use_memory_buffer = True
        self.temp = False

    def before_all_tasks(self, train_data_regime):
        self.buffer_config["implementation"] = "flyweight"

        self.buffer = Buffer(
            train_data_regime.total_num_classes,
            train_data_regime.sample_shape,
            self.batch_size,
            cuda=self._is_on_cuda(),
            num_samples_per_activation=2,
            **self.buffer_config,
        )

        # x, y, _ = next(iter(train_data_regime.get_loader(0)))
        # self.buffer.add_data(x, y, dict(batch=-1))

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

            loss = loss.sum() / loss.size(0)

            if self.temp:
                # Same part common to DER
                buf = self.buffer._Buffer__get_current_augmented_minibatch(step)
                buf_inputs, buf_logits = buf.x, buf.logits
                buf_outputs = self.backbone(buf_inputs)
                loss += self.alpha * F.mse_loss(buf_outputs, buf_logits)

                # Additional Part
                buf = self.buffer._Buffer__get_current_augmented_minibatch(step)
                buf_inputs, buf_y = buf.x, buf.y
                buf_outputs = self.backbone(buf_inputs)
                buf_loss = self.criterion(buf_outputs, buf_y)
                buf_loss = buf_loss.sum() / buf_loss.size(0)
                loss += self.beta * buf_loss

            elif step["task_id"] > 0:
                self.temp = True

            self.scaler.scale(loss).backward()

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

            # If performing multiple passes, update the optimizer only once.
            if i == 0:
                self.optimizer_regime.update(step)
                self.optimizer_regime.zero_grad()

            # Get the activations for the buffer
            # Forward pass
            with autocast(enabled=self.use_amp):
                amp_output, ph_output = self.backbone(x_batch)
                amp_loss = self.criterion(amp_output, amp_batch)
                ph_loss = self.criterion(ph_output, ph_batch)
                loss = amp_loss + ph_loss
                total_loss = loss.sum() / loss.size(0)

            # Get data from the last iteration (blocking)
            aug_x, aug_ground_truth, aug_y, aug_w, aug_logits = self.buffer.update(
                x_batch,
                y_batch,
                step,
                batch_metrics=self.batch_metrics,
                logits=[amp_output.data, ph_output.data],
                ground_truth=[amp_batch, ph_batch],
            )

            # todo: timing not correct with that one
            with get_timer(
                "train",
                step,
                batch_metrics=self.batch_metrics,
                dummy=not measure_performance(step),
            ):
                if step["task_id"] > self.buffer.soft_augmentations_offset:
                    # Forward pass using the soft labels
                    with autocast(enabled=self.use_amp):
                        amp_output1, ph_output1 = self.backbone(
                            aug_x[self.batch_size :]
                        )
                        amp_loss1 = F.mse_loss(
                            amp_output1, aug_logits[0][self.batch_size :]
                        )
                        ph_loss1 = F.mse_loss(
                            ph_output1, aug_logits[1][self.batch_size :]
                        )
                        loss1 = self.alpha * (amp_loss1 + ph_loss1)
                        total_loss += loss1

                if self.buffer.augmentations_enabled:
                    # Get data from the last iteration (blocking)
                    (
                        aug_x_2,
                        aug_ground_truth_2,
                        aug_y_2,
                        aug_w_2,
                        aug_logits_2,
                    ) = self.buffer.update(
                        x_batch,  # not actually adding anything to the buffer
                        y_batch,
                        step,
                        batch_metrics=self.batch_metrics,
                        logits=[amp_output.data, ph_output.data],
                        ground_truth=[amp_batch, ph_batch],
                        derpp=True,
                    )

                    # Forward pass using the hard labels
                    with autocast(enabled=self.use_amp):
                        amp_output2, ph_output2 = self.backbone(
                            aug_x_2[self.batch_size :]
                        )
                        amp_loss2 = self.criterion(
                            amp_output2, aug_ground_truth_2[0][self.batch_size :]
                        )
                        ph_loss2 = self.criterion(
                            ph_output2, aug_ground_truth_2[1][self.batch_size :]
                        )
                        loss2 = self.beta * (amp_loss2 + ph_loss2)
                        total_loss += loss2.sum() / loss2.size(0)

                # Backward pass
                self.scaler.scale(total_loss).backward()
                self.optimizer_regime.optimizer.synchronize()
                with self.optimizer_regime.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer_regime.optimizer)
                    self.scaler.update()

                meters["loss"].update(total_loss)
                # meters["loss_amp"].update(amp_loss.sum() / aug_x.size(0))
                # meters["loss_ph"].update(ph_loss.sum() / aug_x.size(0))
                # meters["num_samples"].update(aug_x.size(0))
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
