import horovod.torch as hvd
import logging
import torch
import torch.nn as nn

from torch.cuda.amp import autocast

from modules import ContinualLearner, Buffer
from train.train import measure_performance
from utils.meters import get_timer, accuracy

from torch.optim import SGD
import copy

import numpy
from torchsummary import summary


__all__ = ["Hal"]


class Hal(ContinualLearner):
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
        super(Hal, self).__init__(
            backbone,
            optimizer_regime,
            use_amp,
            batch_size,
            config,
            buffer_config,
            batch_metrics,
        )
        try:
            self.gamma = config["gamma"]
            self.beta = config["beta"]
            self.hal_lambda = config["hal_lambda"]
            self.lr = config["lr"]
        except:
            raise Exception("Parameters beta, gamma and hal lambda is required for HAL")

        self.use_memory_buffer = True
        self.finetuning_epochs = 1
        self.step = None
        self.x = None
        self.y = None
        self.spare_model = copy.deepcopy(self.backbone)
        # self.spare_model.to(device)
        self.spare_opt = SGD(self.spare_model.parameters(), lr=self.lr)
        self.anchor_optimization_steps = 100
        # self.gamma = 0.1
        # self.beta = 0.5
        # self.hal_lambda = 0.1

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

    def after_every_task(self, task_id, train_data_regime):
        self.get_anchors(task_id, train_data_regime)
        del self.phi

    def get_params(self, backbone) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(backbone.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, backbone, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        progress = 0
        for pp in list(backbone.parameters()):
            cand_params = new_params[
                progress : progress + torch.tensor(pp.size()).prod()
            ].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_anchors(self, task_id, train_data_regime):
        # params = self.backbone.parameters()
        # theta_t = [x.detach().clone() for x in params]
        # theta_t = self.backbone.state_dict()  # state_dict au lieu de parameters ?
        # self.spare_model.load_state_dict(theta_t)
        theta_t = self.get_params(self.backbone).detach().clone()
        self.set_params(self.spare_model, theta_t)

        for _ in range(self.finetuning_epochs):  # Fine Tune
            inputs, labels = self.x.clone(), self.y.clone()
            self.spare_opt.zero_grad()
            out = self.spare_model(inputs)
            loss = self.criterion(out, labels)
            (loss.sum() / loss.size(0)).backward()
            self.spare_opt.step()

        theta_m = self.get_params(self.backbone).detach().clone()

        classes_for_this_task = torch.where(
            torch.tensor(train_data_regime.classes_mask)
        )[0]

        for a_class in classes_for_this_task:
            e_t = torch.rand(
                torch.Size([2] + list(self.x.shape[1:])), requires_grad=True
            )  # device=self.device
            e_t_opt = SGD([e_t], lr=self.lr)
            for i in range(self.anchor_optimization_steps):
                e_t_opt.zero_grad()
                cum_loss = 0

                self.spare_opt.zero_grad()
                self.set_params(self.spare_model, theta_m.detach().clone())
                loss = -torch.sum(
                    self.criterion(
                        self.spare_model(e_t), torch.full((2,), a_class.item())
                    )
                )  # .to(self.device)
                loss.backward()
                cum_loss += loss.item()

                self.spare_opt.zero_grad()
                self.set_params(self.spare_model, theta_m.detach().clone())
                loss = torch.sum(
                    self.criterion(
                        self.spare_model(e_t), torch.full((2,), a_class.item())
                    )
                )  # .to(self.device)
                loss.backward()
                cum_loss += loss.item()

                self.spare_opt.zero_grad()
                loss = torch.sum(
                    self.gamma
                    * (self.spare_model.forward_features(e_t)[0] - self.phi[0]) ** 2
                )
                assert not self.phi.requires_grad
                loss.backward()
                cum_loss += loss.item()

                e_t_opt.step()

            e_t = e_t.detach()
            e_t.requires_grad = False
            self.anchors = torch.cat((self.anchors, e_t[0].unsqueeze(0)))
            del e_t
            print("Total anchors:", len(self.anchors))

    def train_one_step(self, x, y, meters, step):
        aug_x, aug_y, aug_w = self.buffer.update(
            x, y, step, batch_metrics=self.batch_metrics
        )

        self.x = x
        self.y = y

        self.step = step

        if not hasattr(self, "phi"):
            with torch.no_grad():
                self.phi = torch.zeros_like(
                    self.backbone.forward_features(x[:2]), requires_grad=False
                )
            assert not self.phi.requires_grad
        if not hasattr(self, "anchors"):
            self.anchors = torch.zeros(
                tuple([0] + list(torch.Size(list(self.x.shape[1:]))))
            )  # .to(self.device)

        old_weights = self.get_params(self.backbone).detach().clone()

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
                firstloss = self.criterion(output, aug_y)

            # TODO: if true for multiple iterations, trigger this
            # assert not torch.isnan(loss).any(), "Loss is NaN, stopping training"

            # https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments
            total_weight = hvd.allreduce(
                torch.sum(aug_w), name="total_weight", op=hvd.Sum
            )
            dw = aug_w / total_weight * self.batch_size * hvd.size() / self.batch_size

            # Backward pass
            self.scaler.scale(firstloss.sum() / firstloss.size(0)).backward()
            self.optimizer_regime.optimizer.synchronize()
            with self.optimizer_regime.optimizer.skip_synchronize():
                self.scaler.step(self.optimizer_regime.optimizer)
                self.scaler.update()

            if len(self.anchors) > 0:
                anchors_points = torch.tensor(1)
                with torch.no_grad():
                    pred_anchors = self.backbone(self.anchors)

                self.set_params(self.backbone, old_weights)
                pred_anchors -= self.backbone(self.anchors)
                loss = self.hal_lambda * (pred_anchors**2).mean()
                loss.backward()
                self.optimizer_regime.step()

            with torch.no_grad():
                self.phi = self.beta * self.phi + (
                    1 - self.beta
                ) * self.backbone.forward_features(x[: self.batch_size]).mean(0)

            # Measure accuracy and record metrics
            prec1, prec5 = accuracy(output, aug_y, topk=(1, 5))
            meters["loss"].update(firstloss.sum() / aug_y.size(0))
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
