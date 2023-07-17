import horovod.torch as hvd
import logging
import torch
import torch.nn as nn
import numpy as np

from torch.cuda.amp import autocast

from modules import ContinualLearner, Buffer
from train.train import measure_performance
from utils.meters import get_timer, accuracy
from utils.log import PerformanceResultsLog


__all__ = ["Agem"]


class Agem(ContinualLearner):

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
        super(Agem, self).__init__(
            backbone,
            optimizer_regime,
            use_amp,
            batch_size,
            config,
            buffer_config,
            batch_metrics,
        )

        print('[+] DEBUG AGEM : INIT')

        self.use_memory_buffer = True

        # Implémentation via mammoth à voir pour grad_xy et grad_er
        self.grad_dims = []
        for param in self.backbone.parameters():
            self.grad_dims.append(param.data.numel())
        print("[+] DEBUG : ",str(np.sum(self.grad_dims)))
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)) # TODO add device
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)) # TODO add device
        print("[+] DEBUG : ",str((self.grad_xy)))

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

    def store_grad(self, params, grads, grad_dims):
        grads.fill_(0.0)
        count = 0
        for param in params():
            if param.grad is not None:
                begin = 0 if count == 0 else sum(grad_dims[:count])
                end = np.sum(grad_dims[:count + 1])
                grads[begin: end].copy_(param.grad.data.view(-1))
            count += 1

    def overwrite_grad(self, params, newgrad, grad_dims):
        count = 0
        for param in params():
            if param.grad is not None:
                begin = 0 if count == 0 else sum(grad_dims[:count])
                end = sum(grad_dims[:count + 1])
                this_grad = newgrad[begin: end].contiguous().view(
                    param.grad.data.size())
                param.grad.data.copy_(this_grad)
            count += 1

    def project(self, gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
        corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
        return gxy - corr * ger

    def train_one_step(self, x, y, meters, step):

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
                outputs = self.backbone(x)
                loss = self.criterion(outputs,y)

            # Loss Backwards
            self.scaler.scale(loss.sum() / loss.size(0)).backward()

            self.buffer.update(x,y,step)

            if step["task_id"] != 0:

                self.store_grad(self.backbone.parameters,self.grad_xy,self.grad_dims)  # A voir pour le self.backbone.parameters()

                buf = self.buffer._Buffer__get_current_augmented_minibatch(step)
                buf_x,buf_y, = buf.x,buf.y # step ou step-1 ?
                self.optimizer_regime.zero_grad() # self.net.zero_grad()

                buf_outputs = self.backbone(buf_x)
                
                buf_loss = self.criterion(buf_outputs,buf_y)
                self.scaler.scale(buf_loss.sum() / buf_loss.size(0)).backward()

                self.store_grad(self.backbone.parameters,self.grad_er,self.grad_dims) # A voir pour le self.backbone.parameters()

                dot_prod = torch.dot(self.grad_xy,self.grad_er)
                
                if dot_prod.item() < 0:
                    g_tilde = self.project(gxy=self.grad_xy, ger=self.grad_er)
                    self.overwrite_grad(self.backbone.parameters, g_tilde, self.grad_dims)
                else:
                    self.overwrite_grad(self.backbone.parameters, self.grad_xy, self.grad_dims)

            self.optimizer_regime.optimizer.synchronize()
            with self.optimizer_regime.optimizer.skip_synchronize():
                self.scaler.step(self.optimizer_regime.optimizer)
                self.scaler.update()

            # Measure accuracy and record metrics
            prec1, prec5 = accuracy(outputs, y, topk=(1, 5))
            meters["loss"].update(loss.mean())
            meters["prec1"].update(prec1, x.size(0))
            meters["prec5"].update(prec5, x.size(0))
            meters["num_samples"].update(x.size(0))
            meters["local_rehearsal_size"].update(self.buffer.get_size())