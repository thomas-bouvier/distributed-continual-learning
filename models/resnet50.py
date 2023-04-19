import horovod.torch as hvd
import math
import torch.nn as nn
import logging

from torchvision.models import resnet50 as rn50

__all__ = ["resnet50"]


def resnet50(config, steps_per_epoch):
    lr = config.pop("lr")

    # passing num_classes
    model = rn50(**config)

    def rampup_lr(step, steps_per_epoch):
        # Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
        # the first warmup_epochs epochs.
        # See https://arxiv.org/abs/1706.02677 for details.
        warmup_epochs = config.pop("warmup_epochs", 5)
        lr_epoch = step / steps_per_epoch
        return 1.0 / hvd.size() * (lr_epoch * (hvd.size() - 1) / warmup_epochs + 1)

    def config_by_step(step):
        return {
            'lr': lr * rampup_lr(step, steps_per_epoch=steps_per_epoch)
        }

    model.regime = [
        {
            "epoch": 0,
            "optimizer": "SGD",
            "momentum": 0.9,
            "weight_decay": 0.00005,
            "step_lambda": config_by_step,
        },
        {"epoch": 5, "lr": lr},
        {"epoch": 30, "lr": lr * 1e-1},
        {"epoch": 60, "lr": lr * 1e-2},
        {"epoch": 80, "lr": lr * 1e-3},
    ]
    return model
