import horovod.torch as hvd
import math
import torch.nn as nn
import logging

from torchvision.models import resnet50 as rn50

__all__ = ["resnet50"]


def resnet50(config, steps_per_epoch):
    lr = config.pop("lr") * hvd.size()
    warmup_epochs = config.pop("warmup_epochs", 0)
    config.pop("num_epochs", None)

    # passing num_classes
    model = rn50(**config)

    def rampup_lr(lr, step, steps_per_epoch, warmup_epochs):
        # Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
        # the first warmup_epochs epochs.
        # See https://arxiv.org/abs/1706.02677 for details.
        lr_epoch = step / steps_per_epoch
        return lr * 1.0 / hvd.size() * (lr_epoch * (hvd.size() - 1) / warmup_epochs + 1)

    def config_by_step(step):
        warmup_steps = warmup_epochs * steps_per_epoch

        if step < warmup_steps:
            return {'lr': rampup_lr(lr, step, steps_per_epoch, warmup_epochs)}
        return {}

    model.regime = [
        {
            "epoch": 0,
            "optimizer": "SGD",
            "momentum": 0.9,
            "weight_decay": 0.00005,
            "step_lambda": config_by_step,
        },
        {"epoch": 5, "lr": lr},
        {"epoch": 18, "lr": lr * 1e-1},
        {"epoch": 23, "lr": lr * 1e-2},
        {"epoch": 30, "lr": lr * 1e-3},
    ]

    return model
