import horovod.torch as hvd
import math
import torch.nn as nn
import timm

__all__ = ["resnet18"]


def resnet18(config):
    lr = config.pop("lr")
    warmup_epochs = config.pop("warmup_epochs")
    num_steps_per_epoch = config.pop("num_steps_per_epoch")

    # passing num_classes
    model = timm.create_model("resnet18", num_classes=config.get("num_classes"))

    def rampup_lr(step, lr, num_steps_per_epoch, warmup_epochs):
        # Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
        # the first warmup_epochs epochs.
        # See https://arxiv.org/abs/1706.02677 for details.
        lr_epoch = step["epoch"] + step["batch"] / num_steps_per_epoch
        return lr * (lr_epoch * (hvd.size() - 1) / warmup_epochs + 1)

    def config_by_step(step):
        warmup_steps = warmup_epochs * num_steps_per_epoch

        if step["epoch"] * num_steps_per_epoch + step["batch"] < warmup_steps:
            return {"lr": rampup_lr(step, lr, num_steps_per_epoch, warmup_epochs)}
        return {}

    model.regime = [
        {
            "epoch": 0,
            "optimizer": "SGD",
            "momentum": 0.9,
            "weight_decay": 0.00005,
            "step_lambda": config_by_step,
        },
        {"epoch": warmup_epochs, "lr": lr * hvd.size()},
        {"epoch": 18, "lr": lr * hvd.size() * 1e-1},
        {"epoch": 23, "lr": lr * hvd.size() * 1e-2},
        {"epoch": 30, "lr": lr * hvd.size() * 1e-3},
    ]

    return model
