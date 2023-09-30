import horovod.torch as hvd
import math
import torch.nn as nn
import timm

__all__ = ["ghostnet"]


def ghostnet(config):
    lr = config.pop("lr")  # 0.01
    batches_per_allreduce = config.pop("batches_per_allreduce")
    warmup_epochs = config.pop("warmup_epochs")
    num_epochs = config.pop("num_epochs")
    num_steps_per_epoch = config.pop("num_steps_per_epoch")

    scaling_factor = min(batches_per_allreduce * hvd.size(), 64)

    # passing num_classes
    model = timm.create_model(
        "ghostnet_050",
        pretrained=False,
        drop_rate=0.225,
        num_classes=config.get("num_classes"),
    )

    def rampup_lr(step, lr, num_steps_per_epoch, warmup_epochs):
        lr_epoch = step["epoch"] + step["batch"] / num_steps_per_epoch
        return lr * (lr_epoch * (scaling_factor - 1) / warmup_epochs + 1)

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
            "weight_decay": 0.000015,
            "step_lambda": config_by_step,
        },
        {"epoch": warmup_epochs, "lr": lr * scaling_factor},
        {"epoch": 15, "lr": lr * scaling_factor * 5e-1},
        {"epoch": 21, "lr": lr * scaling_factor * 5e-2},
        {"epoch": 28, "lr": lr * scaling_factor * 1e-2},
    ]

    return model
