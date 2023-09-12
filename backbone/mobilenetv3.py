import horovod.torch as hvd
import math
import torch.nn as nn
import timm

__all__ = ["mobilenetv3"]


def mobilenetv3(config):
    lr = config.pop("lr")
    warmup_epochs = config.pop("warmup_epochs")
    num_epochs = config.pop("num_epochs")
    num_steps_per_epoch = config.pop("num_steps_per_epoch")

    # passing num_classes
    model = timm.create_model(
        "mobilenetv3_small_100",
        pretrained=False,
        num_classes=config.get("num_classes"),
    )

    def rampup_lr(step, lr, num_steps_per_epoch, warmup_epochs):
        lr_epoch = step["epoch"] + step["batch"] / num_steps_per_epoch
        return lr * lr_epoch / warmup_epochs

    def config_by_step(step):
        warmup_steps = warmup_epochs * num_steps_per_epoch

        if step["epoch"] * num_steps_per_epoch + step["batch"] < warmup_steps:
            return {"lr": rampup_lr(step, lr, num_steps_per_epoch, warmup_epochs)}
        return {}

    model.regime = [
        {
            "epoch": 0,
            "optimizer": "RMSprop",
            "alpha": 0.9,
            "momentum": 0.9,
            "eps": 0.001,
            "weight_decay": 0.00001,
            "step_lambda": config_by_step,
        },
        {"epoch": warmup_epochs, "lr": lr},
        {"epoch": 8, "lr": lr - 1 * 1e-3},
        {"epoch": 12, "lr": lr - 2 * 1e-3},
        {"epoch": 16, "lr": lr - 3 * 1e-3},
        {"epoch": 22, "lr": lr - 4 * 1e-3},
        {"epoch": 26, "lr": lr - 5 * 1e-3},
    ]

    return model