import math
import torch.nn as nn
import timm

__all__ = ["mobilenetv3"]


def mobilenetv3(config):
    world_size = config.pop("world_size", 1)
    lr = config.pop("lr")  # 0.000375
    batches_per_allreduce = config.pop("batches_per_allreduce")
    warmup_epochs = config.pop("warmup_epochs")
    num_epochs = config.pop("num_epochs")
    num_steps_per_epoch = config.pop("num_steps_per_epoch")

    scaling_factor = batches_per_allreduce * world_size

    # passing num_classes
    model = timm.create_model(
        "mobilenetv3_small_050",
        pretrained=False,
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
            "optimizer": "RMSprop",
            "alpha": 0.9,
            "momentum": 0.9,
            "eps": 0.001,
            "weight_decay": 0.00001,
            "step_lambda": config_by_step,
        },
        {"epoch": warmup_epochs, "lr": lr * scaling_factor},
        {"epoch": 8, "lr": (lr - 1 * 6.25e-5) * scaling_factor},
        {"epoch": 12, "lr": (lr - 2 * 6.25e-5) * scaling_factor},
        {"epoch": 16, "lr": (lr - 3 * 6.25e-5) * scaling_factor},
        {"epoch": 22, "lr": (lr - 4 * 6.25e-5) * scaling_factor},
        {"epoch": 26, "lr": (lr - 5 * 6.25e-5) * scaling_factor},
    ]

    return model
