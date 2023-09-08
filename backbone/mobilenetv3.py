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

    def cosine_anneal_lr(step, lr, T_max, eta_min=1e-6):
        """
        Args:
            eta_min (float): lower lr bound for cyclic schedulers that hit 0 (1e-6)
        """
        lr_epoch = step["epoch"] + step["batch"] / num_steps_per_epoch
        return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * lr_epoch / T_max)) / 2

    def config_by_step(step):
        warmup_steps = warmup_epochs * num_steps_per_epoch

        if step["epoch"] * num_steps_per_epoch + step["batch"] < warmup_steps:
            return {"lr": rampup_lr(step, lr, num_steps_per_epoch, warmup_epochs)}
        else:
            return {"lr": cosine_anneal_lr(step, lr, num_epochs, eta_min=lr_min)}

    model.regime = [
        {
            "epoch": 0,
            "optimizer": "RMSprop",
            "alpha": 0.9,
            "momentum": 0.9,
            "eps": 0.001,
            "weight_decay": 0.000015,
            "step_lambda": config_by_step,
        },
        {
            "epoch": warmup_epochs - 1,
            "step_lambda": config_by_step,
        },
    ]

    return model