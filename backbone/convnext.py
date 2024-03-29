import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

__all__ = ["convnext"]


def convnext(config):
    world_size = config.pop("world_size", 1)
    lr = config.pop("lr") * world_size  # 4e-3 * world_size
    lr_min = config.pop("lr_min") * world_size  # 1e-6 * world_size
    warmup_epochs = config.pop("warmup_epochs")
    num_epochs = config.pop("num_epochs")
    num_steps_per_epoch = config.pop("num_steps_per_epoch")

    model = timm.create_model(
        "convnext_base",
        pretrained=False,
        drop_rate=0.5,
        num_classes=config.get("num_classes"),
    )

    def rampup_lr(lr, step, num_steps_per_epoch, warmup_epochs):
        return lr * step / (num_steps_per_epoch * warmup_epochs)

    def cosine_anneal_lr(lr, step, T_max, eta_min=1e-6):
        """
        eta_min (float): lower lr bound for cyclic schedulers that hit 0 (1e-6)
        """
        return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * step / T_max)) / 2

    def config_by_step(step):
        warmup_steps = warmup_epochs * num_steps_per_epoch

        if step < warmup_steps:
            return {"lr": rampup_lr(lr, step, num_steps_per_epoch, warmup_epochs)}
        return {
            "lr": cosine_anneal_lr(
                lr,
                step,
                num_epochs * num_steps_per_epoch - warmup_steps,
                eta_min=lr_min,
            )
        }

    model.regime = [
        {
            "epoch": 0,
            "optimizer": "AdamW",
            "momentum": 0.9,
            "weight_decay": 0.05,
            "eps": 1e-8,
            "step_lambda": config_by_step,
        }
    ]

    return model
