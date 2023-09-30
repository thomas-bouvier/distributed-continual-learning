import horovod.torch as hvd
import math
import torch.nn as nn
import timm

__all__ = ["resnet18"]


def resnet18(config):
    lr = config.pop("lr")
    batches_per_allreduce = config.pop("batches_per_allreduce")
    warmup_epochs = config.pop("warmup_epochs")
    num_epoch = config.pop("num_epochs")
    num_steps_per_epoch = config.pop("num_steps_per_epoch")

    scaling_factor = min(batches_per_allreduce * hvd.size(), 64)

    # Torchvision defaults zero_init_residual to False. This option set to False
    # will perform better on short epoch runs, however it is not the case on a
    # longer training run and actually will outperform the non-zero out version.
    model = timm.create_model(
        "resnet18", num_classes=config.get("num_classes"), zero_init_last=False
    )

    def rampup_lr(step):
        # Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
        # the first warmup_epochs epochs.
        # See https://arxiv.org/abs/1706.02677 for details.
        lr_epoch = step["epoch"] + step["batch"] / num_steps_per_epoch
        return lr * (lr_epoch * (scaling_factor - 1) / warmup_epochs + 1)

    def config_by_step(step):
        warmup_steps = warmup_epochs * num_steps_per_epoch

        if step["epoch"] * num_steps_per_epoch + step["batch"] < warmup_steps:
            return {"lr": rampup_lr(step)}
        return {}

    if num_epoch < 25:
        model.regime = [
            {
                "epoch": 0,
                "optimizer": "SGD",
                "momentum": 0.9,
                "weight_decay": 0.0001,
                "lr": lr,
            }
        ]
    else:
        model.regime = [
            {
                "epoch": 0,
                "optimizer": "SGD",
                "momentum": 0.9,
                "weight_decay": 0.00005,
                "step_lambda": config_by_step,
            },
            {"epoch": warmup_epochs, "lr": lr * scaling_factor},
            {"epoch": 21, "lr": lr * scaling_factor * 5e-1},
            {"epoch": 26, "lr": lr * scaling_factor * 5e-2},
            {"epoch": 28, "lr": lr * scaling_factor * 1e-2},
        ]

    return model
