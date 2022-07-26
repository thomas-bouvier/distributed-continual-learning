import math
import torch.nn as nn

from torchvision.models import resnet18 as rn18

__all__ = ["resnet18"]


def resnet18(config):
    optimizer = config.pop('optimizer')
    lr = config.pop('lr')
    lr_scaler = config.pop('lr_scaler')
    momentum = config.pop('momentum')
    weight_decay = config.pop('weight_decay')

    # passing num_classes
    model = rn18(**config)
    lr = lr * lr_scaler
    model.regime = [
        {
            "epoch": 0,
            "optimizer": "SGD",
            "lr": lr,
            "lr_rampup": True,
            "momentum": 0.875,
            "weight_decay": 3.0517578125e-5,
        },
        {"epoch": 5, "lr": lr * 1.0, "lr_rampup": False},
        {"epoch": 30, "lr": lr * 1e-1, "weight_decay": 0},
        {"epoch": 45, "lr": lr * 1e-2},
        {"epoch": 80, "lr": lr * 1e-3},
    ]
    return model
