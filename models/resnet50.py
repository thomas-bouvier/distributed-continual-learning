import math
import torch.nn as nn
import logging

from torchvision.models import resnet50 as rn50

__all__ = ["resnet50"]


def resnet50(config):
    # passing num_classes
    model = rn50(**config)

    model.regime = [
        {
            "epoch": 0,
            "optimizer": "SGD",
            "lr_rampup": True,
            "momentum": 0.9,
            "weight_decay": 0.00005,
        },
        {"epoch": 5, "lr_adj": 1.0, "lr_rampup": False},
        {"epoch": 30, "lr_adj": 1e-1},
        {"epoch": 60, "lr_adj": 1e-2},
        {"epoch": 80, "lr_adj": 1e-3},
    ]
    return model
