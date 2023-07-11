import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

"""
def cross_entropy(weight=None, reduction='none'):
    self.mask = torch.ones(train_data_regime.total_num_classes, device=self._device()).float()
    self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')

    # use_mask
    self.mask = torch.tensor(train_data_regime.previous_classes_mask, device=self._device()).float()
    self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')

    # meters["loss"].update(loss.sum() / self.mask[aug_y].sum())
"""


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, reduction="none"):
        super().__init__(reduction=reduction)


class ScaledMeanAbsoluteErrorLoss(_Loss):
    def __init__(self, reduction="none", scaling=1):
        super().__init__(reduction=reduction)
        self.scaling = scaling

    def forward(self, t1, t2):
        return torch.mean(torch.abs(t1 - t2), axis=(-1, -2)) / self.scaling
