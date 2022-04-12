import torch
import math
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(
    inputs, target, weight, ignore_index, reduction="mean", from_logits=True
):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    if from_logits:
        return F.cross_entropy(
            inputs,
            target,
            weight,
            ignore_index=ignore_index,
            reduction=reduction,
        )
    else:
        return F.nll_loss(
            inputs,
            target,
            weight,
            ignore_index=ignore_index,
            reduction=reduction,
        )


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        weight=None,
        ignore_index=-100,
        reduction="mean",
        from_logits=True,
    ):
        super(CrossEntropyLoss, self).__init__(
            weight=weight, ignore_index=ignore_index, reduction=reduction
        )
        self.from_logits = from_logits

    def forward(self, input, target):
        return cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            from_logits=self.from_logits,
        )
