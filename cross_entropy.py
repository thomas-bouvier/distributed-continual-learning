import torch
import torch.nn as nn

"""
def cross_entropy(weight=None, reduction='none'):
    self.mask = torch.ones(train_data_regime.total_num_classes, device=self._device()).float()
    self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')

    # use_mask
    self.mask = torch.tensor(train_data_regime.previous_classes_mask, device=self._device()).float()
    self.criterion = nn.CrossEntropyLoss(weight=self.mask, reduction='none')
"""

class CrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, reduction='none'):
        super(CrossEntropyLoss, self).__init__(reduction=reduction)
