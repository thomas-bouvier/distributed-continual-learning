import abc
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from utils import get_data_loader
from models import fc
from models.utils.ncl import additive_nearest_kf


class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module to add continual learning capabilities to a classifier (e.g., param regularization, replay).'''

    def __init__(self):
        super().__init__()

        self.optimizer_regime = None
        self.data_regime = None