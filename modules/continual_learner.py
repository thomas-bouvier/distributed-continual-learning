import abc
import copy
import numpy as np
import torch

from torch import nn
from torch.cuda.amp import GradScaler
from torch.distributions import Categorical
from torch.nn import functional as F

from utils.log import PerformanceResultsLog


class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module to add continual learning capabilities to a classifier (e.g., param regularization, replay).'''

    def __init__(
        self,
        backbone_model,
        use_amp,
        optimizer_regime,
        batch_size,
        batch_metrics,
        state_dict,
    ):
        super().__init__()
        self.backbone_model = backbone_model
        self.use_amp = use_amp
        self.optimizer_regime = optimizer_regime
        self.batch_size = batch_size
        self.batch_metrics = batch_metrics

        self.minimal_eval_loss = float("inf")
        self.best_model = None

        self.scaler = None
        if self.use_amp:
            self.scaler = GradScaler()

        if state_dict is not None:
            self.backbone_model.load_state_dict(state_dict)
            self.initial_snapshot = copy.deepcopy(state_dict)
        else:
            self.initial_snapshot = copy.deepcopy(self.backbone_model.state_dict())

        self.perf_metrics = PerformanceResultsLog()


    def _device(self):
        return next(self.backbone_model.parameters()).device
