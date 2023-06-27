import abc
import copy
import numpy as np
import torch

from torch import nn
from torch.cuda.amp import GradScaler
from torch.distributions import Categorical
from torch.nn import functional as F

from cross_entropy import CrossEntropyLoss
from utils.log import PerformanceResultsLog


class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module to add continual learning capabilities to a classifier (e.g., param regularization, replay).'''

    def __init__(
        self,
        backbone: nn.Module,
        optimizer_regime,
        use_amp,
        batch_size,
        buffer_config,
        batch_metrics=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.criterion = getattr(backbone, 'criterion', CrossEntropyLoss)()
        self.optimizer_regime = optimizer_regime
        self.use_amp = use_amp
        self.batch_size = batch_size
        self.buffer_config = buffer_config
        self.batch_metrics = batch_metrics

        self.initial_snapshot = copy.deepcopy(self.backbone.state_dict())
        self.minimal_eval_loss = float("inf")
        self.best_model = None
        self.scaler = GradScaler(enabled=use_amp)
        self.perf_metrics = PerformanceResultsLog()


    def before_every_task(self, task_id, train_data_regime):
        # Distribute the data
        train_data_regime.get_loader(task_id)

        if self.best_model is not None:
            logging.debug(
                f"Loading best model with minimal eval loss ({self.minimal_eval_loss}).."
            )
            self.backbone.load_state_dict(self.best_model)
            self.minimal_eval_loss = float("inf")

        if task_id > 0:
            if self.config.get("reset_state_dict", False):
                logging.debug("Resetting model internal state..")
                self.backbone.load_state_dict(
                    copy.deepcopy(self.initial_snapshot))
            self.optimizer_regime.reset(self.backbone.parameters())


    def _device(self):
        return next(self.backbone.parameters()).device


    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
