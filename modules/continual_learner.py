import copy
import logging
import torch

from torch import nn
from torch.cuda.amp import GradScaler
from torch.distributions import Categorical

from cross_entropy import CrossEntropyLoss
from utils.log import PerformanceResultsLog


class ContinualLearner:
    def __init__(
        self,
        backbone: nn.Module,
        optimizer_regime,
        use_amp,
        nsys_run,
        batch_size,
        config,
        buffer_config,
        batch_metrics=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.criterion = getattr(backbone, "criterion", CrossEntropyLoss())
        self.optimizer_regime = optimizer_regime
        self.use_amp = use_amp
        self.nsys_run = nsys_run
        self.batch_size = batch_size
        self.config = config
        self.buffer_config = buffer_config
        self.batch_metrics = batch_metrics

        self.initial_snapshot = copy.deepcopy(self.backbone.state_dict())
        self.minimal_eval_loss = float("inf")
        self.best_model = None
        self.scaler = GradScaler(enabled=use_amp)

    def before_all_tasks(self, train_data_regime):
        pass

    def before_every_task(self, task_id, train_data_regime):
        # Distribute the data
        train_data_regime.get_loader(task_id)

        """
        if self.best_model is not None:
            logging.debug(
                f"Loading best model with minimal eval loss ({self.minimal_eval_loss}).."
            )
            self.backbone.load_state_dict(self.best_model)
            self.minimal_eval_loss = float("inf")
        """

        if task_id > 0:
            if self.config.get("reset_state_dict", False):
                logging.debug("Resetting model internal state..")
                self.backbone.load_state_dict(copy.deepcopy(self.initial_snapshot))
            self.optimizer_regime.reset(self.backbone.parameters())

    def after_every_task(self, task_id, train_data_regime):
        pass

    def after_all_tasks(self):
        pass

    def _device(self):
        return next(self.backbone.parameters()).device

    def _is_on_cuda(self):
        return next(self.backbone.parameters()).is_cuda
