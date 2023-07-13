import horovod.torch as hvd
import logging
import torch

from regime import Regime
from copy import deepcopy

_OPTIMIZERS = {name: func for name, func in torch.optim.__dict__.items()}


class OptimizerRegime(Regime):
    def __init__(
        self,
        model,
        compression,
        reduction,
        gradient_predivide_factor,
        regime,
        use_amp,
        defaults={},
    ):
        super(OptimizerRegime, self).__init__(regime, defaults)
        self.parameters = list(model.parameters())
        self.named_parameters = list(model.named_parameters())
        self.compression = compression
        self.reduction = reduction
        self.gradient_predivide_factor = gradient_predivide_factor
        self.use_amp = use_amp

        self.optimizer = None

    def create_optimizer(self, config):
        optim_method = _OPTIMIZERS[config.get("optimizer", "SGD")]
        if not isinstance(self.optimizer, optim_method):
            self.create_distributed_optimizer(optim_method(self.parameters, lr=0))
            logging.debug(f"OPTIMIZER REGIME - setting method = {config['optimizer']}")

    def create_distributed_optimizer(self, optimizer):
        # Horovod: broadcast optimizer state and parameters
        hvd.broadcast_parameters(self.named_parameters, root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # Horovod: wrap optimizer with DistributedOptimizer.
        self.optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=self.named_parameters,
            compression=self.compression,
            op=self.reduction,
            gradient_predivide_factor=self.gradient_predivide_factor,
        )

    def update(self, step):
        """
        Adjusts config according to current epoch or steps and regime.

        Args:
            epoch (int): Current local epoch (within the current task).
            steps (int): Current local step number (within the current task).

        Returns:
            boolean: whether the regime has been updated.
        """
        if self.regime is None:
            return False

        updated = False
        if super(OptimizerRegime, self).update(step):
            logging.debug(
                f"OPTIMIZER REGIME - update (epoch: {step['epoch']}, batch: {step['batch']})"
            )
            self.adjust_from_config(self.config)
            updated = True

        return updated

    def adjust_from_config(self, config):
        if "optimizer" in config:
            self.create_optimizer(config)

        for param_group in self.optimizer.param_groups:
            for key in param_group.keys():
                if key in config:
                    new_val = config[key]
                    if new_val != param_group[key]:
                        logging.debug(f"OPTIMIZER REGIME - updating {key} = {new_val}")
                        param_group[key] = config[key]

    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        self.optimizer.zero_grad()

    def step(self, *args, **kwargs):
        """Performs a single optimization step (parameter update)."""
        self.optimizer.step(*args, **kwargs)

    def __getstate__(self):
        return {
            "optimizer_state": self.optimizer.__getstate__(),
            "regime": self.regime,
        }

    def __setstate__(self, state):
        self.regime = state.get("regime")
        self.optimizer.__setstate__(state.get("optimizer_state"))

    def state_dict(self):
        """Returns the state of the optimizer as a :class:`dict`."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Loads the optimizer state.
        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        self.optimizer.load_state_dict(state_dict)

    def reset(self, parameters):
        logging.debug("OPTIMIZER REGIME - resetting state..")
        self.optimizer.load_state_dict(
            _OPTIMIZERS[self.config.get("optimizer", "SGD")](
                parameters, lr=0
            ).state_dict()
        )
        # Horovod: broadcast optimizer state.
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
        self.config = self.defaults
        self.current_regime_phase = None

    def get_value(self, key):
        return [group[key] for group in self.optimizer.param_groups]

    def get_lr(self):
        return self.get_value("lr")
