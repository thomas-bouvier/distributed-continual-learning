import horovod.torch as hvd
import logging
import torch

from regime import Regime

_OPTIMIZERS = {name: func for name, func in torch.optim.__dict__.items()}


class OptimizerRegime(Regime, torch.optim.Optimizer):
    def __init__(
        self,
        model,
        use_amp,
        compression,
        reduction,
        batches_per_allreduce,
        gradient_predivide_factor,
        regime,
        defaults={},
    ):
        super(OptimizerRegime, self).__init__(regime, defaults)
        self.parameters = list(model.parameters())
        self.compression = compression
        self.reduction = reduction
        self.batches_per_allreduce = batches_per_allreduce
        self.gradient_predivide_factor = gradient_predivide_factor

        optimizer = torch.optim.SGD(self.parameters, lr=0)

        # Horovod: wrap optimizer with DistributedOptimizer.
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=self.compression,
            op=self.reduction,
            backward_passes_per_step=self.batches_per_allreduce,
            gradient_predivide_factor=self.gradient_predivide_factor,
        )

        # Horovod: broadcast optimizer state and parameters
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        if use_amp:
            try:
                global amp
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this app.")
            self.model, self.optimizer = amp.initialize(
                model, optimizer, opt_level="O1")
        else:
            self.model = model
            self.optimizer = optimizer

    def update(self, epoch=None, steps=None):
        """Adjust optimizer according to current epoch or steps and training regime."""
        if super(OptimizerRegime, self).update(epoch, steps):
            logging.debug(
                f"OPTIMIZER REGIME - update (epoch: {epoch}, steps: {steps})"
            )
            self.adjust_from_config(self.config)

    def adjust_from_config(self, config):
        if "optimizer" in config:
            optim_method = _OPTIMIZERS[config.get("optimizer", "SGD")]
            if not isinstance(self.optimizer, optim_method):
                self.optimizer = optim_method(self.optimizer.param_groups)
                logging.debug(
                    f"OPTIMIZER REGIME - setting method = {config['optimizer']}"
                )
        for param_group in self.optimizer.param_groups:
            for key in param_group.keys():
                if key in config:
                    new_val = config[key]
                    if new_val != param_group[key]:
                        logging.debug(
                            f"OPTIMIZER REGIME - updating {key} = {new_val}")
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
            torch.optim.SGD(parameters, lr=0).state_dict())
        # Horovod: broadcast optimizer state.
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
        self.config = self.defaults
        self.current_regime_phase = None

    def get_value(self, key):
        return [group[key] for group in self.optimizer.param_groups]

    def get_lr(self):
        return self.get_value("lr")
