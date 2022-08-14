import horovod.torch as hvd
import logging
import torch

from copy import deepcopy

_OPTIMIZERS = {name: func for name, func in torch.optim.__dict__.items()}

class OptimizerRegime(object):
    def __init__(
        self,
        model,
        lr,
        compression,
        reduction,
        batches_per_allreduce,
        gradient_predivide_factor,
        regime,
        use_amp,
        defaults={},
    ):
        self.lr = lr
        self.compression = compression
        self.reduction = reduction
        self.batches_per_allreduce = batches_per_allreduce
        self.gradient_predivide_factor = gradient_predivide_factor
        self.regime = regime
        self.use_amp = use_amp
        self.defaults = defaults
        self.config = defaults
        self.current_regime_phase = None

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # Horovod: wrap optimizer with DistributedOptimizer.
        self.optimizer = hvd.DistributedOptimizer(
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

    def update(self, epoch, steps):
        """Adjusts config according to current epoch or steps and regime."""
        if self.regime is None:
            return False

        epoch = -1 if epoch is None else epoch
        steps = -1 if steps is None else steps
        config = deepcopy(self.config)

        if self.current_regime_phase is None:
            # Find the first entry where the epoch is smallest than current
            for regime_phase, regime_config in enumerate(self.regime):
                start_epoch = regime_config.get("epoch", 0)
                start_step = regime_config.get("step", 0)
                if epoch >= start_epoch or steps >= start_step:
                    self.current_regime_phase = regime_phase
                    break
                config.update(regime_config)

        if len(self.regime) > self.current_regime_phase + 1:
            next_phase = self.current_regime_phase + 1
            # Any more regime steps?
            start_epoch = self.regime[next_phase].get("epoch", float("inf"))
            start_step = self.regime[next_phase].get("step", float("inf"))
            if epoch >= start_epoch or steps >= start_step:
                self.current_regime_phase = next_phase

        config.update(self.regime[self.current_regime_phase])

        if "lr_decay" in config:
            decay_steps = config.pop("lr_decay_steps", 100)
            if steps % decay_steps == 0:
                decay_rate = config.pop("lr_decay")
                config["lr_adj"] = decay_rate ** (steps / decay_steps)
        elif "lr_rampup" in config and config["lr_rampup"]:
            # Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
            # accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
            # the first warmup_epochs epochs.
            # See https://arxiv.org/abs/1706.02677 for details.
            warmup_epochs = config.pop("warmup_epochs", 5)
            lr_epoch = float(steps + 1) / self.num_steps
            config["lr_adj"] = (
                1.0 / hvd.size() * (lr_epoch * (hvd.size() - 1) / warmup_epochs + 1)
            )

        if config != self.config:
            self.config = config
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
        if "lr_adj" in config:
            logging.debug(
                f"OPTIMIZER REGIME - lr adjusted - lr_adj = {config['lr_adj']}"
            )

        for param_group in self.optimizer.param_groups:
            if "lr_adj" in config:
                lr = self.lr * config['lr_adj']
                param_group['lr'] = lr
                logging.debug(f"OPTIMIZER REGIME - updating lr = {lr}")
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
