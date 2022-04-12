import horovod.torch as hvd

from copy import deepcopy


class Regime(object):
    """
    Examples for regime:
    1)  "[{'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
          {'epoch': 2, 'optimizer': 'Adam', 'lr': 5e-4},
          {'epoch': 4, 'optimizer': 'Adam', 'lr': 1e-4},
          {'epoch': 8, 'optimizer': 'Adam', 'lr': 5e-5}
         ]"
    """

    def __init__(self, regime: list, defaults={}):
        self.regime = regime
        self.defaults = defaults
        self.config = defaults
        self.current_regime_phase = None

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

        if "lr_decay" in config and "lr" in config:
            decay_steps = config.pop("lr_decay_steps", 100)
            if steps % decay_steps == 0:
                decay_rate = config.pop("lr_decay")
                config["lr"] *= decay_rate ** (steps / decay_steps)
        elif "lr_rampup" in config and "lr" in config and config["lr_rampup"]:
            # Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
            # accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
            # the first warmup_epochs epochs.
            # See https://arxiv.org/abs/1706.02677 for details.
            warmup_epochs = config.pop("warmup_epochs", 5)
            lr_epoch = epoch + float(steps + 1) / self.num_steps
            config["lr"] *= (
                1.0 / hvd.size() * (lr_epoch * (hvd.size() - 1) / warmup_epochs + 1)
            )
        elif "step_lambda" in config:
            config.update(eval_func(config.pop("step_lambda"), steps))
        elif "epoch_lambda" in config:
            config.update(eval_func(config.pop("epoch_lambda"), epoch))

        if config == self.config:
            return False
        else:
            self.config = config
            return True
