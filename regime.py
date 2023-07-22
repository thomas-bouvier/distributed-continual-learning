import horovod.torch as hvd

from copy import deepcopy
from utils.utils import eval_func


class Regime:
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

    def update(self, step):
        """
        Adjusts config according to current epoch or steps and regime.

        Args:
            epoch (int): Current local epoch (within the current task).
            steps (int): Current local step number (within the current task).
        """
        if self.regime is None:
            return False

        config = deepcopy(self.config)
        if self.current_regime_phase is None:
            # Find the first entry where the epoch is smallest than current
            for regime_phase, regime_config in enumerate(self.regime):
                start_epoch = regime_config.get("epoch", 0)
                start_step = regime_config.get("step", 0)
                if step["epoch"] >= start_epoch or step["batch"] >= start_step:
                    self.current_regime_phase = regime_phase
                    break
                config.update(regime_config)

        if len(self.regime) > self.current_regime_phase + 1:
            next_phase = self.current_regime_phase + 1
            # Any more regime steps?
            start_epoch = self.regime[next_phase].get("epoch", float("inf"))
            start_step = self.regime[next_phase].get("step", float("inf"))
            if step["epoch"] >= start_epoch or step["batch"] >= start_step:
                self.current_regime_phase = next_phase

        config.update(self.regime[self.current_regime_phase])

        if "lr_decay" in config and "lr" in config:
            decay_steps = config.pop("lr_decay_steps", 100)
            if step["batch"] % decay_steps == 0:
                decay_rate = config.pop("lr_decay")
                config["lr"] *= decay_rate ** (step["batch"] / decay_steps)

        elif "step_lambda" in config:
            config.update(eval_func(config.pop("step_lambda"), step))

        if "execute" in config:
            config.pop("execute")()

        if config == self.config:
            return False
        self.config = config
        return True
