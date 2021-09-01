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

    def __init__(self, regime: list, defaults = {}):
        self.regime = regime
        self.defaults = defaults
        self.config = defaults
        self.current_regime_phase = None


    def update(self, epoch, training_steps):
        """Adjusts config according to current epoch or steps and regime.
        """
        if self.regime is None:
            return False

        epoch = -1 if epoch is None else epoch
        training_steps = -1 if training_steps is None else training_steps
        config = deepcopy(self.config)

        if self.current_regime_phase is None:
            # Find the first entry where the epoch is smallest than current
            for regime_phase, regime_config in enumerate(self.regime):
                start_epoch = regime_config.get('epoch', 0)
                start_step = regime_config.get('step', 0)
                if epoch >= start_epoch or training_steps >= start_step:
                    self.current_regime_phase = regime_phase
                    break
                config.update(regime_config)

        if len(self.regime) > self.current_regime_phase + 1:
            next_phase = self.current_regime_phase + 1
            # Any more regime steps?
            start_epoch = self.regime[next_phase].get('epoch', float('inf'))
            start_step = self.regime[next_phase].get('step', float('inf'))
            if epoch >= start_epoch or training_steps >= start_step:
                self.current_regime_phase = next_phase

        config.update(self.regime[self.current_regime_phase])

        if config == self.config:
            return False
        else:
            self.config = config
            return True