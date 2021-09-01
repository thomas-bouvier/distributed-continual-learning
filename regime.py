class Regime(object):
    def __init__(self, regime, defaults={}):
        self.regime = regime
        self.config = defaults


    def update(self, epoch, training_steps):
        pass