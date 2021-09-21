import torch
import horovod.torch as hvd

from regime import Regime
from regularizer import Regularizer

class OptimizerRegime(Regime, torch.optim.Optimizer):
    def __init__(self, model, compression, reduction, batches_per_allreduce, gradient_predivide_factor, regime, defaults={}):
        super(OptimizerRegime, self).__init__(regime, defaults)
        self.parameters = list(model.parameters())
        self.regularizer = Regularizer(model)

        optimizer = torch.optim.SGD(self.parameters, lr=0)

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # Horovod: wrap optimizer with DistributedOptimizer.
        self.optimizer = hvd.DistributedOptimizer(optimizer,
                                    named_parameters=model.named_parameters(),
                                    compression=compression,
                                    op=reduction,
                                    backward_passes_per_step=batches_per_allreduce,
                                    gradient_predivide_factor=gradient_predivide_factor)


    def update(self, epoch=None, training_steps=None):
        """Adjust optimizer according to current epoch or steps and training regime.
        """
        if super(OptimizerRegime, self).update(epoch, training_steps):
            self.adjust_from_config(self.config)


    def adjust_from_config(self, config):
        for param_group in self.optimizer.param_groups:
            for key in param_group.keys():
                if key in config:
                    new_val = config[key]
                    if new_val != param_group[key]:
                        print(f"Updating {key}: {new_val}")
                        param_group[key] = config[key]


    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        self.optimizer.zero_grad()


    def step(self, *args, **kwargs):
        """Performs a single optimization step (parameter update)."""
        self.regularizer.pre_step()
        self.optimizer.step(*args, **kwargs)
        self.regularizer.post_step()


    def __getstate__(self):
        return {
            'optimizer_state': self.optimizer.__getstate__(),
            'regime': self.regime,
        }


    def __setstate__(self, state):
        self.regime = state.get('regime')
        self.optimizer.__setstate__(state.get('optimizer_state'))


    def state_dict(self):
        """Returns the state of the optimizer as a :class:`dict`.
        """
        return self.optimizer.state_dict()


    def load_state_dict(self, state_dict):
        """Loads the optimizer state.
        Arguments:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        self.optimizer.load_state_dict(state_dict)