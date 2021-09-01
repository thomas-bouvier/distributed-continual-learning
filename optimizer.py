import torch
import horovod.torch as hvd

from regime import Regime
from regularizer import Regularizer

class OptimizerRegime(Regime, torch.optim.Optimizer):
    def __init__(self, model, regime, lr, lr_scaler, momentum, fp16_allreduce, use_adasum, gradient_predivide_factor, defaults={}):
        super(OptimizerRegime, self).__init__(regime, defaults)
        self.parameters = list(model.parameters())
        self.optimizer = self._create_optimizer(model, lr, lr_scaler, momentum, fp16_allreduce, use_adasum, gradient_predivide_factor)
        self.regularizer = Regularizer(model)


    def _create_optimizer(self, model, lr, lr_scaler, momentum, fp16_allreduce, use_adasum, gradient_predivide_factor):
        optimizer = torch.optim.SGD(self.parameters, lr=lr*lr_scaler, momentum=momentum)

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if fp16_allreduce else hvd.Compression.none

        # Horovod: wrap optimizer with DistributedOptimizer.
        return hvd.DistributedOptimizer(optimizer,
                                            named_parameters=model.named_parameters(),
                                            compression=compression,
                                            op=hvd.Adasum if use_adasum else hvd.Average,
                                            gradient_predivide_factor=gradient_predivide_factor)


    def update(self, epoch=None, training_steps=None):
        """adjusts optimizer according to current epoch or steps and training regime.
        """
        updated = False
        if super(OptimizerRegime, self).update(epoch, training_steps):
            self.adjust(self.config)
            updated = True


    def zero_grad(self):
        """Clears the gradients of all optimized :class:`Variable` s."""
        self.optimizer.zero_grad()


    def step(self, *args, **kwargs):
        """Performs a single optimization step (parameter update)."""
        self.regularizer.pre_step()
        self.optimizer.step(*args, **kwargs)
        self.regularizer.post_step()