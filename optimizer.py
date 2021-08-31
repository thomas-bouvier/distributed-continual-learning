import torch
import horovod.torch as hvd

from regime import Regime

class OptimizerRegime(Regime, torch.optim.Optimizer):
    def __init__(self, model, regime, fp16_allreduce, use_adasum, gradient_predivide_factor, defaults={}):
        super(OptimizerRegime, self).__init__(regime, defaults)
        self.parameters = list(model.parameters())

        optimizer = torch.optim.SGD(self.parameters, lr=0)

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # Horovod: (optional) compression algorithm.
        compression = hvd.Compression.fp16 if fp16_allreduce else hvd.Compression.none

        # Horovod: wrap optimizer with DistributedOptimizer.
        self.optimizer = hvd.DistributedOptimizer(optimizer,
                                            named_parameters=model.named_parameters(),
                                            compression=compression,
                                            op=hvd.Adasum if use_adasum else hvd.Average,
                                            gradient_predivide_factor=gradient_predivide_factor)