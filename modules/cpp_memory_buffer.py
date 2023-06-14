import abc
import torch
from torch import nn
from torch.nn import functional as F
from utils import get_data_loader
import copy
import numpy as np
from models.cl.fromp_optimizer import softmax_hessian


class CppMemoryBuffer(nn.Module, metaclass=abc.ABCMeta):
    """Abstract module for a classifier that enables it to maintain a memory buffer."""

    def __init__(self):
        super().__init__()

        # Settings
        self.use_memory_buffer = False
        self.budget_per_class = 100
        self.sample_selection = 'random'
        self.norm_exemplars = True

        engine = neomem.EngineLoader(self.provider,
            ctypes.c_uint16(hvd.rank()).value, self.cuda_rdma
        )
        self.dsl = neomem.DistributedStreamLoader(
            engine,
            neomem.Classification,
            train_data_regime.total_num_classes, self.rehearsal_size, self.num_representatives, self.num_candidates,
            ctypes.c_int64(torch.random.initial_seed() + hvd.rank()).value,
            1, list(shape), neomem.CPUBuffer, self.discover_endpoints, self.log_level not in ('info')
        )
        #self.dsl.enable_augmentation(True)
        self.dsl.start()


    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
