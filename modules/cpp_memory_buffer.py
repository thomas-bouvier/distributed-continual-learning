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

        self.rehearsal_size = config.get("rehearsal_size", 16)
        self.num_candidates = config.get("num_candidates", 8)
        self.num_representatives = config.get("num_representatives", 8)
        self.provider = config.get('provider', 'na+sm')
        self.discover_endpoints = config.get('discover_endpoints', True)
        self.cuda_rdma = config.get('cuda_rdma', False)

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


    def before_all_tasks(self, train_data_regime):
        '''Former nil_cpp implementation'''

        x, y, _ = next(iter(train_data_regime.get_loader(0)))
        shape = x[0].size()

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

        self.minibatches_ahead = 2
        self.next_minibatches = []
        for i in range(self.minibatches_ahead):
            self.next_minibatches.append(AugmentedMinibatch(self.batch_size, self.num_representatives, shape, self.device))

        self.dsl.accumulate(x, y, self.next_minibatches[0].x,
                                  self.next_minibatches[0].y,
                                  self.next_minibatches[0].w)


    def before_all_tasks(self, train_data_regime):
        '''Former nil_cpp_cat implementation'''

        x, y, _ = next(iter(train_data_regime.get_loader()))
        shape = x[0].size()
        self.next_minibatch = AugmentedMinibatch(self.num_representatives, shape, self.device)

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
        self.dsl.use_these_allocated_variables(self.next_minibatch.x, self.next_minibatch.y, self.next_minibatch.w)
        self.dsl.start()

        self.dsl.accumulate(x, y)


    def _device(self):
        return next(self.parameters()).device


    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
