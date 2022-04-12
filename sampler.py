import math
import torch
import torch.distributed as dist

from torch.utils.data import Sampler


"""
Re-implementation of torch.utils.data.distributed.DistributedSampler but
where data is only sampled from a same subset of the dataset at each epoch.
"""


class MyDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        self.local_indices = torch.tensor(
            indices[self.rank: self.total_size: self.num_replicas]
        )
        assert len(self.local_indices) == self.num_samples

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(
                len(self.local_indices), generator=g).tolist()
        else:
            indices = list(range(len(self.local_indices)))

        assert len(indices) == self.num_samples

        # shuffle local indices
        res = self.local_indices[indices].tolist()
        assert len(res) == self.num_samples

        return iter(res)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
