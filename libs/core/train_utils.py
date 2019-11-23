import numpy as np

import torch
from torch.utils.data.sampler import Sampler


class DistributedSampler4Iter(Sampler):
    def __init__(self, dataset, total_iter, batch_size, world_size, rank,
                 iter_now=0, rand_seed=666):
        super(DistributedSampler4Iter, self).__init__()
        assert rank < world_size

        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.iter_now = iter_now
        self.rand_seed = rand_seed

        self.indices = self.gen_indices()
        self.called = False

    def __iter__(self):
        if not self.called:
            self.called = True
            return iter(self.indices[self.iter_now*self.batch_size:])
        else:
            raise RuntimeError('This sampler is not designed to be called
                                more than once!')

    def gen_indices(self):
        np.random.seed(self.rand_seed)

        own_size = self.total_iter * self.batch_size
        all_size = own_size * world_size

        indices = np.arange(len(self.dataset))
        num_repeat = (all_size - 1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)
        beg = self.own_size * self.rank
        end = beg + self.own_size
        indices = indices[beg: end]

        return indices

