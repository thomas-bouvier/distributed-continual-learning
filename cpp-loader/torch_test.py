import torch
import rehearsal
import random
import ctypes

if __name__ == "__main__":
    l = [(random.choice('abcdefghijklmnopqrstuvwxyz'), torch.rand(4)) for i in range(0, 1000)]
    sl = rehearsal.StreamLoader(26, 10, ctypes.c_int64(torch.random.initial_seed()).value)
    sl.accumulate(l)
    print(sl.get_samples(10))
