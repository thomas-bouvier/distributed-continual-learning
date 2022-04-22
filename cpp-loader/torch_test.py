import torch
import rehearsal
import random
import ctypes

K = 5
N = 10
R = 2

def random_batch():
    samples = torch.rand(N, 3)
    labels = torch.randint(high=K, size=(N,))
    return (samples, labels)


if __name__ == "__main__":
    sl = rehearsal.StreamLoader(K, N // 2, ctypes.c_int64(torch.random.initial_seed()).value)

    aug_samples = torch.zeros(N + R, 3)
    aug_labels = torch.randint(high=K, size=(N + R,))
    aug_weights = torch.zeros(N + R)

    print("Round 1")
    samples, labels = random_batch()
    sl.accumulate(samples, labels, aug_samples, aug_labels, aug_weights)
    size = sl.wait()
    print(size, samples, aug_samples)

    print("Round 2")
    samples, labels = random_batch()
    sl.accumulate(samples, labels, aug_samples, aug_labels, aug_weights)
    size = sl.wait()
    print(size, samples, aug_samples)
