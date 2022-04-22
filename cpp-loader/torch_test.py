import torch
import rehearsal
import random
import ctypes
import numpy as np

from torch.utils.data import Dataset, DataLoader



class MyDataset(Dataset):
    def __init__(self, values, labels):
        super(MyDataset, self).__init__()
        self.values = values
        self.labels = labels

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], self.labels[index]


# num_candidates
K = 5
# num_samples
R = 2


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    sl = rehearsal.StreamLoader(
        K, 128 // 2, ctypes.c_int64(torch.random.initial_seed()).value)

    # https://github.com/pytorch/pytorch/issues/5059
    values = np.random.rand(5000, 3)
    labels = np.random.rand(5000)
    dataset = MyDataset(values, labels)
    loader = DataLoader(dataset=dataset, batch_size=128,
                        shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(10):
        for inputs, target in loader:
            N = len(inputs)
            aug_samples = torch.zeros(N + R, 3)
            aug_labels = torch.randint(high=K, size=(N + R,))
            aug_weights = torch.zeros(N + R)
            sl.accumulate(inputs, target, aug_samples, aug_labels, aug_weights)
            size = sl.wait()
            print(size, inputs, aug_samples)
