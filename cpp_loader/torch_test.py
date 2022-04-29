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


# num_classes
K = 100
# num_representatives
N = 65
# num_candidates
C = 20
# num_samples
R = 20
# batch_size
B = 128

aug_samples = torch.zeros(B + R, 3, 32, 32)
aug_labels = torch.randint(high=K, size=(B + R,))
aug_weights = torch.zeros(B + R)


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    sl = rehearsal.StreamLoader(
        K, N, C, ctypes.c_int64(torch.random.initial_seed()).value)

    # https://github.com/pytorch/pytorch/issues/5059
    values = np.random.rand(5000, 3, 32, 32)
    labels = np.random.randint(0, K, 5000)
    dataset = MyDataset(values, labels)
    loader = DataLoader(dataset=dataset, batch_size=128,
                        shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(10):
        for inputs, target in loader:
            sl.accumulate(inputs, target, aug_samples, aug_labels, aug_weights)
            size = sl.wait()

            print(aug_samples.shape, aug_labels.shape, aug_weights.shape)
            print(aug_labels[-R:], aug_weights[-R:])
