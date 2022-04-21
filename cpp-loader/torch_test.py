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


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    values = np.random.rand(5000, 3, 32, 32)
    labels = np.random.rand(5000)
    dataset = MyDataset(values, labels)
    loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    sl = rehearsal.StreamLoader(26, 10, ctypes.c_int64(
        torch.random.initial_seed()).value)

    for epoch in range(10):
        for inputs, target in loader:
            # inputs shape: torch.Size([128, 3, 32, 32])
            # target shape: torch.Size([128])
            pass
            #sl.accumulate((inputs, target))