import torch
import torch.nn as nn
import torch.nn.functional as F

class mnist_model(nn.Module):
    def __init__(self):
        super(mnist_model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
            nn.Flatten(),

            nn.Linear(in_features=9216, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=128, out_features=10),

            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.features(x)


def mnistnet(**kwargs):
    return mnist_model()
