import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["mnistnet"]


class mnistnet_model(nn.Module):
    def __init__(self, num_features=50):
        super(mnistnet_model, self).__init__()
        self.num_features = 50
        self.num_classes = 10
        self.feature_vector = None

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(64 * 5 * 5, num_features)
        self.fc2 = nn.Linear(num_features, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 64 * 5 * 5)
        x = self.fc1(x)
        self.feature_vector = x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def mnistnet(config):
    return mnistnet_model()
