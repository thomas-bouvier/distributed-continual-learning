import torch
import torch.nn as nn
import torch.nn.functional as F

class icarl_mnistnet_model(nn.Module):
    def __init__(self, num_features=50):
        super(icarl_mnistnet_model, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, num_features)
        self.fc2 = nn.Linear(num_features, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        fv = x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return fv, x


def icarl_mnistnet(config):
    return icarl_mnistnet_model()
