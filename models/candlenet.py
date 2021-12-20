import torch
import torch.nn as nn
import torch.nn.functional as F

class candlenet_model(nn.Module):
    def __init__(self):
        super(candlenet_model, self).__init__()
        self.feature_vector = None

        self.conv1 = nn.Conv1d(1, 128, kernel_size=20, stride=1, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=10, stride=1, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=10)
        self.fc1 = nn.Linear(773760, 200)
        self.drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(200, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        self.feature_vector = x
        x = F.relu(x)
        x = self.fc3(x)
        return x


def candlenet(config):
    return candle_model()
