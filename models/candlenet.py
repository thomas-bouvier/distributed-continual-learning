import torch
import torch.nn as nn
import torch.nn.functional as F

class candle_model(nn.Module):
    def __init__(self):
        super(candle_model, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=20, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1),
            nn.Conv1d(128, 128, kernel_size=10, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10),
            nn.Flatten(),
            nn.Linear(773760, 200),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(200, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, x):
        return self.features(x)

    def should_distill(self):
        return False


def candlenet(**config):
    return candle_model()
