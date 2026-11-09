import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        channels = input_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(channels, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, *input_shape)).view(1, -1).shape[1]
        self.classifier = nn.Sequential(nn.Linear(flat, 120), nn.ReLU(), nn.Linear(120, 10))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))
