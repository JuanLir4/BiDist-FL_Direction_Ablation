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


class ConvNIDD(nn.Module):
    """Compact CNN for CIFAR-10 with GroupNorm for non-IID federated learning."""

    def __init__(self, in_channels=3, num_classes=10, gn_groups=8, dropout_p=0.1):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(gn_groups, out_ch), num_channels=out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout_p),
            )

        self.stem = nn.Sequential(
            conv_block(in_channels, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            conv_block(128, 256),
            conv_block(256, 256),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        return self.fc(torch.flatten(x, 1))


def build_model(dataset_name, input_shape):
    if dataset_name == "cifar10":
        return ConvNIDD(in_channels=input_shape[0])
    return Net(input_shape)
