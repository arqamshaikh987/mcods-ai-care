import torch
import torch.nn as nn


def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        # Pool 4×4 → 1×1 at classifier time
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)   # 256→64
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        self.conv3 = ConvBlock(128, 256, pool=True)  # 64→16
        self.conv4 = ConvBlock(256, 512, pool=True)  # 16→4
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        # **This** block will collapse 4×4 → 1×1
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),   # 4→1
            nn.Flatten(),      # 512×1×1 → 512
            nn.Linear(512, num_diseases)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)
