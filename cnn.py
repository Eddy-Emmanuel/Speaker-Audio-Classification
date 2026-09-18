"""Simple 2D-CNN classifier operating on mel-spectrogram "images"."""

import torch.nn as nn


class ModelCNN(nn.Module):
    def __init__(self, name: str = "cnn_model"):
        super().__init__()
        self._name = name

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(32)

        self.mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )

        self.flatten = nn.Flatten()

        self.out = nn.Sequential(
            nn.Dropout(p=0),
            nn.Linear(65536, 2),
        )

    @property
    def name(self):
        return self._name

    def forward(self, x):
        x = self.encoder(x)
        x = self.avg_pool(x)
        x = self.mlp(x)
        x = self.flatten(x)
        x = self.out(x)
        return x
