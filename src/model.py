from __future__ import annotations

import torch
import torch.nn as nn


class EyeEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GazeRegressor(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.encoder_left = EyeEncoder(128)
        self.encoder_right = EyeEncoder(128)
        self.mlp = nn.Sequential(
            nn.Linear(128 + 128 + 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, left_eye: torch.Tensor, right_eye: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        lf = self.encoder_left(left_eye)
        rf = self.encoder_right(right_eye)
        x = torch.cat([lf, rf, pose], dim=1)
        return self.mlp(x)
