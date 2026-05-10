from __future__ import annotations

import torch
import torch.nn as nn


class ItrackerEyeBackbone(nn.Module):
    # Mirrors GazeCapture iTracker eye pathway so pretrained weights can load.
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W] -> repeat channels to match pretrained RGB filters.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.features(x)
        return x.view(x.size(0), -1)


class GazeCaptureEyeFTModel(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.eyeModel = ItrackerEyeBackbone()
        self.eyesFC = nn.Sequential(
            nn.Linear(2 * 12 * 12 * 64, 128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, left_eye: torch.Tensor, right_eye: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        xl = self.eyeModel(left_eye)
        xr = self.eyeModel(right_eye)
        xe = self.eyesFC(torch.cat([xl, xr], dim=1))
        return self.head(torch.cat([xe, pose], dim=1))

