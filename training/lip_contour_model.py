from __future__ import annotations

import torch
from torch import nn

from model import ConvBlock, SeparableConv


class LipContourRegressor(nn.Module):
    def __init__(self, point_count: int = 64, width: int = 32, input_channels: int = 3) -> None:
        super().__init__()
        self.point_count = int(point_count)
        self.input_channels = int(input_channels)

        self.stem = ConvBlock(self.input_channels, width)
        self.down1 = ConvBlock(width, width * 2, stride=2)
        self.down2 = ConvBlock(width * 2, width * 4, stride=2)
        self.down3 = ConvBlock(width * 4, width * 6, stride=2)
        self.down4 = ConvBlock(width * 6, width * 8, stride=2)
        self.context = nn.Sequential(
            SeparableConv(width * 8, width * 8),
            nn.Dropout2d(0.08),
            SeparableConv(width * 8, width * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 8, width * 8),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(width * 8, self.point_count * 2),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.context(x)
        points = torch.sigmoid(self.head(self.pool(x)))
        return points.view(image.shape[0], self.point_count, 2)


def build_lip_contour_model(point_count: int = 64, width: int = 32, input_channels: int = 3) -> LipContourRegressor:
    return LipContourRegressor(point_count=point_count, width=width, input_channels=input_channels)
