from __future__ import annotations

import torch
from torch import nn

from model import ConvBlock, SeparableConv


class LipCurveRegressor(nn.Module):
    def __init__(
        self,
        curve_count: int = 4,
        points_per_curve: int = 24,
        width: int = 32,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        self.curve_count = int(curve_count)
        self.points_per_curve = int(points_per_curve)
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
            nn.Linear(width * 8, self.curve_count * self.points_per_curve * 2),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.context(x)
        curves = torch.sigmoid(self.head(self.pool(x)))
        return curves.view(image.shape[0], self.curve_count, self.points_per_curve, 2)


def build_lip_curve_model(
    curve_count: int = 4,
    points_per_curve: int = 24,
    width: int = 32,
    input_channels: int = 3,
) -> LipCurveRegressor:
    return LipCurveRegressor(
        curve_count=curve_count,
        points_per_curve=points_per_curve,
        width=width,
        input_channels=input_channels,
    )
