from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            SeparableConv(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.proj(torch.cat([x, skip], dim=1))


class MakeupParsingSmall(nn.Module):
    """Small encoder-decoder segmentation model for on-device makeup masks."""

    def __init__(self, num_classes: int = 9, width: int = 32, input_channels: int = 3) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.stem = ConvBlock(self.input_channels, width)
        self.down1 = ConvBlock(width, width * 2, stride=2)
        self.down2 = ConvBlock(width * 2, width * 4, stride=2)
        self.down3 = ConvBlock(width * 4, width * 6, stride=2)
        self.down4 = ConvBlock(width * 6, width * 8, stride=2)

        self.context = nn.Sequential(
            SeparableConv(width * 8, width * 8),
            nn.Dropout2d(0.1),
            SeparableConv(width * 8, width * 8),
        )

        self.up3 = UpBlock(width * 8, width * 6, width * 6)
        self.up2 = UpBlock(width * 6, width * 4, width * 4)
        self.up1 = UpBlock(width * 4, width * 2, width * 2)
        self.up0 = UpBlock(width * 2, width, width)
        self.head = nn.Conv2d(width, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = self.down4(s3)
        x = self.context(x)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        logits = self.head(x)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


def build_model(num_classes: int = 9, width: int = 32, input_channels: int = 3) -> MakeupParsingSmall:
    return MakeupParsingSmall(num_classes=num_classes, width=width, input_channels=input_channels)
