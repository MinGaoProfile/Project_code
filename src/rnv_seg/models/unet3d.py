from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .base import SegmentationOutput


def _group_count(channels: int) -> int:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return groups


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 16,
        levels: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if levels < 2:
            raise ValueError("UNet3D requires at least two levels")
        channels = [base_channels * (2**index) for index in range(levels)]
        self.encoder = nn.ModuleList()
        previous = in_channels
        for channel in channels:
            self.encoder.append(ConvBlock(previous, channel, dropout))
            previous = channel
        self.pools = nn.ModuleList([nn.MaxPool3d(2) for _ in range(levels - 1)])
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for index in range(levels - 1, 0, -1):
            self.upconvs.append(
                nn.ConvTranspose3d(channels[index], channels[index - 1], kernel_size=2, stride=2)
            )
            self.decoder.append(ConvBlock(channels[index], channels[index - 1], dropout))
        self.head = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        current = x
        for index, block in enumerate(self.encoder):
            current = block(current)
            if index < len(self.pools):
                skips.append(current)
                current = self.pools[index](current)
        for upconv, block, skip in zip(self.upconvs, self.decoder, reversed(skips), strict=True):
            current = upconv(current)
            if current.shape[-3:] != skip.shape[-3:]:
                current = F.interpolate(current, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            current = block(torch.cat([current, skip], dim=1))
        return self.head(current)

    def forward(self, x: torch.Tensor, target: torch.Tensor | None = None) -> SegmentationOutput:
        del target
        return SegmentationOutput(logits=self.forward_logits(x))

