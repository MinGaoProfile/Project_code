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
    """Legacy double-convolution block used by pretrained model adapters."""

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


class ResidualConvBlock(nn.Module):
    """Two-convolution 3D ResNet block with an identity or projected shortcut."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation1 = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(_group_count(out_channels), out_channels),
            )
        )
        self.output_activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        current = self.activation1(self.norm1(self.conv1(x)))
        current = self.dropout(current)
        current = self.norm2(self.conv2(current))
        return self.output_activation(current + residual)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 24,
        levels: int = 5,
        dropout: float = 0.0,
        residual_blocks: bool = True,
    ) -> None:
        super().__init__()
        if levels < 2:
            raise ValueError("UNet3D requires at least two levels")
        channels = [base_channels * (2**index) for index in range(levels)]
        block_type = ResidualConvBlock if residual_blocks else ConvBlock
        self.encoder = nn.ModuleList()
        previous = in_channels
        for channel in channels:
            self.encoder.append(block_type(previous, channel, dropout))
            previous = channel
        self.pools = nn.ModuleList([nn.MaxPool3d(2) for _ in range(levels - 1)])
        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for index in range(levels - 1, 0, -1):
            self.upconvs.append(
                nn.ConvTranspose3d(channels[index], channels[index - 1], kernel_size=2, stride=2)
            )
            self.decoder.append(
                block_type(channels[index], channels[index - 1], dropout)
            )
        self.head = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
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
        return current

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def forward(self, x: torch.Tensor, target: torch.Tensor | None = None) -> SegmentationOutput:
        del target
        return SegmentationOutput(logits=self.forward_logits(x))


class MultiTaskUNet3D(UNet3D):
    """3D U-Net with independent sigmoid heads for overlapping binary targets."""

    def __init__(
        self,
        in_channels: int,
        output_heads: list[str],
        base_channels: int = 24,
        levels: int = 5,
        dropout: float = 0.0,
        residual_blocks: bool = True,
    ) -> None:
        if len(output_heads) < 2 or len(set(output_heads)) != len(output_heads):
            raise ValueError("output_heads must contain at least two unique task names")
        super().__init__(
            in_channels,
            1,
            base_channels,
            levels,
            dropout,
            residual_blocks,
        )
        del self.head
        self.output_heads = tuple(output_heads)
        self.heads = nn.ModuleDict(
            {name: nn.Conv3d(base_channels, 1, kernel_size=1) for name in self.output_heads}
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return torch.cat([self.heads[name](features) for name in self.output_heads], dim=1)
