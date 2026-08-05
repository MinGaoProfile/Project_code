from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .base import SegmentationOutput
from .unet3d import UNet3D


class DiffusionUNet3D(nn.Module):
    """Conditional diffusion-style mask refiner around a coarse 3D U-Net.

    During training a noised ground-truth mask is denoised at a random timestep.
    During inference the coarse probability map is refined deterministically at t=0.
    This keeps the same segmentation interface as the baseline while exposing a
    diffusion auxiliary objective.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 16,
        levels: int = 4,
        dropout: float = 0.0,
        diffusion_steps: int = 50,
        diffusion_loss_weight: float = 0.2,
    ) -> None:
        super().__init__()
        if out_channels != 1:
            raise ValueError("DiffusionUNet3D currently supports binary segmentation")
        self.coarse = UNet3D(in_channels, out_channels, base_channels, levels, dropout)
        self.refiner = UNet3D(in_channels + 2, 1, max(8, base_channels // 2), levels, dropout)
        self.diffusion_steps = int(diffusion_steps)
        self.diffusion_loss_weight = float(diffusion_loss_weight)
        betas = torch.linspace(1.0e-4, 0.02, self.diffusion_steps)
        self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, dim=0))

    def _predict_noise(self, image: torch.Tensor, noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_channel = t.float().view(-1, 1, 1, 1, 1) / max(self.diffusion_steps - 1, 1)
        time_channel = time_channel.expand(-1, 1, *noisy.shape[-3:])
        return self.refiner.forward_logits(torch.cat([image, noisy, time_channel], dim=1))

    def forward(self, image: torch.Tensor, target: torch.Tensor | None = None) -> SegmentationOutput:
        coarse_logits = self.coarse.forward_logits(image)
        batch = image.shape[0]
        if target is not None and self.training:
            timestep = torch.randint(0, self.diffusion_steps, (batch,), device=image.device)
            alpha_bar = self.alpha_bars[timestep].view(-1, 1, 1, 1, 1)
            noise = torch.randn_like(target)
            noisy = alpha_bar.sqrt() * target + (1.0 - alpha_bar).sqrt() * noise
            predicted_noise = self._predict_noise(image, noisy, timestep)
            reconstructed = (noisy - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
            refined_logits = torch.logit(reconstructed.clamp(1.0e-4, 1.0 - 1.0e-4))
            logits = 0.5 * (coarse_logits + refined_logits)
            diffusion_loss = F.mse_loss(predicted_noise, noise) * self.diffusion_loss_weight
            return SegmentationOutput(logits=logits, auxiliary_losses={"diffusion": diffusion_loss})

        probability = torch.sigmoid(coarse_logits)
        timestep = torch.zeros(batch, dtype=torch.long, device=image.device)
        predicted_noise = self._predict_noise(image, probability, timestep)
        alpha_bar = self.alpha_bars[0]
        refined = (probability - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
        refined_logits = torch.logit(refined.clamp(1.0e-4, 1.0 - 1.0e-4))
        return SegmentationOutput(logits=0.5 * (coarse_logits + refined_logits))

