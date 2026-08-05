from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SegmentationOutput:
    logits: torch.Tensor
    auxiliary_losses: dict[str, torch.Tensor] = field(default_factory=dict)

