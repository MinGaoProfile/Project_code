from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from rnv_seg.engine import sliding_window_logits, train
from rnv_seg.losses import build_loss
from rnv_seg.models.unet3d import UNet3D


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        image = torch.zeros(1, 8, 8, 8)
        target = torch.zeros(1, 8, 8, 8)
        target[:, 2:6, 2:6, 2:6] = 1
        image.copy_(target)
        return {"image": image, "target": target, "sample_id": f"synthetic-{index}"}


def test_one_epoch_training_and_sliding_window(tmp_path: Path) -> None:
    loader = DataLoader(TinyDataset(), batch_size=1)
    model = UNet3D(1, base_channels=2, levels=2)
    criterion = build_loss({"components": {"dice": 1.0, "bce": 1.0}})
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    config = {
        "training": {
            "epochs": 1,
            "gradient_accumulation_steps": 1,
            "amp": False,
            "checkpoint_metric": "dice",
            "checkpoint_mode": "max",
            "early_stopping_patience": 2,
        },
        "inference": {"threshold": 0.5},
        "metrics": {
            "lesion": {"enabled": True, "connectivity": 26, "minimum_voxels": 1, "match_iou": 0.1},
            "volume": {"voxel_spacing_mm": [1.0, 1.0, 1.0]},
            "surface": {"enabled": False},
        },
    }
    for child in ("checkpoints", "logs"):
        (tmp_path / child).mkdir()
    summary = train(
        model,
        loader,
        loader,
        criterion,
        optimizer,
        None,
        config,
        tmp_path,
        torch.device("cpu"),
    )
    assert summary["epochs_completed"] == 1
    assert (tmp_path / "checkpoints" / "last.pt").is_file()
    logits = sliding_window_logits(model.eval(), torch.zeros(1, 8, 8, 8), (4, 4, 4), 0.5)
    assert logits.shape == (1, 1, 8, 8, 8)
