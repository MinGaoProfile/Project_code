from pathlib import Path

import pytest
import yaml

from rnv_seg.config import load_config


def test_defaults_merge_and_environment_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "project": {"experiment_name": "base"},
                "data": {"dataset_root": "${RNV_TEST_ROOT}", "modalities": ["oct"], "patch_size": [8, 8, 8]},
                "model": {"levels": 4},
                "loss": {},
                "training": {},
            }
        ),
        encoding="utf-8",
    )
    child.write_text("defaults: [base.yaml]\nproject:\n  experiment_name: child\n", encoding="utf-8")
    monkeypatch.setenv("RNV_TEST_ROOT", str(tmp_path))
    config = load_config(child)
    assert config["project"]["experiment_name"] == "child"
    assert config["data"]["dataset_root"] == str(tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "octcube_unet_oct_frozen.yaml",
        "octcube_unet_oct_finetune.yaml",
        "octcube_unet_oct_octa.yaml",
        "octcube_unet_multimodal.yaml",
        "octcube_unet_multimodal_attention.yaml",
    ],
)
def test_required_octcube_experiment_configs_load(name: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "experiments" / name)
    assert config["model"]["name"] == "octcube_unet3d"
    assert "oct" in config["data"]["modalities"]
