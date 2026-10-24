"""Predicting spatial gene expression from H&E with frozen pathology foundation models."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str = "project") -> dict:
    with open(ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)
