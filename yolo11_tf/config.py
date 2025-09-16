"""Configuration utilities for YOLO11-TF."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_DEFAULT_CFG_PATH = Path(__file__).resolve().parent / "cfg" / "default.yaml"


@dataclass
class TrainSettings:
    """Training defaults derived from the Ultralytics YOLO11 configuration."""

    imgsz: int = 640
    batch: int = 16
    epochs: int = 100
    lr0: float = 1e-3
    warmup_epochs: float = 3.0
    weight_decay: float = 5e-4

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "TrainSettings":
        data = data or {}
        kwargs = {f.name: data.get(f.name, getattr(cls, f.name)) for f in fields(cls)}
        return cls(**kwargs)


@dataclass
class LossSettings:
    """Loss gain defaults that mirror Ultralytics hyper-parameters."""

    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "LossSettings":
        data = data or {}
        kwargs = {f.name: data.get(f.name, getattr(cls, f.name)) for f in fields(cls)}
        return cls(**kwargs)


@dataclass
class AugmentationConfig:
    """Image augmentation hyper-parameters inspired by Ultralytics YOLO11."""

    mosaic: float = 1.0
    mixup: float = 0.0
    mixup_beta: float = 8.0
    mosaic_scale: tuple[float, float] = (0.5, 1.5)
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    pad_val: float = 114.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "AugmentationConfig":
        data = data or {}
        kwargs = {}
        for f in fields(cls):
            value = data.get(f.name, getattr(cls, f.name))
            if f.name == "mosaic_scale" and isinstance(value, list):
                value = tuple(float(v) for v in value)
            kwargs[f.name] = value
        return cls(**kwargs)


@dataclass
class YOLOConfig:
    """Container combining train, loss and augmentation sections."""

    train: TrainSettings
    loss: LossSettings
    augmentation: AugmentationConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train": {f.name: getattr(self.train, f.name) for f in fields(self.train)},
            "loss": {f.name: getattr(self.loss, f.name) for f in fields(self.loss)},
            "augmentation": {f.name: getattr(self.augmentation, f.name) for f in fields(self.augmentation)},
        }


def default_config_path() -> Path:
    """Return the path to the bundled default configuration file."""

    return _DEFAULT_CFG_PATH


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(dict(base[k]), v)
        else:
            base[k] = v
    return base


def load_config(path: Optional[str] = None) -> YOLOConfig:
    """Load configuration defaults, optionally overriding with a user provided YAML file."""

    with open(_DEFAULT_CFG_PATH, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    if path:
        user_path = Path(path)
        if not user_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {user_path}")
        with open(user_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        base = _deep_update(base, user_cfg)
    return YOLOConfig(
        train=TrainSettings.from_dict(base.get("train")),
        loss=LossSettings.from_dict(base.get("loss")),
        augmentation=AugmentationConfig.from_dict(base.get("augmentation")),
    )


__all__ = [
    "AugmentationConfig",
    "LossSettings",
    "TrainSettings",
    "YOLOConfig",
    "default_config_path",
    "load_config",
]
