"""Typed config loaded from configs/m1_fall.yaml.

Hard contracts (tensor shape, normalization mode, ONNX I/O) are validated on load
so a typo can't silently desync the training pipeline from the rp5 inference repo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

# rp5 db_spec_v2 §7 — the one shape the whole pipeline is built around.
CONTRACT_CHANNELS = 64
CONTRACT_FRAMES = 100


@dataclass
class TensorCfg:
    n_channels: int
    window_frames: int
    sample_rate_hz: int
    guard_indices: List[int]


@dataclass
class NormalizeCfg:
    mode: str
    eps: float


@dataclass
class WindowCfg:
    stride_frames: int
    fall_stride_frames: Optional[int]
    drop_last_partial: bool


@dataclass
class LabelCfg:
    task: str
    fall_halfwidth_ms: int
    fall_pos_frac: float


@dataclass
class SplitCfg:
    by: str
    val_frac: float
    test_frac: float
    seed: int


@dataclass
class ImbalanceCfg:
    strategy: str
    focal_gamma: float
    pos_weight: Optional[float]


@dataclass
class ModelCfg:
    conv_channels: List[int]
    gru_hidden: int
    gru_layers: int
    dropout: float


@dataclass
class TrainCfg:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    early_stop_patience: int
    primary_metric: str
    report_metrics: List[str]


@dataclass
class PathsCfg:
    raw_dir: str
    labels_csv: str
    cache_dir: str
    out_dir: str


@dataclass
class ExportCfg:
    opset: int
    input_name: str
    output_name: str
    onnx_path: str


@dataclass
class Config:
    tensor: TensorCfg
    normalize: NormalizeCfg
    window: WindowCfg
    label: LabelCfg
    split: SplitCfg
    imbalance: ImbalanceCfg
    model: ModelCfg
    train: TrainCfg
    paths: PathsCfg
    export: ExportCfg
    _raw: dict = field(default_factory=dict, repr=False)

    # ── validation ────────────────────────────────────────────────────────────
    def validate(self) -> "Config":
        t = self.tensor
        if t.n_channels != CONTRACT_CHANNELS or t.window_frames != CONTRACT_FRAMES:
            raise ValueError(
                f"tensor shape contract violation: got ({t.n_channels},{t.window_frames}), "
                f"must be ({CONTRACT_CHANNELS},{CONTRACT_FRAMES}). "
                "rp5 expects (B,1,64,100); 192 is the forbidden PulseFi legacy."
            )
        if max(t.guard_indices, default=0) >= t.n_channels or min(t.guard_indices, default=0) < 0:
            raise ValueError("guard_indices out of [0, n_channels) range")
        if self.normalize.mode != "per_frame_peak":
            raise ValueError("normalize.mode must be 'per_frame_peak' (matches ESP on-device norm)")
        if self.label.task not in ("binary", "multiclass"):
            raise ValueError("label.task must be 'binary' or 'multiclass'")
        if not (0.0 < self.label.fall_pos_frac <= 1.0):
            raise ValueError("label.fall_pos_frac must be in (0, 1]")
        if self.imbalance.strategy not in ("weighted_sampler", "focal_loss", "none"):
            raise ValueError("imbalance.strategy invalid")
        if self.export.opset != 17:
            raise ValueError("export.opset must be 17 (rp5 ONNX contract)")
        return self


def load_config(path: str | Path) -> Config:
    """Parse + validate the YAML config."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = Config(
        tensor=TensorCfg(**raw["tensor"]),
        normalize=NormalizeCfg(**raw["normalize"]),
        window=WindowCfg(**raw["window"]),
        label=LabelCfg(**raw["label"]),
        split=SplitCfg(**raw["split"]),
        imbalance=ImbalanceCfg(**raw["imbalance"]),
        model=ModelCfg(**raw["model"]),
        train=TrainCfg(**raw["train"]),
        paths=PathsCfg(**raw["paths"]),
        export=ExportCfg(**raw["export"]),
        _raw=raw,
    )
    return cfg.validate()
