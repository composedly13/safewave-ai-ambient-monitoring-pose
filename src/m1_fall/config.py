"""Typed config loaded from configs/m1_fall.yaml.

Hard contracts (tensor shape, normalization mode, ONNX I/O) are validated on load
so a typo can't silently desync the training pipeline from the rp5 inference repo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

# rp5 db_spec_v2 §7 — subcarriers × frames are the immutable half of the shape.
# The node axis (n_nodes) is the conv INPUT-CHANNEL axis: 5 nodes are stacked as
# (B, n_nodes, 64, 100) for early fusion. Concatenating nodes on the subcarrier
# axis (the legacy 192 = 3×64) is forbidden.
CONTRACT_CHANNELS = 64
CONTRACT_FRAMES = 100
MAX_NODES = 8


@dataclass
class TensorCfg:
    n_channels: int
    window_frames: int
    sample_rate_hz: int
    guard_indices: List[int]
    # Number of CSI nodes stacked on the conv input-channel axis.
    # 1 = the old single-node layout (B,1,64,100); 5 = the 5-node deployment.
    # Defaults to 1 so configs written before this field keep loading unchanged.
    n_nodes: int = 1


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
class PretrainCfg:
    """Optional warm start from an earlier checkpoint (e.g. runs/best_B2_final.pt).

    Absent from the YAML -> Config.pretrain stays None and training is unchanged.
    load='full' restores every key; load='backbone' restores only the conv stack
    (`convs.*`) and leaves GRU + head randomly initialized. Either way the load is
    strict=True — a silently half-restored model is the failure mode this guards.

    NOTE: a checkpoint trained at a different n_nodes cannot be loaded — the first
    conv's input-channel count differs, and strict=True fails loudly (intended).
    """
    ckpt: str
    load: str = "full"                  # {full, backbone}
    freeze_epochs: int = 0              # first N epochs: convs frozen, GRU+head train
    backbone_lr_mult: float = 1.0       # after thaw: backbone lr = train.lr * this


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
    pretrain: Optional[PretrainCfg] = None
    _raw: dict = field(default_factory=dict, repr=False)

    # ── validation ────────────────────────────────────────────────────────────
    def validate(self) -> "Config":
        t = self.tensor
        if t.n_channels != CONTRACT_CHANNELS or t.window_frames != CONTRACT_FRAMES:
            raise ValueError(
                f"tensor shape contract violation: got ({t.n_channels},{t.window_frames}), "
                f"must be ({CONTRACT_CHANNELS},{CONTRACT_FRAMES}). "
                "Input is (B,n_nodes,64,100); 192 is the forbidden PulseFi legacy "
                "(nodes stack on the channel axis, never concat on subcarriers)."
            )
        if not (1 <= int(t.n_nodes) <= MAX_NODES):
            raise ValueError(
                f"tensor.n_nodes must be in [1, {MAX_NODES}], got {t.n_nodes}. "
                "1 = legacy single-node, 5 = the 5-node deployment."
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
        if self.pretrain is not None:
            p = self.pretrain
            # Every checkpoint that exists today is single-node: conv1.weight is
            # (16,1,3,3), while n_nodes=5 needs (16,5,3,3). strict=True would catch it
            # at load time, but only after the data pipeline has already been built —
            # and the error would be a shape dump, not an explanation. Refuse here.
            if int(t.n_nodes) != 1:
                raise ValueError(
                    f"pretrain 블록이 설정됐는데 tensor.n_nodes={t.n_nodes} 다. "
                    f"기존 체크포인트({p.ckpt})는 전부 단일노드라 conv1 입력채널이 1이고, "
                    f"{t.n_nodes}노드 모델에는 strict=True로 로드되지 않는다. "
                    "5노드는 스크래치 재학습이 정상 경로다 — pretrain 블록을 지워라. "
                    "(단일노드로 되돌릴 때만 n_nodes: 1 과 함께 다시 쓸 것)"
                )
            if not p.ckpt:
                raise ValueError("pretrain.ckpt must be a non-empty path")
            if p.load not in ("full", "backbone"):
                raise ValueError("pretrain.load must be 'full' or 'backbone'")
            if int(p.freeze_epochs) < 0:
                raise ValueError("pretrain.freeze_epochs must be >= 0")
            if not (0.0 < float(p.backbone_lr_mult) <= 1.0):
                raise ValueError("pretrain.backbone_lr_mult must be in (0, 1]")
        return self


def load_config(path: str | Path) -> Config:
    """Parse + validate the YAML config."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    # `pretrain` is optional — configs written before it must keep loading unchanged.
    raw_pretrain = raw.get("pretrain")
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
        pretrain=PretrainCfg(**raw_pretrain) if raw_pretrain else None,
        _raw=raw,
    )
    return cfg.validate()
