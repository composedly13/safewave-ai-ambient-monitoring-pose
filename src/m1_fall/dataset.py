"""Data loader: labels.csv + raw CSI CSVs  →  (N, 1, 64, 100) windows + labels.

Pipeline per session file:
  1. read csv_YYYYMMDD_HHMM.csv, keep raw_0..63, group rows by node_id, sort by ts_ms
  2. per-frame peak-normalize (matches ESP on-device norm) + zero guard subcarriers
  3. slide 100-frame windows (stride; denser stride near fall instants)
  4. label each window 0/1 from labels.csv fall_ts_ms (±halfwidth, fraction rule)

Window unit = ONE node × 100 frames  → tensor (1, 64, 100). No 5-node fusion.
Split unit = session/file, so no session leaks across train/val/test.

The numpy functions (load/normalize/window/label) are torch-free and unit-testable.
Only `M1WindowDataset` and the dataloader helpers pull in torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config

RAW_COLS = [f"raw_{i}" for i in range(64)]


# ── session reading ──────────────────────────────────────────────────────────

@dataclass
class NodeStream:
    node_id: str
    ts_ms: np.ndarray      # (T,) int64, sorted ascending
    frames: np.ndarray     # (T, 64) float32, raw amplitude


def load_session(csv_path: Path) -> List[NodeStream]:
    """Read one collection CSV → one NodeStream per node_id (M1 = per-node)."""
    df = pd.read_csv(csv_path)
    missing = [c for c in (["node_id", "ts_ms"] + RAW_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns {missing[:5]}{'...' if len(missing) > 5 else ''}")

    streams: List[NodeStream] = []
    for node_id, g in df.groupby("node_id"):
        g = g.sort_values("ts_ms")
        streams.append(
            NodeStream(
                node_id=str(node_id),
                ts_ms=g["ts_ms"].to_numpy(dtype=np.int64),
                frames=g[RAW_COLS].to_numpy(dtype=np.float32),
            )
        )
    return streams


# ── normalization ────────────────────────────────────────────────────────────

def normalize_per_frame(frames: np.ndarray, guard_indices: List[int], eps: float) -> np.ndarray:
    """Per-frame peak normalization (max=1.0) over non-guard channels; guards → 0.

    Mirrors the ESP packet-time normalization so training == inference distribution.
    Absolute amplitude is intentionally destroyed — only relative channel pattern
    and temporal change survive.
    """
    out = frames.astype(np.float32).copy()
    guard = np.zeros(out.shape[1], dtype=bool)
    guard[guard_indices] = True
    out[:, guard] = 0.0

    peak = np.max(np.abs(out[:, ~guard]), axis=1, keepdims=True)  # (T, 1)
    out[:, ~guard] = out[:, ~guard] / np.maximum(peak, eps)
    return out


# ── fall masking + windowing ─────────────────────────────────────────────────

def fall_frame_mask(ts_ms: np.ndarray, fall_ts: List[int], halfwidth_ms: int) -> np.ndarray:
    """Boolean (T,): True where a frame lies within ±halfwidth of any fall instant."""
    mask = np.zeros(len(ts_ms), dtype=bool)
    for t in fall_ts:
        mask |= (ts_ms >= t - halfwidth_ms) & (ts_ms <= t + halfwidth_ms)
    return mask


def window_starts(
    n_frames: int, window: int, stride: int,
    fall_stride: Optional[int], frame_is_fall: Optional[np.ndarray],
) -> List[int]:
    """Start indices for sliding windows; dense `fall_stride` near fall frames."""
    if n_frames < window:
        return []
    starts = set(range(0, n_frames - window + 1, stride))
    if fall_stride and frame_is_fall is not None and frame_is_fall.any():
        for s in range(0, n_frames - window + 1, fall_stride):
            if frame_is_fall[s : s + window].any():
                starts.add(s)
    return sorted(starts)


def make_windows(
    stream: NodeStream, cfg: Config, fall_ts: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """One node stream → (W, 1, 64, 100) windows + (W,) binary labels."""
    W = cfg.tensor.window_frames
    norm = normalize_per_frame(stream.frames, cfg.tensor.guard_indices, cfg.normalize.eps)

    is_fall = fall_frame_mask(stream.ts_ms, fall_ts, cfg.label.fall_halfwidth_ms) if fall_ts else None
    starts = window_starts(
        len(norm), W, cfg.window.stride_frames, cfg.window.fall_stride_frames, is_fall
    )

    X, y = [], []
    for s in starts:
        win = norm[s : s + W]                       # (100, 64)  [time, subcarrier]
        X.append(win.T[np.newaxis, :, :])           # (1, 64, 100) [chan, subcarrier, time]
        if is_fall is None:
            y.append(0)
        else:
            frac = float(is_fall[s : s + W].mean())
            y.append(int(frac >= cfg.label.fall_pos_frac))

    if not X:
        return np.empty((0, 1, 64, W), np.float32), np.empty((0,), np.int64)
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64)


# ── label table + session-level split ────────────────────────────────────────

@dataclass
class SessionSpec:
    csv_file: str
    session_label: str          # NORMAL | FALL
    fall_ts: List[int]
    split: Optional[str]        # explicit override or None


def read_labels(cfg: Config) -> List[SessionSpec]:
    df = pd.read_csv(cfg.paths.labels_csv, dtype=str).fillna("")
    specs: List[SessionSpec] = []
    for _, r in df.iterrows():
        ts_field = r.get("fall_ts_ms", "").strip()
        fall_ts = [int(x) for x in ts_field.split(";") if x.strip()] if ts_field else []
        label = r["session_label"].strip().upper()
        if label == "FALL" and not fall_ts:
            raise ValueError(f"{r['csv_file']}: FALL session needs at least one fall_ts_ms")
        specs.append(SessionSpec(
            csv_file=r["csv_file"].strip(),
            session_label=label,
            fall_ts=fall_ts,
            split=(r.get("split", "").strip().lower() or None),
        ))
    return specs


def assign_splits(specs: List[SessionSpec], cfg: Config) -> Dict[str, str]:
    """Map csv_file → {'train','val','test'} at SESSION granularity (no leakage)."""
    rng = np.random.default_rng(cfg.split.seed)
    fixed = {s.csv_file: s.split for s in specs if s.split}
    pool = [s.csv_file for s in specs if not s.split]
    rng.shuffle(pool)

    n = len(pool)
    n_test = int(round(n * cfg.split.test_frac))
    n_val = int(round(n * cfg.split.val_frac))
    out = dict(fixed)
    for i, f in enumerate(pool):
        out[f] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


# ── build a split into arrays ────────────────────────────────────────────────

def build_split(cfg: Config, split: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """All windows for one split. Returns X (N,1,64,100), y (N,), groups (session id per window)."""
    specs = read_labels(cfg)
    split_of = assign_splits(specs, cfg)
    raw_dir = Path(cfg.paths.raw_dir)

    Xs, ys, groups = [], [], []
    for spec in specs:
        if split_of[spec.csv_file] != split:
            continue
        for stream in load_session(raw_dir / spec.csv_file):
            Xw, yw = make_windows(stream, cfg, spec.fall_ts)
            if len(Xw) == 0:
                continue
            Xs.append(Xw)
            ys.append(yw)
            groups.extend([f"{spec.csv_file}:{stream.node_id}"] * len(Xw))

    if not Xs:
        return np.empty((0, 1, 64, cfg.tensor.window_frames), np.float32), np.empty((0,), np.int64), []
    return np.concatenate(Xs), np.concatenate(ys), groups


# ── torch glue (imported lazily so numpy logic stays torch-free) ──────────────

def _torch():
    import torch  # noqa: WPS433 — local import keeps numpy path dependency-free
    return torch


class M1WindowDataset:
    """torch.utils.data.Dataset over prebuilt windows. (Duck-typed to avoid a hard
    torch import at module load.)"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)  # BCEWithLogits target

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        torch = _torch()
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])


def make_weighted_sampler(y: np.ndarray):
    """WeightedRandomSampler balancing the rare fall class."""
    torch = _torch()
    counts = np.bincount(y.astype(int), minlength=2).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    weights = inv[y.astype(int)]
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(y),
        replacement=True,
    )


def make_dataloaders(cfg: Config):
    """Convenience: build train/val DataLoaders honoring imbalance.strategy."""
    torch = _torch()
    Xtr, ytr, _ = build_split(cfg, "train")
    Xva, yva, _ = build_split(cfg, "val")

    train_ds = M1WindowDataset(Xtr, ytr)
    val_ds = M1WindowDataset(Xva, yva)

    sampler = None
    shuffle = True
    if cfg.imbalance.strategy == "weighted_sampler" and len(ytr):
        sampler = make_weighted_sampler(ytr)
        shuffle = False

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.train.batch_size, sampler=sampler, shuffle=shuffle, drop_last=False
    )
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)
    return train_dl, val_dl


if __name__ == "__main__":
    # Smoke check: prints per-split window counts + class balance.
    import sys
    from .config import load_config

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/m1_fall.yaml")
    for split in ("train", "val", "test"):
        X, y, g = build_split(cfg, split)
        pos = int(y.sum()) if len(y) else 0
        print(f"{split:5s}: {len(y):6d} windows | pos(fall)={pos} | sessions={len(set(g))}")
