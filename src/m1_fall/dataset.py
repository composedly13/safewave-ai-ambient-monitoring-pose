"""Data loader: labels.csv + raw CSI CSVs  →  (N, n_nodes, 64, 100) windows + labels.

5-node early fusion. Pipeline per session file:
  1. read csv_YYYYMMDD_HHMM.csv (stream_id, node_id, ts_ms, raw_0..63)
  2. per-row EPOCH from stream_id ("1781165460004-0" → 1781165460004).
     THE EPOCH IS THE ONLY SHARED CLOCK: ts_ms is each board's own device clock
     (lower 32 bits of Unix ms), so aligning nodes on ts_ms is forbidden.
  3. build a session-wide 100 Hz epoch grid; snap each node's frames to the
     nearest grid slot (±5 ms); a slot with no frame for a node stays all-zero
  4. per-frame peak-normalize per node (matches ESP on-device norm) + zero guards
  5. slide 100-frame windows (stride; denser stride near fall instants)
     → window tensor (n_nodes, 64, 100); nodes NEVER concat on the subcarrier axis
  6. label each window 0/1 from labels.csv fall_ts_ms in the EPOCH domain
     (±halfwidth, fraction rule). Values < 2^32 are legacy device-clock labels
     (verify_sync_v2 output) and are auto-converted via the file's clock offset.

Split unit = session/file, so no session leaks across train/val/test.

The numpy functions (load/normalize/grid/window/label) are torch-free and
unit-testable. Only `M1WindowDataset` and the dataloader helpers pull in torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config

RAW_COLS = [f"raw_{i}" for i in range(64)]

# ts_ms is the lower 32 bits of Unix ms (firmware packet.h), so it is always
# < 2^32 ≈ 4.29e9, while epoch ms is ≈ 1.79e12. A fall timestamp ≥ 2^32 is
# therefore unambiguously epoch-domain (mark_falls.py); below it is a legacy
# device-clock label (verify_sync_v2 wrote fall_ts_ms as `epoch - offset`).
EPOCH_DOMAIN_MIN = 2 ** 32


# ── session reading ──────────────────────────────────────────────────────────

@dataclass
class SessionData:
    """One collection file, aligned onto the shared 100 Hz epoch grid."""
    grid_epoch_ms: np.ndarray   # (T,) int64 — grid slot times, 10 ms pitch
    frames: np.ndarray          # (T, n_nodes, 64) float32 — normalized; missing slots 0
    presence: np.ndarray        # (T, n_nodes) bool — True where a real frame landed
    offset_ms: int              # median(epoch - ts_ms) — converts legacy labels to epoch
    node_ids: List[int]         # node ids seen in the file (1-based)


def _epoch_from_stream_id(stream_id: pd.Series, csv_name: str) -> np.ndarray:
    """'1781165460004-0' → 1781165460004 (rp5 receive epoch ms, the shared clock)."""
    head = stream_id.astype(str).str.split("-", n=1).str[0]
    bad = ~head.str.fullmatch(r"\d+")
    if bad.any():
        sample = stream_id[bad].iloc[0]
        raise ValueError(
            f"{csv_name}: stream_id에서 epoch를 못 읽음 (예: {sample!r}). "
            "'epochms-seq' 형태여야 한다."
        )
    return head.astype(np.int64).to_numpy()


def load_session(csv_path: Path, cfg: Config) -> SessionData:
    """Read one collection CSV → epoch-gridded (T, n_nodes, 64) session tensor."""
    df = pd.read_csv(csv_path)
    need = ["stream_id", "node_id", "ts_ms"] + RAW_COLS
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name}: missing columns {missing[:5]}{'...' if len(missing) > 5 else ''} "
            f"(found: {list(df.columns)[:8]}...)"
        )
    if len(df) == 0:
        raise ValueError(f"{csv_path.name}: 빈 파일")

    n_nodes = cfg.tensor.n_nodes
    period_ms = round(1000 / cfg.tensor.sample_rate_hz)          # 10 ms @ 100 Hz
    tol_ms = period_ms // 2                                      # nearest = ±5 ms

    epoch = _epoch_from_stream_id(df["stream_id"], csv_path.name)
    ts_ms = df["ts_ms"].to_numpy(dtype=np.int64)
    # Median absorbs per-board clock skew; a ts_ms 32-bit wraparound mid-file makes
    # this offset bimodal — verify_sync_v2 flags such sessions as discard-worthy.
    offset_ms = int(np.median(epoch - ts_ms))

    node_raw = pd.to_numeric(df["node_id"], errors="coerce")
    if node_raw.isna().any():
        raise ValueError(
            f"{csv_path.name}: node_id가 숫자가 아님: {sorted(df['node_id'].astype(str).unique())[:8]}"
        )
    node = node_raw.astype(np.int64).to_numpy()
    out_of_range = sorted(set(node[(node < 1) | (node > n_nodes)].tolist()))
    if out_of_range:
        raise ValueError(
            f"{csv_path.name}: node_id {out_of_range} 는 1..{n_nodes} 범위 밖. "
            "tensor.n_nodes 설정과 수집 노드 번호가 안 맞는다."
        )

    # Session-wide 100 Hz epoch grid.
    t0 = int(epoch.min())
    n_grid = int((int(epoch.max()) - t0) // period_ms) + 1
    frames = np.zeros((n_grid, n_nodes, 64), dtype=np.float32)
    presence = np.zeros((n_grid, n_nodes), dtype=bool)
    grid_epoch_ms = t0 + period_ms * np.arange(n_grid, dtype=np.int64)

    raw = df[RAW_COLS].to_numpy(dtype=np.float32)
    for nid in sorted(set(node.tolist())):
        sel = node == nid
        e = epoch[sel]
        norm = normalize_per_frame(raw[sel], cfg.tensor.guard_indices, cfg.normalize.eps)

        idx = np.round((e - t0) / period_ms).astype(np.int64)
        resid = np.abs(e - (t0 + idx * period_ms))
        keep = (resid <= tol_ms) & (idx >= 0) & (idx < n_grid)
        idx, resid, norm = idx[keep], resid[keep], norm[keep]

        # Collisions (two frames → one slot): keep the one closest to the slot time.
        order = np.lexsort((resid, idx))            # sort by idx, then |residual|
        idx, norm = idx[order], norm[order]
        first = np.ones(len(idx), dtype=bool)
        first[1:] = idx[1:] != idx[:-1]
        idx, norm = idx[first], norm[first]

        frames[idx, nid - 1] = norm
        presence[idx, nid - 1] = True

    return SessionData(
        grid_epoch_ms=grid_epoch_ms,
        frames=frames,
        presence=presence,
        offset_ms=offset_ms,
        node_ids=sorted(set(node.tolist())),
    )


# ── normalization ────────────────────────────────────────────────────────────

def normalize_per_frame(frames: np.ndarray, guard_indices: List[int], eps: float) -> np.ndarray:
    """Per-frame peak normalization (max=1.0) over non-guard channels; guards → 0.

    Applied per node per frame, INDEPENDENTLY — mirrors the ESP packet-time
    normalization so training == inference distribution. Absolute amplitude is
    intentionally destroyed — only relative channel pattern and temporal change
    survive. Re-normalizing across nodes or across time is forbidden.
    """
    out = frames.astype(np.float32).copy()
    guard = np.zeros(out.shape[1], dtype=bool)
    guard[guard_indices] = True
    out[:, guard] = 0.0

    peak = np.max(np.abs(out[:, ~guard]), axis=1, keepdims=True)  # (T, 1)
    out[:, ~guard] = out[:, ~guard] / np.maximum(peak, eps)
    return out


# ── fall labels (epoch domain) ───────────────────────────────────────────────

def fall_ts_to_epoch(fall_ts: List[int], offset_ms: int) -> List[int]:
    """Normalize label timestamps into the epoch domain.

    mark_falls.py records epoch ms (≥ 2^32) — used as-is. Legacy labels written
    by verify_sync_v2 are device-clock ts_ms (< 2^32) — converted with the
    file's own offset. The 400× magnitude gap makes detection unambiguous.
    """
    return [t if t >= EPOCH_DOMAIN_MIN else t + offset_ms for t in fall_ts]


def fall_frame_mask(grid_epoch_ms: np.ndarray, fall_epochs: List[int], halfwidth_ms: int) -> np.ndarray:
    """Boolean (T,): True where a grid slot lies within ±halfwidth of any fall instant."""
    mask = np.zeros(len(grid_epoch_ms), dtype=bool)
    for t in fall_epochs:
        mask |= (grid_epoch_ms >= t - halfwidth_ms) & (grid_epoch_ms <= t + halfwidth_ms)
    return mask


# ── windowing ────────────────────────────────────────────────────────────────

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
    session: SessionData, cfg: Config, fall_ts: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """One gridded session → (W, n_nodes, 64, 100) windows + (W,) binary labels."""
    W = cfg.tensor.window_frames
    n_nodes = cfg.tensor.n_nodes

    is_fall = None
    if fall_ts:
        fall_epochs = fall_ts_to_epoch(fall_ts, session.offset_ms)
        is_fall = fall_frame_mask(session.grid_epoch_ms, fall_epochs, cfg.label.fall_halfwidth_ms)

    starts = window_starts(
        len(session.frames), W, cfg.window.stride_frames, cfg.window.fall_stride_frames, is_fall
    )

    X, y = [], []
    for s in starts:
        win = session.frames[s : s + W]             # (100, n_nodes, 64) [time, node, subc]
        X.append(win.transpose(1, 2, 0))            # (n_nodes, 64, 100) [node, subc, time]
        if is_fall is None:
            y.append(0)
        else:
            frac = float(is_fall[s : s + W].mean())
            y.append(int(frac >= cfg.label.fall_pos_frac))

    if not X:
        return np.empty((0, n_nodes, 64, W), np.float32), np.empty((0,), np.int64)
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
    """All windows for one split. Returns X (N,n_nodes,64,100), y (N,), groups (session per window)."""
    specs = read_labels(cfg)
    split_of = assign_splits(specs, cfg)
    raw_dir = Path(cfg.paths.raw_dir)

    Xs, ys, groups = [], [], []
    for spec in specs:
        if split_of[spec.csv_file] != split:
            continue
        session = load_session(raw_dir / spec.csv_file, cfg)
        Xw, yw = make_windows(session, cfg, spec.fall_ts)
        if len(Xw) == 0:
            continue
        Xs.append(Xw)
        ys.append(yw)
        groups.extend([spec.csv_file] * len(Xw))

    if not Xs:
        return (np.empty((0, cfg.tensor.n_nodes, 64, cfg.tensor.window_frames), np.float32),
                np.empty((0,), np.int64), [])
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
    # Smoke check: per-split window counts, class balance, node coverage.
    import sys
    from .config import load_config

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/m1_fall.yaml")
    specs = read_labels(cfg)
    split_of = assign_splits(specs, cfg)
    raw_dir = Path(cfg.paths.raw_dir)

    for split in ("train", "val", "test"):
        X, y, g = build_split(cfg, split)
        pos = int(y.sum()) if len(y) else 0
        print(f"{split:5s}: {len(y):6d} windows {tuple(X.shape[1:])} | pos(fall)={pos} "
              f"| sessions={len(set(g))}")

    # Per-session node coverage (how full each node's grid slots are).
    for spec in specs:
        path = raw_dir / spec.csv_file
        if not path.exists():
            continue
        s = load_session(path, cfg)
        cov = " ".join(
            f"n{i + 1}={s.presence[:, i].mean() * 100:4.1f}%" for i in range(cfg.tensor.n_nodes)
        )
        print(f"  {spec.csv_file} [{split_of[spec.csv_file]:5s}] grid={len(s.frames)} | {cov}")
