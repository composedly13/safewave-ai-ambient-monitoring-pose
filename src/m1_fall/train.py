"""Training loop for M1 fall detection (CNN-GRU -> single fall logit).

- loss   : BCEWithLogitsLoss (pos_weight auto/explicit) or focal-BCE per cfg.imbalance
- metrics: fall-recall (TP/(TP+FN)) is primary; PR-AUC / precision / F1 reported too
- imbalance: dataset.make_dataloaders honors cfg.imbalance.strategy (weighted sampler);
             focal is handled here on the loss side
- select : early stop on cfg.train.primary_metric (fall_recall, HIGHER better);
           best state_dict checkpointed to {paths.out_dir}/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from .config import Config, load_config
from .dataset import make_dataloaders


def _torch():
    import torch  # local import keeps the module importable without torch
    return torch


# ── loss (binary) ────────────────────────────────────────────────────────────

def _pos_weight(cfg: Config, y: np.ndarray):
    """BCEWithLogits pos_weight: explicit cfg value, else auto neg/pos from train y."""
    torch = _torch()
    if cfg.imbalance.pos_weight is not None:
        w = float(cfg.imbalance.pos_weight)
    else:
        counts = np.bincount(y.astype(int), minlength=2).astype(np.float64)
        neg, pos = counts[0], counts[1]
        w = float(neg / pos) if pos > 0 else 1.0
    return torch.tensor([w], dtype=torch.float32)


def make_loss_fn(cfg: Config, y: np.ndarray, device: str):
    """Return a callable loss_fn(logits (B,1), targets (B,1)) honoring cfg.imbalance."""
    torch = _torch()
    pos_weight = _pos_weight(cfg, y).to(device)

    if cfg.imbalance.strategy == "focal_loss":
        gamma = float(cfg.imbalance.focal_gamma)

        def focal_bce(logits, targets):
            # focal modulation on top of (pos_weighted) BCE — down-weights easy negatives
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction="none", pos_weight=pos_weight
            )
            p = torch.sigmoid(logits)
            p_t = p * targets + (1.0 - p) * (1.0 - targets)
            return (((1.0 - p_t) ** gamma) * bce).mean()

        return focal_bce

    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return lambda logits, targets: bce(logits, targets)


# ── metrics ──────────────────────────────────────────────────────────────────

def evaluate(model, dl, device="cpu") -> Dict[str, float]:
    """Sigmoid + threshold 0.5 -> fall_recall / precision / f1; PR-AUC from scores.

    Empty loader -> all nan.
    """
    torch = _torch()
    from sklearn.metrics import average_precision_score

    model.eval()
    scores, tgts = [], []
    with torch.no_grad():
        for x, y in dl:
            logits = model(x.to(device))                  # (B, 1)
            scores.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
            tgts.append(y.cpu().numpy().reshape(-1))
    if not scores:
        return {"fall_recall": float("nan"), "precision": float("nan"),
                "f1": float("nan"), "pr_auc": float("nan")}

    s = np.concatenate(scores)
    t = np.concatenate(tgts).astype(int)
    pred = (s >= 0.5).astype(int)

    tp = int(np.sum((pred == 1) & (t == 1)))
    fp = int(np.sum((pred == 1) & (t == 0)))
    fn = int(np.sum((pred == 0) & (t == 1)))

    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    if np.isfinite(recall) and np.isfinite(precision) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    # average_precision_score needs both classes present
    pr_auc = float(average_precision_score(t, s)) if len(np.unique(t)) == 2 else float("nan")

    return {"fall_recall": recall, "precision": precision, "f1": f1, "pr_auc": pr_auc}


# ── train ────────────────────────────────────────────────────────────────────

def train(config_path: str) -> dict:
    cfg = load_config(config_path)
    torch = _torch()
    from .model import build_model

    train_dl, val_dl = make_dataloaders(cfg)
    ytr = np.asarray(train_dl.dataset.y)
    if not len(ytr):
        raise RuntimeError("train split is empty — check data/raw, labels.csv, and split fracs")
    pos = int(ytr.sum())
    print(f"[data] train={len(ytr)} val={len(val_dl.dataset)} windows | "
          f"train pos(fall)={pos} neg={len(ytr) - pos} | imbalance={cfg.imbalance.strategy}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[device] cuda -> {torch.cuda.get_device_name(0)}")
    else:
        print("[device] cpu (cuda not available)")

    model = build_model(cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loss_fn = make_loss_fn(cfg, ytr, device)

    out_dir = Path(cfg.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"

    best = float("-inf")        # M1 MAXIMIZES fall_recall (opposite of M2's MAE minimization)
    best_pr = float("-inf")     # tie-break: prefer higher PR-AUC at equal recall
    best_epoch = -1
    patience = 0
    history = []

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_dl:
            x = x.to(device)
            target = y.to(device).float().unsqueeze(1)    # (B,) -> (B,1) to match logits
            opt.zero_grad()
            loss = loss_fn(model(x), target)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(x)
            n += len(x)
        train_loss = running / max(n, 1)

        m = evaluate(model, val_dl, device)
        recall, pr_auc = m["fall_recall"], m["pr_auc"]
        # selection: cfg.train.primary_metric (HIGHER better); -train_loss when no val set
        primary = m.get(cfg.train.primary_metric, float("nan"))
        score = primary if np.isfinite(primary) else -train_loss
        tie_key = "fall_recall" if cfg.train.primary_metric == "pr_auc" else "pr_auc"
        tie_val = m.get(tie_key, float("nan"))
        tie = tie_val if np.isfinite(tie_val) else float("-inf")
        history.append({"epoch": epoch, "train_loss": train_loss, **m})
        print(f"[ep {epoch:3d}] train_loss={train_loss:.4f} | val recall={recall:.3f} "
              f"prec={m['precision']:.3f} f1={m['f1']:.3f} pr_auc={pr_auc:.3f}")

        improved = (score > best + 1e-6) or (abs(score - best) <= 1e-6 and tie > best_pr + 1e-6)
        if improved:
            best = score
            best_pr = tie
            best_epoch = epoch
            patience = 0
            torch.save({"state_dict": model.state_dict(), "cfg_path": config_path,
                        "epoch": epoch, "metrics": history[-1]}, ckpt_path)
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"[early-stop] no improvement in {patience} epochs (best ep {best_epoch})")
                break

    summary = {"best_epoch": best_epoch, "best_score": best, "primary_metric": cfg.train.primary_metric,
               "checkpoint": str(ckpt_path), "history": history}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] best ep {best_epoch} {cfg.train.primary_metric}={best:.3f} -> {ckpt_path}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m1_fall.yaml")
    train(p.parse_args().config)
