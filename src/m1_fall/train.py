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

from .config import Config, PretrainCfg, load_config
from .dataset import make_dataloaders


def _torch():
    import torch  # local import keeps the module importable without torch
    return torch


# ── pretrained warm start ────────────────────────────────────────────────────

BACKBONE_ATTR = "convs"          # M1FallNet.convs — the Conv2d stack


def _load_blob(ckpt_path: str):
    """torch.load across versions (weights_only default flipped in torch 2.6)."""
    torch = _torch()
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu")


def load_pretrained(model, ckpt_path: str, mode: str = "full") -> dict:
    """Restore weights from `ckpt_path` into `model`. ALWAYS strict=True.

    strict=False would happily accept a checkpoint whose keys no longer match and
    leave the model half-random, which reads as "pretraining didn't help" instead
    of as the bug it is. Any mismatch must be loud. Returns a dict of load stats.
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise RuntimeError(
            f"[pretrain] checkpoint not found: {ckpt_path} "
            "(runs/ is git-ignored — copy the .pt in, or drop the pretrain block / --init-ckpt)"
        )

    blob = _load_blob(str(path))
    if isinstance(blob, dict) and "state_dict" in blob:
        state_dict, meta = blob["state_dict"], blob
    else:
        state_dict, meta = blob, {}

    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        raise RuntimeError(f"[pretrain] {ckpt_path} holds no state_dict keys — nothing to load")

    total = len(state_dict)
    if mode == "backbone":
        prefix = BACKBONE_ATTR + "."
        sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if not sub:
            raise RuntimeError(
                f"[pretrain] {ckpt_path} has 0 '{prefix}*' keys — cannot load a backbone from it"
            )
        getattr(model, BACKBONE_ATTR).load_state_dict(sub, strict=True)
        loaded = len(sub)
    else:
        model.load_state_dict(state_dict, strict=True)
        loaded = total

    if loaded == 0:  # unreachable via strict=True, kept as an explicit floor
        raise RuntimeError(f"[pretrain] 0 keys loaded from {ckpt_path}")

    metrics = meta.get("metrics") or {}
    parts = [f"[pretrain] {ckpt_path}", f"keys {loaded}/{total}"]
    if mode != "full":
        parts[-1] += f" ({mode})"
    if meta.get("epoch") is not None:
        parts.append(f"epoch {meta['epoch']}")
    if isinstance(metrics, dict) and metrics.get("pr_auc") is not None:
        parts.append(f"pr_auc {float(metrics['pr_auc']):.4f}")
    print(" | ".join(parts))

    return {"loaded": loaded, "total": total, "mode": mode,
            "epoch": meta.get("epoch"), "metrics": metrics if isinstance(metrics, dict) else {}}


def _backbone_params(model):
    return list(getattr(model, BACKBONE_ATTR).parameters())


def set_backbone_trainable(model, trainable: bool) -> None:
    for p in _backbone_params(model):
        p.requires_grad = trainable


def build_optimizer(model, cfg: Config, backbone_lr_mult: float = 1.0, freeze_backbone: bool = False):
    """Adam with an explicit backbone group so its lr can differ from the head's.

    While frozen the backbone group is omitted entirely (not merely zero-lr), so the
    printed group list is an honest picture of what is actually being updated.
    """
    torch = _torch()
    backbone = _backbone_params(model)
    backbone_ids = {id(p) for p in backbone}
    head = [p for p in model.parameters() if id(p) not in backbone_ids]

    lr = float(cfg.train.lr)
    groups = []
    if not freeze_backbone:
        groups.append({"params": backbone, "lr": lr * float(backbone_lr_mult), "name": "backbone"})
    groups.append({"params": head, "lr": lr, "name": "head"})

    opt = torch.optim.Adam(groups, lr=lr, weight_decay=cfg.train.weight_decay)
    _log_param_groups(opt, frozen_n=len(backbone) if freeze_backbone else 0)
    return opt


def _log_param_groups(opt, frozen_n: int = 0) -> None:
    desc = ", ".join(
        f"{g.get('name', f'g{i}')}: {sum(p.numel() for p in g['params']):,}p lr={g['lr']:.1e}"
        for i, g in enumerate(opt.param_groups)
    )
    tail = f" | frozen: backbone {frozen_n} tensors (excluded from optimizer)" if frozen_n else ""
    print(f"[optim] param groups -> {desc}{tail}")


def resolve_pretrain(cfg: Config, init_ckpt: str | None) -> PretrainCfg | None:
    """CLI --init-ckpt wins over the config block; it reuses the block's knobs if present."""
    if init_ckpt:
        base = cfg.pretrain
        return PretrainCfg(
            ckpt=init_ckpt,
            load=base.load if base else "full",
            freeze_epochs=base.freeze_epochs if base else 0,
            backbone_lr_mult=base.backbone_lr_mult if base else 1.0,
        )
    return cfg.pretrain


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

def train(config_path: str, init_ckpt: str | None = None,
          ckpt_out: str | None = None) -> dict:
    cfg = load_config(config_path)
    torch = _torch()
    from .model import build_model

    pre = resolve_pretrain(cfg, init_ckpt)

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

    model = build_model(cfg.model, cfg.tensor.n_channels, cfg.tensor.window_frames,
                        cfg.tensor.n_nodes).to(device)

    # Warm start: load BEFORE the optimizer is built so the freeze decision below
    # and the optimizer's param groups describe the same model.
    pretrain_info = None
    if pre is not None:
        pretrain_info = load_pretrained(model, pre.ckpt, mode=pre.load)
        if init_ckpt:
            print("[pretrain] source: --init-ckpt (CLI overrides config pretrain block)")

    freeze_epochs = int(pre.freeze_epochs) if pre is not None else 0
    backbone_lr_mult = float(pre.backbone_lr_mult) if pre is not None else 1.0
    frozen = freeze_epochs > 0
    if frozen:
        set_backbone_trainable(model, False)
        print(f"[pretrain] backbone frozen for the first {freeze_epochs} epoch(s) "
              f"(GRU + head only); thaw at epoch {freeze_epochs + 1}")

    opt = build_optimizer(model, cfg, backbone_lr_mult=backbone_lr_mult, freeze_backbone=frozen)
    loss_fn = make_loss_fn(cfg, ytr, device)

    out_dir = Path(cfg.paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 기본은 best.pt. --ckpt-out 을 주면 그 경로에 저장한다 — 기존 체크포인트를
    # 덮어쓰지 않고 다른 조건의 학습을 나란히 남길 때 쓴다.
    ckpt_path = Path(ckpt_out) if ckpt_out else (out_dir / "best.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    # 요약도 체크포인트 이름을 따라간다. 안 그러면 train_summary.json 이 덮인다.
    summary_path = (ckpt_path.with_name(ckpt_path.stem + "_summary.json")
                    if ckpt_out else out_dir / "train_summary.json")
    print(f"[out] checkpoint -> {ckpt_path}")
    print(f"[out] summary    -> {summary_path}")

    best = float("-inf")        # M1 MAXIMIZES fall_recall (opposite of M2's MAE minimization)
    best_pr = float("-inf")     # tie-break: prefer higher PR-AUC at equal recall
    best_epoch = -1
    patience = 0
    history = []

    for epoch in range(1, cfg.train.epochs + 1):
        if frozen and epoch > freeze_epochs:
            # Thaw: re-grant grads, then REBUILD the optimizer so the backbone gets its
            # own param group at the reduced lr. Adam state for the head is dropped,
            # which is the intended clean break at the fine-tuning boundary.
            set_backbone_trainable(model, True)
            frozen = False
            print(f"[pretrain] epoch {epoch}: 백본 해동, "
                  f"backbone_lr={cfg.train.lr * backbone_lr_mult:.1e}")
            opt = build_optimizer(model, cfg, backbone_lr_mult=backbone_lr_mult, freeze_backbone=False)

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
    if pretrain_info is not None:
        summary["pretrain"] = {
            "ckpt": pre.ckpt, "load": pre.load, "freeze_epochs": freeze_epochs,
            "backbone_lr_mult": backbone_lr_mult,
            "keys_loaded": pretrain_info["loaded"], "keys_total": pretrain_info["total"],
            "source_epoch": pretrain_info["epoch"],
            "source": "cli" if init_ckpt else "config",
        }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] best ep {best_epoch} {cfg.train.primary_metric}={best:.3f} -> {ckpt_path}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m1_fall.yaml")
    p.add_argument("--init-ckpt", default=None,
                   help="warm start from this checkpoint (overrides the config's pretrain.ckpt)")
    p.add_argument("--ckpt-out", default=None,
                   help="best 체크포인트를 저장할 경로 (기본: {paths.out_dir}/best.pt). "
                        "기존 체크포인트를 덮지 않으려면 지정할 것. 요약 json 도 같은 이름을 따른다.")
    a = p.parse_args()
    train(a.config, init_ckpt=a.init_ckpt, ckpt_out=a.ckpt_out)
