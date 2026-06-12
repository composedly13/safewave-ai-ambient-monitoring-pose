"""Training loop for M1 fall detection.  [STUB — implemented in feature/m1-fall PR]

Planned:
  - build train/val DataLoaders via dataset.make_dataloaders(cfg)
  - BCEWithLogitsLoss (pos_weight or focal per cfg.imbalance)
  - optimize for fall-recall; report PR-AUC / precision / F1 each epoch
  - early stopping on val fall-recall; checkpoint best state_dict to runs/
"""
from __future__ import annotations

import argparse

from .config import load_config


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    raise NotImplementedError(
        "train.py is a scaffold stub — implemented in the feature/m1-fall PR. "
        f"(config loaded OK: primary_metric={cfg.train.primary_metric}, "
        f"imbalance={cfg.imbalance.strategy})"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m1_fall.yaml")
    train(p.parse_args().config)
