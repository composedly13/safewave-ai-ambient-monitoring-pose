# scripts/

Operational entry points (thin wrappers over `src/m1_fall/`).

| script | stage | status |
|--------|-------|--------|
| `python -m m1_fall.dataset configs/m1_fall.yaml` | inspect window counts / class balance | ready |
| `python -m m1_fall.model` | model shape + param sanity | ready |
| `python -m m1_fall.train --config configs/m1_fall.yaml` | train | stub (feature/m1-fall) |
| `python -m m1_fall.export --config configs/m1_fall.yaml --ckpt runs/best.pt` | torch → ONNX | stub (feature/m1-fall) |

Raw-CSI collection lives in the **rp5 repo** (100 Hz dual/5-node collector);
drop its `csi_YYYYMMDD_HHMM.csv` output into `data/raw/` here.
