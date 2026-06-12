# SafeWave-AI · M1 fall-detection training

Trains the M1 (fall detection) model from self-collected CSI data and exports
`m1_wifi_pose.onnx` for the **rp5 inference repo** to load. rp5 already has an
export script, but it ships a *randomly initialized* model and has no training
code — this repo fills that gap.

## Hard contracts (do NOT change — derived from rp5 db_spec_v2 + inference code)

| Contract | Value | Why |
|----------|-------|-----|
| Input tensor | `(1, 1, 64, 100)` | `data_raw` 64 ch × 100 frames (1 s @ 100 Hz). db_spec §7. **Never 192** (PulseFi legacy). |
| Normalization | per-frame peak (max=1.0) | ESP normalizes each packet on-device; training matches it. No absolute-amplitude recovery. |
| Guard subcarriers | idx `0–5, 32, 59–63` = 0 | Kept in place; shape stays 64 (not reduced to 52). |
| Inference unit | 1 node × 100 frames | M1 is per-node. **No 5-node early fusion** (fusion = M5/Qwen in rp5). |
| Output | `fall_logit` `(B,1)`, pre-sigmoid | rp5 `_infer_onnx` else-branch reads `output.reshape(-1)[0]` → sigmoid → fall_score. DT-Pose keypoint path is dropped. |
| ONNX | opset 17, input `csi_data` | rp5 ONNX I/O contract. |

> **Output name nuance:** rp5's nominal contract output name is `keypoints`, but its
> else-branch fetches output **by position** and only special-cases shape `(1,17,2)`.
> We emit `fall_logit` (shape `(B,1)`), which lands in that else-branch. If a future
> rp5 build binds the output *by name*, set `export.output_name: keypoints` in the config.

## Repo layout
```
safewave-ai-ambient-monitoring-M1/
├── README.md                     this file
├── pyproject.toml                package metadata (pip install -e .)
├── requirements.txt              pinned deps (torch, onnx, pandas, sklearn…)
├── .gitignore                    raw CSI / artifacts ignored; labels.csv tracked
├── configs/
│   └── m1_fall.yaml              all knobs; hard contracts validated on load
├── data/
│   ├── README.md                 labels.csv schema + raw CSV format
│   ├── labels.csv                session→label map (TRACKED)
│   └── raw/                      collection CSVs csi_*.csv (git-ignored)
├── scripts/
│   └── README.md                 operational entry points
└── src/
    └── m1_fall/
        ├── __init__.py
        ├── config.py             typed config + contract validation
        ├── dataset.py            labels.csv + CSV → (N,1,64,100) windows + labels
        ├── model.py              CNN-GRU → fall_logit (B,1)
        ├── train.py              train loop            [stub → feature/m1-fall]
        └── export.py             torch → ONNX opset 17 [stub → feature/m1-fall]
runs/                             checkpoints + exported onnx (git-ignored)
```

## Quickstart
```bash
pip install -e .                                   # installs deps + m1_fall package
# put collection CSVs in data/raw/, fill data/labels.csv, then:
python -m m1_fall.model                            # model shape + param sanity
python -m m1_fall.dataset configs/m1_fall.yaml     # per-split window counts + class balance
```

## Windowing & labeling (finalized — see `configs/m1_fall.yaml`)
- **Window** = 100 frames (fixed by tensor shape), one per node.
- **stride** 25 frames (75% overlap); **fall_stride** 5 frames densifies sampling near falls.
- **Fall window label**: positive if ≥ `fall_pos_frac` (0.5) of its frames fall within
  `±fall_halfwidth_ms` (1000 ms) of any `fall_ts_ms`. NORMAL sessions are all 0.
- **Split** is by session (file), so no session leaks across train/val/test.
- **Imbalance**: `weighted_sampler` by default (or `focal_loss`).

## ⚠️ rp5 sync requirement (must land together)
Switching rp5 from the 192-channel legacy to this 64-channel model requires **both**
edits in the inference repo, in the same change:

1. `ai/experts/m1_wifi_pose.py._preprocess` — `192 → 64`
2. `scripts/export_m1_wifi_pose_onnx.py` — `dummy_input (1,1,192,100) → (1,1,64,100)`
   *(and load this repo's trained `state_dict` instead of the random `DTPoseModel`)*

Ship one without the other → **ONNX shape mismatch → inference fails**. The export
patch produced by this repo's `export.py` (next PR) targets exactly these two spots.

## Branches
`chore/scaffold` → `develop` → `feature/m1-fall`. PR per stage.
This is the `chore/scaffold` deliverable: structure + config + data-loader interface.
