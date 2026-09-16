# SafeWave-AI · M1 fall-detection training

Trains the M1 (fall detection) model from self-collected CSI data and exports
`m1_wifi_pose.onnx` for the **rp5 inference repo** to load. rp5 already has an
export script, but it ships a *randomly initialized* model and has no training
code — this repo fills that gap.

> **정식 입력은 `(B,5,64,100)` — 5노드 조기융합.** 노드축은 첫 conv의 입력채널이다.
> 과거 per-node `(B,1,64,100)` 전제의 문서·체크포인트·ONNX는 레거시다 (2026-09-16 전환).

## Hard contracts (do NOT change — derived from rp5 db_spec_v2 + inference code)

| Contract | Value | Why |
|----------|-------|-----|
| Input tensor | `(1, 5, 64, 100)` | 5 nodes × `data_raw` 64 ch × 100 frames (1 s @ 100 Hz). db_spec §7. Nodes stack on the **channel axis**, never concatenated onto subcarriers — **never 192** (PulseFi legacy). |
| Subcarriers / frames | 64 × 100 — **immutable** | `config.py` enforces both on load; only `n_nodes` (1–8) is configurable. |
| Node axis | `n_nodes = 5`, conv input channels | Early fusion: the first conv sees all 5 nodes at once. Fusion is **M1's job**, not M5's. |
| Node time alignment | `stream_id` epoch, 100 Hz grid ±5 ms | The epoch prefix of `stream_id` is the rp5 receive time — the **only shared clock**. `ts_ms` is each board's own device clock; aligning nodes on it is forbidden. |
| Normalization | per-frame peak (max=1.0), **per node** | ESP normalizes each packet on-device; training matches it. No absolute-amplitude recovery, no cross-node renormalization. |
| Guard subcarriers | idx `0–5, 32, 59–63` = 0 | Kept in place; shape stays 64 (not reduced to 52). |
| Output | `fall_logit` `(B,1)`, pre-sigmoid | rp5 `_infer_onnx` else-branch reads `output.reshape(-1)[0]` → sigmoid → fall_score. DT-Pose keypoint path is dropped. |
| ONNX | opset 17, input `csi_data` | rp5 ONNX I/O contract. |
| Sigmoid | **rp5 side only — never in the ONNX graph** | The exported graph ends at `fall_logit`. rp5 applies sigmoid in `_infer_onnx`; training applies it inside `BCEWithLogitsLoss`. Bake one into the graph and it is applied **twice** → scores collapse into 0.5–0.73 and the 0.7 threshold stops meaning anything. Also do not add sigmoid options under `export:` in the configs. |

> ⚠️ **미검증 — rp5 쪽 동작 전반** — 위 표의 `Output` / `Sigmoid` 행이 서술하는 **rp5 동작**은
> 2026-08-25 정찰(T1)에서 확인하지 못했다. rp5에 접속이 안 됐다
> ([`docs/archive/RP5_DEVELOP_AUDIT.md`](docs/archive/RP5_DEVELOP_AUDIT.md)).
> 로컬에 남아 있던 2026-06-05 배포 레포 스냅샷의 `_infer_onnx`에는 **sigmoid가 없었다**(부록 A1).
> 이 레포가 내보내는 텐서 계약(`fall_logit`, pre-sigmoid, 그래프에 Sigmoid 없음)은 그대로 유지하되,
> **rp5 쪽에서 sigmoid가 실제로 몇 번 걸리는지는 배포 전에 반드시 직접 확인할 것.**
>
> **5노드 전환으로 미검증 범위가 넓어졌다.** rp5가 `(1,5,64,100)`을 어떻게 만들지 —
> 5노드를 채널축으로 쌓는지, 노드를 무엇으로 시간 정렬하는지, 노드가 빠졌을 때 0을 채우는지 —
> 는 **전부 미확인**이다. 아래 "rp5 sync" 절 참조.

> **Output name nuance:** rp5's nominal contract output name is `keypoints`, but its
> else-branch fetches output **by position** and only special-cases shape `(1,17,2)`.
> We emit `fall_logit` (shape `(B,1)`), which lands in that else-branch. If a future
> rp5 build binds the output *by name*, set `export.output_name: keypoints` in the config.

## Repo layout
```
safewave-ai-ambient-monitoring-pose/
├── README.md                     this file
├── CLAUDE.md                     절대 규칙 + 검증된 사실 (작업 전 먼저 읽을 것)
├── pyproject.toml                package metadata (pip install -e .)
├── requirements.txt              pinned deps (torch, onnx, pandas, sklearn…)
├── .gitignore                    raw CSI / artifacts ignored; labels.csv tracked
├── configs/
│   ├── m1_fall.yaml              본 학습용 (n_nodes 5); hard contracts validated on load
│   └── m1_fall_pretrain.yaml     레거시 단일노드 사전학습용
├── data/
│   ├── README.md                 labels 스키마 + raw CSV 포맷 + 5노드 정렬 방식
│   ├── labels.csv                session→label map (TRACKED) — 수집 후 교체 필요
│   └── raw/                      collection CSVs csi_*.csv (git-ignored)
├── docs/
│   ├── COLLECTION_5NODE.md       5노드 현장 체크리스트 (현행)
│   ├── REFACTOR_5NODE.md         5노드 전환 정리 + 사람이 결정할 것
│   └── archive/                  2026-08-25 시점 기록 (현재 지침 아님)
├── scripts/
│   └── README.md                 operational entry points
└── src/
    └── m1_fall/
        ├── __init__.py
        ├── config.py             typed config + contract validation (n_nodes 포함)
        ├── dataset.py            labels + CSV → (N,5,64,100) windows + labels
        ├── model.py              CNN-GRU (conv1 in_ch = n_nodes) → fall_logit (B,1)
        ├── train.py              train loop
        └── export.py             torch → ONNX opset 17 + parity check
runs/                             checkpoints + exported onnx (git-ignored)
runs/legacy_singlenode/           단일노드 .pt/.onnx — 5노드에선 로드 불가
```

## Quickstart

로컬 파이썬은 레포 안 `.venv` (3.11, torch 2.14 CPU). 시스템 3.14는 numpy DLL이 깨져 있다.

```powershell
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m m1_fall.model                          # (4,5,64,100) -> (4,1)
.venv\Scripts\python.exe -m m1_fall.dataset configs/m1_fall.yaml   # 창 수 + 노드 커버리지
.venv\Scripts\python.exe -m m1_fall.train --config configs/m1_fall.yaml
```

## Windowing & labeling (finalized — see `configs/m1_fall.yaml`)
- **Window** = 100 frames × **5 nodes** → `(5, 64, 100)`. 노드마다 따로가 아니라 한 덩어리.
- **Alignment**: 세션 전체를 `stream_id` epoch 기준 100 Hz 그리드로 깔고, 노드별 프레임을
  최근접 슬롯(±5 ms)에 매핑한다. 결측 노드 슬롯은 0으로 패딩.
- **stride** 25 frames (75% overlap); **fall_stride** 5 frames densifies sampling near falls.
- **Fall window label**: positive if ≥ `fall_pos_frac` (0.5) of its slots fall within
  `±fall_halfwidth_ms` (1000 ms) of any `fall_ts_ms`. NORMAL sessions are all 0.
  `fall_ts_ms`는 **epoch 도메인**이 기본 (레거시 `ts_ms` 도메인은 자동 판별·변환 — `data/README.md`).
- **Split** is by session (file), so no session leaks across train/val/test.
- **Imbalance**: `weighted_sampler` by default (or `focal_loss`).

## ⚠️ rp5 sync requirement — **rp5 `develop` 직접 확인 전까지 전부 미확정**

> ⚠️ **미검증.** 2026-08-25 정찰(T1)에서 rp5에 접속하지 못했고
> ([`docs/archive/RP5_DEVELOP_AUDIT.md`](docs/archive/RP5_DEVELOP_AUDIT.md)),
> **5노드 전환 이후로는 한 번도 확인하지 않았다.**
> 아래는 "확정된 작업 목록"이 아니라 **확인해야 할 질문 목록**이다.
> 어느 쪽이든 이 레포의 텐서 계약 `(B,5,64,100)`은 바뀌지 않는다.

rp5를 이 모델로 돌리려면 최소한 다음이 맞아떨어져야 한다 — **각 항목은 미검증 질문이다**:

1. `ai/experts/m1_wifi_pose.py._preprocess` — 입력을 `(1,5,64,100)`으로 만드는가?
   (과거 스냅샷은 `192*100` 하드코딩, `CLAUDE.md` 함정 표는 "ONNX에서 shape을 직접 읽는다"고
   적고 있어 **서로 모순**이다. 직접 읽어서 판정할 것.)
2. `scripts/export_m1_wifi_pose_onnx.py` — 더미 입력 `(1,5,64,100)` + 이 레포의 학습된
   `state_dict` 로드. (이 레포 `export.py`가 `runs/rp5_patch/`에 대응 번들을 만든다.)
3. `ai/main.py`의 롤링 버퍼 — **5노드를 무엇으로 시간 정렬하는가?**
   이 레포는 `stream_id` epoch로 정렬한다. rp5가 `ts_ms`로 정렬한다면 노드가 서로 어긋난
   입력이 들어가고, 학습 분포와 달라진다. **가장 위험한 미확인 항목.**
4. 노드가 빠진 시점에 rp5가 **0으로 패딩**하는가? 이 레포 로더는 0으로 채운다.
   rp5가 직전 프레임을 복제하거나 추론을 건너뛰면 학습/추론 분포가 갈린다.

하나라도 어긋나면 **shape 불일치로 추론이 실패하거나(그나마 다행), 조용히 틀린 입력으로
돌아간다(더 나쁨).** 배포 전 `develop`을 직접 읽어 4개를 전부 확인하라.

## Branches
`develop` (베이스라인) → `feature/m1-5node` (5노드 전환) → `chore/refactor-5node` (문서·정합).
이 레포는 로컬 git이며 **원격이 없다** — GitHub `composedly13/...pose`와 히스토리가 무관하다
(`docs/REFACTOR_5NODE.md` 참조).
