# data/ — labels + raw CSI

> **M1의 정식 입력은 `(B,5,64,100)` — 5노드 조기융합이다.**
> 한 창(window)은 노드 하나가 아니라 **5노드가 시간 정렬된 한 덩어리**다.

## Layout
```
data/
├── labels.csv                # session→label map  (TRACKED in git) — 학습이 읽는 라벨
├── labels_self.csv           # 자체 수집 라벨 (5노드) — verify_sync_v2.py 출력을 붙인다
├── labels_self_legacy.csv    # 6/11 단일노드 시절 NORMAL 8세션 (보관용, 학습에 안 씀)
├── labels_csibench.csv       # 공개 데이터 라벨 원본 (단일노드 시절)
├── node_layout.csv           # 노드 좌표 (수집 기록)
├── session_log.csv           # 세션 기록 (수집 기록)
├── raw/                      # csi_YYYYMMDD_HHMM.csv collection files  (GIT-IGNORED)
└── windows_cache/            # optional cached (N,5,64,100) windows  (GIT-IGNORED)
```
Drop the raw collection CSVs from the rp5 collector into `data/raw/`.
Only the label CSVs are committed — raw CSI is large and may be private.

> ⚠️ **현재 `labels.csv`는 csibench 494행(단일노드, `stream_id` 없음)이라 5노드 로더에
> 들어가지 못한다.** 5노드 수집이 끝나면 `labels_self.csv`의 내용으로 교체하거나,
> `configs/m1_fall.yaml`의 `paths.labels_csv`를 그쪽으로 돌려야 학습이 돈다.
> 수집 전에는 바꿀 수 없으므로(파일이 비어 있음) **수집 직후 첫 할 일**로 남겨둔다.

## `labels.csv` schema  (one row per collection session/file)

| column          | required | meaning |
|-----------------|----------|---------|
| `csv_file`      | yes      | filename under `data/raw/` (e.g. `csi_20260102_1010.csv`) |
| `session_label` | yes      | `NORMAL` or `FALL` — the session type |
| `fall_ts_ms`    | FALL only| fall-instant timestamp(s). **기본은 epoch ms 도메인** (아래 참조). Multiple falls → semicolon-separated. Empty for NORMAL. |
| `split`         | no       | force `train` / `val` / `test`. Empty → assigned by session-level auto split (no session leaks across splits). |
| `notes`         | no       | free text |

### `fall_ts_ms`의 시간 도메인 — epoch 기본, 레거시 자동 변환

낙상 마킹(`scripts/mark_falls.py`)은 **epoch ms**(Unix 시각, 약 1.79e12)를 찍는다.
반면 예전 `verify_sync_v2.py`는 이것을 보드 장치 시계 `ts_ms` 도메인
(32비트 랩이라 항상 4.29e9 미만)으로 변환해 기록했다.

로더는 값의 크기로 도메인을 판별한다:

| `fall_ts_ms` 값 | 해석 | 처리 |
|---|---|---|
| ≥ 2³² (≈4.29e9) | **epoch 도메인** (현행, `mark_falls.py`) | 그대로 사용 |
| < 2³² | 레거시 `ts_ms` 도메인 (`verify_sync_v2.py` 출력) | 파일별 오프셋 `median(epoch − ts_ms)`로 epoch 변환 |

두 도메인은 크기가 약 400배 차이라 오판 여지가 없다. **새 라벨은 epoch로 적어라.**

### How a row becomes window labels
- Fall is a session-level *event in physical space*, so `fall_ts_ms` is **per session, not per node** — 5노드가 그 순간을 함께 본 것이므로 한 창 전체에 같은 라벨이 붙는다.
- For each fall instant `t`, the positive interval is `[t - fall_halfwidth_ms, t + fall_halfwidth_ms]` (default ±1000 ms), evaluated **in the epoch domain**.
- A 100-frame window is labeled **1** when ≥ `fall_pos_frac` (default 0.5) of its grid slots fall inside any positive interval; otherwise **0**.
- NORMAL sessions are all 0.

See `configs/m1_fall.yaml` for `fall_halfwidth_ms`, `fall_pos_frac`, stride, etc.

## Raw CSV format (from rp5 collector)
Columns: `stream_id, node_id, ts_ms, raw_0..63, resp_0..63, heart_0..63`.
**M1 uses `raw_0..63` only** (resp_*/heart_* belong to other experts).

### 로더가 5노드를 정렬하는 방식 (`src/m1_fall/dataset.py`)

```
행마다 epoch = stream_id 앞부분 ("1781165460004-0" → 1781165460004)
  → 세션 전체를 공통 100 Hz epoch 그리드로 (10 ms 간격)
  → 노드별 프레임을 최근접 슬롯(±5 ms)에 매핑, 한 슬롯에 둘이 겹치면 더 가까운 쪽
  → 결측 노드 슬롯은 0으로 패딩              → (T_grid, 5, 64)
  → 노드·프레임별 per-frame peak 정규화 + guard 서브캐리어 0
  → 창 100프레임 (stride 25, 낙상 근처 dense stride 5)
  → 창 텐서 (5, 64, 100)                      → 배치 시 (B, 5, 64, 100)
```

**`stream_id`의 epoch가 유일한 공통 시계다.** `ts_ms`는 보드마다 다른 장치 시계라
노드 간 정렬에 쓰면 어긋난다 — `ts_ms`로 노드를 정렬하지 마라.

노드가 빠진 슬롯은 0으로 채워지므로, 한 노드가 통째로 죽어도 학습은 돈다.
`python -m m1_fall.dataset configs/m1_fall.yaml`이 세션별 **노드 커버리지(%)**를
출력하니, 이 값이 낮으면 동기·수집 문제를 먼저 의심해라.

Split 단위는 **세션(파일)**이라 같은 세션이 train/val에 걸치지 않는다.
