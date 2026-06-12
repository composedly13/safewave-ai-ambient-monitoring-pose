# data/ — labels + raw CSI

## Layout
```
data/
├── labels.csv          # session→label map  (TRACKED in git)
├── raw/                # csi_YYYYMMDD_HHMM.csv collection files  (GIT-IGNORED)
└── windows_cache/      # optional cached (N,1,64,100) windows  (GIT-IGNORED)
```
Drop the raw collection CSVs from the rp5 collector into `data/raw/`.
Only `labels.csv` is committed — raw CSI is large and may be private.

## `labels.csv` schema  (one row per collection session/file)

| column          | required | meaning |
|-----------------|----------|---------|
| `csv_file`      | yes      | filename under `data/raw/` (e.g. `csi_20260102_1010.csv`) |
| `session_label` | yes      | `NORMAL` or `FALL` — the session type |
| `fall_ts_ms`    | FALL only| fall-instant timestamp(s) matching the `ts_ms` column in the raw CSV. Multiple falls → semicolon-separated. Empty for NORMAL. |
| `split`         | no       | force `train` / `val` / `test`. Empty → assigned by session-level auto split (no session leaks across splits). |
| `notes`         | no       | free text |

### How a row becomes window labels
- Fall is a session-level *event in physical space*, so `fall_ts_ms` is **per session, not per node** — every node's stream around that instant is labeled together.
- For each fall instant `t`, the positive interval is `[t - fall_halfwidth_ms, t + fall_halfwidth_ms]` (default ±1000 ms).
- A 100-frame window is labeled **1** when ≥ `fall_pos_frac` (default 0.5) of its frames fall inside any positive interval; otherwise **0**.
- NORMAL sessions are all 0.

See `configs/m1_fall.yaml` for `fall_halfwidth_ms`, `fall_pos_frac`, stride, etc.

## Raw CSV format (from rp5 collector)
Columns: `stream_id, node_id, ts_ms, raw_0..63, resp_0..63, heart_0..63`.
**M1 uses `raw_0..63` only** (resp_*/heart_* belong to other experts).
Loader groups rows by `node_id` and windows each node's stream independently
(M1 = per-node inference; no 5-node early fusion).
