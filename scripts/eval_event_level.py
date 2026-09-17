#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_event_level.py — M1 을 창 단위가 아니라 시스템(사건) 단위로 잰다.

학습은 하지 않는다. 체크포인트·라벨·yaml·원본 CSV·마커는 읽기만 한다.

시스템 모사
  검증 세션마다 100 Hz 그리드 위에서 창 시작 0, 20, 40, ... 프레임 = **200 ms 간격(5 Hz) 추론**.
  추론 시각 = 창 마지막 슬롯 시각. 창·라벨은 `m1_fall.dataset` 의 load_session /
  fall_ts_to_epoch / fall_frame_mask 와 같은 규칙으로 만든다 (오프셋 0 은 make_windows 와
  바이트 단위로 같은지 확인한다). yaml 은 건드리지 않는다.

정의 (측정 전 고정)
  낙상 사건       data/MAIN.markers.csv 의 kind=fall 중 검증 세션 epoch 범위에 드는 것.
                  라벨 fall_ts_ms(= 마커 − 반응지연 300 ms, 레거시 도메인)와 1:1 로 맞춘다.
  낙상 j 의 양성 창 P_j
                  데이터셋 규칙으로 라벨 1 인 창 중, 낙상 j 의 ±halfwidth 구간이 창 프레임의
                  pos_frac 이상을 덮는 창. 라벨 1 인데 어느 낙상도 단독으로 못 덮으면
                  덮는 비율이 가장 큰 낙상에 귀속 (그런 창 수를 보고).
  발화 fire(i)    score ≥ THRESHOLD (게이트 적용 시: 게이트 통과한 추론만)
  게이트          §9 규칙 — 세 노드 모두 창 끝 GATE_TAIL 프레임에 실프레임이 있어야 추론.
                  생략된 추론은 발화 없음. K-of-N 의 최근 N 회에도 "발화 아님"으로 들어간다.
  경보 alarm(i)   집계 없음: fire(i) / K-of-N: 최근 N 회(i-N+1..i) 발화 수 ≥ K.
                  세션 첫 N-1 회는 있는 만큼만 센다.
  사건 감지       낙상 j 에 대해 alarm(i) 가 참인 i ∈ P_j 가 하나라도 있으면 감지.
                  (집계 없음이면 "양성 창 중 하나라도 임계를 넘으면 감지" 와 같다.)
                  P_j 가 비어 있으면 미감지로 센다.
  오탐 건수       빈 방 / 활동 검증 세션에서 alarm 이 거짓→참 으로 바뀐 횟수 (연속 참 = 1건).
                  참인 추론 수도 같이 낸다.
  관측 시간       추론 횟수 × 200 ms.  FA/h = 오탐 건수 / 관측 시간(h).

판정 (측정 전 고정) — 오프셋 0, 판정 시각 연장 없음 기준
  후보 = K ∈ 1..5 × 게이트 {미적용, 적용}.
  주 판정 A: 빈 방 FA/h < 1.0 그리고 활동 FA/h < 1.0 인 후보 중 사건 재현율 최대.
  참고 B  : 빈 방 FA/h < 1.0 만 요구.
  동률    : 오탐 건수 합이 적은 쪽 → 게이트 적용 → K 작은 쪽.
  후보가 없으면: 빈 방 FA/h 최솟값(동률이면 재현율 높은 쪽)과 그때 재현율, 활동도 같은 방식으로 보고.

보조 (판정에 쓰지 않음)
  - 추론 시작 오프셋 PHASES (0/40/80/120/160 ms) 별 결과 범위 — 200 ms 격자 위치에 대한 민감도
  - 판정 시각을 P_j 마지막 창 뒤 N-1 회(0.8 s)까지 연장한 사건 재현율
  - bad 마커가 붙은 낙상을 뺀 사건 재현율 (bad 마커가 낙상 마커 뒤 1 s 안에 있으면 그 낙상에 붙은 것으로 본다)
  - 엄격 재현율: 경보를 켠 최근 N 회 발화 중 낙상 j 자기 양성 창의 발화가 1개 이상일 때만 감지.
    (첫 측정에서 1340 #4 가 자기 창이 아닌 직전 발화로 K=1 감지된 것을 보고 추가한 참고 지표 — 판정에 쓰지 않음)

사용
    $env:PYTHONPATH="src"
    .venv\\Scripts\\python.exe scripts\\eval_event_level.py --md <경로>
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

THRESHOLD = 0.80
HOP_FRAMES = 20                  # 200 ms @ 100 Hz
N_AGG = 5                        # K-of-N 의 N (= 1 초)
K_LIST = [1, 2, 3, 4, 5]
GATE_TAIL = 1                    # §9 권고 N
FA_TARGET = 1.0                  # FA/h 기준
PHASES = [0, 4, 8, 12, 16]       # 보조: 추론 시작 오프셋 (프레임)
MARKER_LAG_MS = 300              # verify_sync_v2 기본 --lag-ms (라벨 = 마커 − 이것)
EXPECTED_VAL_WINDOWS = 924
EMPTY_TAG, ACTIVE_TAG = "normal_empty", "normal_active"
AGGS = ["none"] + [f"K{k}" for k in K_LIST]


def tag_of(csv_file: str) -> str:
    parts = Path(csv_file).stem.split("_")
    return "_".join(parts[3:]) if len(parts) > 3 else "untagged"


def load_checkpoint(model, ckpt_path: Path):
    import torch
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model.load_state_dict(state, strict=True)    # strict=False 금지 (CLAUDE.md 규칙 6)
    print(f"[ckpt] {ckpt_path} · strict=True 로드 성공 · 키 {len(state)}개 · epoch {blob.get('epoch')}")
    return model


def infer(model, X: np.ndarray, bs: int) -> np.ndarray:
    import torch
    out = np.empty(len(X), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out[i:i + bs] = torch.sigmoid(model(torch.from_numpy(X[i:i + bs]))).numpy().reshape(-1)
    return out


@dataclass
class Stream:
    """한 세션 · 한 오프셋의 200 ms 추론 스트림."""
    csv_file: str
    tag: str
    starts: np.ndarray            # (M,) 창 시작 프레임
    end_ms: np.ndarray            # (M,) 추론 시각 = 창 마지막 슬롯 epoch
    label: np.ndarray             # (M,) 데이터셋 규칙 라벨
    owner: List[List[int]]        # (M,) 귀속 낙상 인덱스 (세션 내)
    gate: np.ndarray              # (M,) 게이트 통과
    score: np.ndarray             # (M,)
    falls: List[dict] = field(default_factory=list)   # 세션 내 낙상 정보
    n_orphan_pos: int = 0         # 단독으로 pos_frac 을 못 넘어 최대 비율 낙상에 귀속된 창


def attribute(grid_epoch_ms: np.ndarray, starts, lab_epochs: List[int], cfg):
    """창 시작들 → (데이터셋 규칙 라벨, 창별 귀속 낙상 인덱스, 단독 귀속 실패 창 수)."""
    from m1_fall.dataset import fall_frame_mask

    W = cfg.tensor.window_frames
    hw, pf = cfg.label.fall_halfwidth_ms, cfg.label.fall_pos_frac
    label = np.zeros(len(starts), dtype=np.int64)
    owner: List[List[int]] = [[] for _ in starts]
    n_orphan = 0
    if not lab_epochs:
        return label, owner, n_orphan
    union = fall_frame_mask(grid_epoch_ms, lab_epochs, hw)
    per = [fall_frame_mask(grid_epoch_ms, [t], hw) for t in lab_epochs]
    for i, s in enumerate(starts):
        if float(union[s:s + W].mean()) < pf:
            continue
        label[i] = 1
        fr = [float(m[s:s + W].mean()) for m in per]
        own = [j for j, v in enumerate(fr) if v >= pf]
        if not own:
            own = [int(np.argmax(fr))]
            n_orphan += 1
        owner[i] = own
    return label, owner, n_orphan


def build_stream(session, spec, cfg, phase: int, markers: List[int], bads: List[int]):
    """(창 텐서, Stream 틀) — 점수는 나중에 채운다."""
    from m1_fall.dataset import fall_frame_mask, fall_ts_to_epoch

    W = cfg.tensor.window_frames
    T = len(session.frames)
    starts = np.arange(phase, T - W + 1, HOP_FRAMES)
    X = np.stack([session.frames[s:s + W].transpose(1, 2, 0) for s in starts]).astype(np.float32)
    end_ms = session.grid_epoch_ms[starts + W - 1]
    gate = np.array([session.presence[s + W - GATE_TAIL:s + W].all() for s in starts])

    lab_epochs = sorted(fall_ts_to_epoch(spec.fall_ts, session.offset_ms)) if spec.fall_ts else []
    lo, hi = int(session.grid_epoch_ms[0]), int(session.grid_epoch_ms[-1])
    mine = sorted(t for t in markers if lo <= t <= hi)
    if len(mine) != len(lab_epochs):
        sys.exit(f"{spec.csv_file}: 마커 {len(mine)}건 ≠ 라벨 {len(lab_epochs)}건")

    label, owner, n_orphan = attribute(session.grid_epoch_ms, starts, lab_epochs, cfg)

    falls = []
    for j, (mk, le) in enumerate(zip(mine, lab_epochs)):
        falls.append({"idx": j, "marker_ms": mk, "label_epoch_ms": le, "lag_ms": mk - le,
                      "bad": any(0 <= b - mk <= 1000 for b in bads)})
    st = Stream(csv_file=spec.csv_file, tag=tag_of(spec.csv_file), starts=starts, end_ms=end_ms,
                label=label, owner=owner, gate=gate, score=np.empty(0), falls=falls, n_orphan_pos=n_orphan)
    return X, st


def alarm_of(st: Stream, gate_on: bool, agg: str) -> np.ndarray:
    fire = st.score >= THRESHOLD
    if gate_on:
        fire = fire & st.gate
    if agg == "none":
        return fire
    k = int(agg[1:])
    cnt = np.convolve(fire.astype(np.int64), np.ones(N_AGG, dtype=np.int64))[:len(fire)]
    return cnt >= k


def evaluate(streams: List[Stream], gate_on: bool, agg: str) -> dict:
    fall_n = fall_k = fall_k_ext = fall_k_strict = nobad_n = nobad_k = empty_pos = 0
    per_fall = []
    fa = {EMPTY_TAG: [0, 0, 0], ACTIVE_TAG: [0, 0, 0]}     # [건수, 참인 추론, 추론 수]
    fire_cnt = {EMPTY_TAG: 0, ACTIVE_TAG: 0}
    pos_n = pos_fire = 0
    for st in streams:
        alarm = alarm_of(st, gate_on, agg)
        fire = (st.score >= THRESHOLD) & (st.gate if gate_on else True)
        if st.tag in fa:
            rise = alarm & ~np.r_[False, alarm[:-1]]
            fa[st.tag][0] += int(rise.sum())
            fa[st.tag][1] += int(alarm.sum())
            fa[st.tag][2] += len(alarm)
            fire_cnt[st.tag] += int(fire.sum())
        pos = st.label == 1
        pos_n += int(pos.sum())
        pos_fire += int((fire & pos).sum())
        for f in st.falls:
            P = np.array([i for i, o in enumerate(st.owner) if f["idx"] in o], dtype=np.int64)
            det = bool(len(P) and alarm[P].any())
            # 엄격: 경보를 켠 최근 N 회 발화 중 낙상 j 자기 양성 창의 발화가 1개 이상
            pset = set(P.tolist())
            det_strict = any(alarm[i] and any(fire[q] for q in range(max(0, i - N_AGG + 1), i + 1) if q in pset)
                             for i in P) if len(P) else False
            fall_k_strict += det_strict
            if len(P):
                ext = np.arange(P.min(), min(P.max() + N_AGG, len(alarm)))
                det_ext = bool(alarm[ext].any())
            else:
                det_ext = False
                empty_pos += 1
            fall_n += 1
            fall_k += det
            fall_k_ext += det_ext
            if not f["bad"]:
                nobad_n += 1
                nobad_k += det
            per_fall.append({"csv_file": st.csv_file, "fall_idx": f["idx"], "detected": det,
                             "detected_ext": det_ext, "detected_strict": det_strict, "n_pos": len(P)})
    out = {"gate": gate_on, "agg": agg, "fall_n": fall_n, "fall_k": fall_k, "fall_k_ext": fall_k_ext,
           "fall_k_strict": fall_k_strict,
           "nobad_n": nobad_n, "nobad_k": nobad_k, "falls_without_pos": empty_pos,
           "pos_n": pos_n, "pos_fire": pos_fire, "per_fall": per_fall}
    for tag, key in ((EMPTY_TAG, "empty"), (ACTIVE_TAG, "active")):
        ev, on, n = fa[tag]
        hours = n * HOP_FRAMES * 10 / 1000 / 3600
        out.update({f"{key}_events": ev, f"{key}_alarm_infer": on, f"{key}_infer": n,
                    f"{key}_fire_infer": fire_cnt[tag], f"{key}_hours": hours,
                    f"{key}_fah": ev / hours if hours else float("nan")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m1_fall.yaml")
    ap.add_argument("--ckpt", default="runs/m1_base_3node_20260916.pt")
    ap.add_argument("--markers", default="data/MAIN.markers.csv")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--md", default=None, help="문서용 markdown 표를 이 경로에 쓴다")
    a = ap.parse_args()

    from m1_fall.config import load_config
    from m1_fall.dataset import (assign_splits, build_split, load_session, make_windows,
                                 read_labels, window_starts, fall_frame_mask, fall_ts_to_epoch)
    from m1_fall.model import build_model

    cfg = load_config(a.config)
    W = cfg.tensor.window_frames
    specs = read_labels(cfg)
    split_of = assign_splits(specs, cfg)
    val_specs = [s for s in specs if split_of[s.csv_file] == "val"]
    print(f"[cfg] {a.config} · labels={cfg.paths.labels_csv} · val 세션 {len(val_specs)}개 "
          f"· halfwidth={cfg.label.fall_halfwidth_ms} · pos_frac={cfg.label.fall_pos_frac}")

    with open(a.markers, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    markers = sorted(int(r["ts_ms"]) for r in rows if r["kind"] == "fall")
    bads = sorted(int(r["ts_ms"]) for r in rows if r["kind"] == "bad")
    print(f"[markers] {a.markers} · fall {len(markers)} · bad {len(bads)}")

    model = build_model(cfg.model, cfg.tensor.n_channels, W, cfg.tensor.n_nodes)
    model = load_checkpoint(model, Path(a.ckpt)).eval()
    bs = int(cfg.train.batch_size)

    # ── 창 단위: 학습 검증 924창 ─────────────────────────────────────────────────
    Xv, yv, gv = build_split(cfg, "val")
    if len(Xv) != EXPECTED_VAL_WINDOWS:
        sys.exit(f"검증 창 {len(Xv)} != {EXPECTED_VAL_WINDOWS}")
    sv = infer(model, Xv, bs)
    tv = np.array([tag_of(g) for g in gv])
    gate_v = np.any(Xv != 0, axis=2)[:, :, W - GATE_TAIL:].all(axis=2).all(axis=1)
    del Xv
    win924 = {}
    for gate_on in (False, True):
        fire = (sv >= THRESHOLD) & (gate_v if gate_on else True)
        win924[gate_on] = {
            "pos": (int(fire[yv == 1].sum()), int((yv == 1).sum())),
            "empty": (int(fire[tv == EMPTY_TAG].sum()), int((tv == EMPTY_TAG).sum())),
            "active": (int(fire[tv == ACTIVE_TAG].sum()), int((tv == ACTIVE_TAG).sum())),
        }
    print(f"[924창] 게이트 미적용 재현율 {win924[False]['pos']} · 빈방 {win924[False]['empty']} "
          f"· 활동 {win924[False]['active']}")

    # ── 200 ms 스트림 ───────────────────────────────────────────────────────────
    sessions = {s.csv_file: load_session(Path(cfg.paths.raw_dir) / s.csv_file, cfg) for s in val_specs}

    # ── 사건 단위 · 924창 세트 (집계 없음) — 요청 [1] 을 기존 검증 창 그대로 적용 ─────────────
    ev924 = {False: [0, 0], True: [0, 0]}          # [감지, 낙상 수]
    cursor = 0
    for spec in val_specs:
        ses = sessions[spec.csv_file]
        lab_epochs = sorted(fall_ts_to_epoch(spec.fall_ts, ses.offset_ms)) if spec.fall_ts else []
        is_fall = fall_frame_mask(ses.grid_epoch_ms, lab_epochs, cfg.label.fall_halfwidth_ms) if lab_epochs else None
        s924 = window_starts(len(ses.frames), W, cfg.window.stride_frames, cfg.window.fall_stride_frames, is_fall)
        seg = slice(cursor, cursor + len(s924))
        cursor += len(s924)
        if not all(g == spec.csv_file for g in gv[seg]):
            sys.exit(f"924창 순서가 {spec.csv_file} 과 맞지 않는다")
        lab, own, _ = attribute(ses.grid_epoch_ms, s924, lab_epochs, cfg)
        if not np.array_equal(lab, yv[seg]):
            sys.exit(f"{spec.csv_file}: 924창 라벨 재계산이 build_split 과 다르다")
        for gate_on in (False, True):
            fire = (sv[seg] >= THRESHOLD) & (gate_v[seg] if gate_on else True)
            for j in range(len(lab_epochs)):
                P = [i for i, o in enumerate(own) if j in o]
                ev924[gate_on][0] += bool(P) and bool(fire[P].any())
                ev924[gate_on][1] += 1
    if cursor != len(sv):
        sys.exit("924창 세션 분할이 전체 창 수와 맞지 않는다")
    print(f"[924창 사건] 게이트 미적용 {ev924[False][0]}/{ev924[False][1]} · 적용 {ev924[True][0]}/{ev924[True][1]}")

    streams_by_phase: Dict[int, List[Stream]] = {}
    for phase in PHASES:
        streams = []
        for spec in val_specs:
            ses = sessions[spec.csv_file]
            X, st = build_stream(ses, spec, cfg, phase, markers, bads)
            if phase == 0:
                # (1) 오프셋 0 스트림 = make_windows(stride 20, fall_stride 없음) 와 동일한가
                c20 = copy.deepcopy(cfg)
                c20.window.stride_frames = HOP_FRAMES
                c20.window.fall_stride_frames = None
                Xm, ym = make_windows(ses, c20, spec.fall_ts)
                same = Xm.shape == X.shape and np.array_equal(Xm, X) and np.array_equal(ym, st.label)
                # (2) 924창 세트와 시작이 겹치는 창의 텐서가 같은가
                is_fall = (fall_frame_mask(ses.grid_epoch_ms, fall_ts_to_epoch(spec.fall_ts, ses.offset_ms),
                                           cfg.label.fall_halfwidth_ms) if spec.fall_ts else None)
                s924 = window_starts(len(ses.frames), W, cfg.window.stride_frames,
                                     cfg.window.fall_stride_frames, is_fall)
                X924, _ = make_windows(ses, cfg, spec.fall_ts)
                pos = {s: i for i, s in enumerate(s924)}
                shared = [(i, pos[s]) for i, s in enumerate(st.starts) if s in pos]
                same924 = all(np.array_equal(X[i], X924[k]) for i, k in shared)
                # (3) 게이트: presence 기반 == 텐서 비영 기반
                g_tensor = np.any(X != 0, axis=2)[:, :, W - GATE_TAIL:].all(axis=2).all(axis=1)
                print(f"[check] {spec.csv_file}: make_windows 동일 {same} · 924창과 겹치는 {len(shared)}창 "
                      f"텐서 동일 {same924} · 게이트 정의 일치 {bool(np.array_equal(g_tensor, st.gate))} "
                      f"· 마커-라벨 {sorted({f['lag_ms'] for f in st.falls})} ms")
                if not (same and same924 and np.array_equal(g_tensor, st.gate)):
                    sys.exit("스트림 창이 데이터셋 규칙과 다르다 — 측정 중단")
                del Xm, X924
            st.score = infer(model, X, bs)
            del X
            streams.append(st)
        streams_by_phase[phase] = streams
        print(f"[stream] 오프셋 {phase * 10} ms · 추론 {sum(len(s.starts) for s in streams)}회", flush=True)

    # ── 집계 ─────────────────────────────────────────────────────────────────
    results = {(p, g, agg): evaluate(streams_by_phase[p], g, agg)
               for p in PHASES for g in (False, True) for agg in AGGS}
    s0 = streams_by_phase[0]

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 추론 단위 (오프셋 0)
    with (out_dir / "eval_event_stream.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["csv_file", "tag", "infer_idx", "start_frame", "end_epoch_ms", "label", "fall_idx",
                    "gate_pass", "score"])
        for st in s0:
            for i in range(len(st.starts)):
                w.writerow([st.csv_file, st.tag, i, int(st.starts[i]), int(st.end_ms[i]), int(st.label[i]),
                            ";".join(str(j) for j in st.owner[i]), int(st.gate[i]), f"{st.score[i]:.6f}"])
    # 낙상 단위 (오프셋 0)
    with (out_dir / "eval_event_falls.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = [f"{'gate' if g else 'nogate'}_{agg}" for g in (False, True) for agg in AGGS]
        w.writerow(["csv_file", "fall_idx", "marker_ms", "label_epoch_ms", "bad", "n_pos_windows",
                    "pos_max_score", "pos_gate_pass"] + cols)
        for st in s0:
            for f in st.falls:
                P = [i for i, o in enumerate(st.owner) if f["idx"] in o]
                det = []
                for g in (False, True):
                    for agg in AGGS:
                        pf = [x for x in results[(0, g, agg)]["per_fall"]
                              if x["csv_file"] == st.csv_file and x["fall_idx"] == f["idx"]][0]
                        det.append(int(pf["detected"]))
                w.writerow([st.csv_file, f["idx"], f["marker_ms"], f["label_epoch_ms"], int(f["bad"]), len(P),
                            f"{st.score[P].max():.6f}" if P else "", int(st.gate[P].sum()) if P else 0] + det)
    # 조건 요약 (전 오프셋)
    keys = ["fall_n", "fall_k", "fall_k_ext", "fall_k_strict", "nobad_n", "nobad_k", "falls_without_pos", "pos_n", "pos_fire",
            "empty_events", "empty_alarm_infer", "empty_fire_infer", "empty_infer", "empty_hours", "empty_fah",
            "active_events", "active_alarm_infer", "active_fire_infer", "active_infer", "active_hours",
            "active_fah"]
    with (out_dir / "eval_event_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["phase_ms", "gate", "agg"] + keys)
        for (p, g, agg), r in results.items():
            w.writerow([p * 10, int(g), agg] + [f"{r[k]:.6f}" if isinstance(r[k], float) else r[k] for k in keys])
    print(f"[out] {out_dir}/eval_event_stream.csv · eval_event_falls.csv · eval_event_summary.csv")

    # ── 판정 ─────────────────────────────────────────────────────────────────
    cands = [results[(0, g, f"K{k}")] for k in K_LIST for g in (False, True)]

    def rank(r):
        return (-r["fall_k"], r["empty_events"] + r["active_events"], 0 if r["gate"] else 1, int(r["agg"][1:]))

    judge = {}
    for name, ok in (("A", lambda r: r["empty_fah"] < FA_TARGET and r["active_fah"] < FA_TARGET),
                     ("B", lambda r: r["empty_fah"] < FA_TARGET)):
        passed = sorted([r for r in cands if ok(r)], key=rank)
        judge[name] = passed[0] if passed else None
    low_empty = sorted(cands, key=lambda r: (r["empty_fah"], -r["fall_k"]))[0]
    low_active = sorted(cands, key=lambda r: (r["active_fah"], -r["fall_k"]))[0]

    # ── markdown ─────────────────────────────────────────────────────────────
    L: List[str] = []
    e = L.append
    n_inf_empty = results[(0, False, "none")]["empty_infer"]
    n_inf_active = results[(0, False, "none")]["active_infer"]
    h_e = results[(0, False, "none")]["empty_hours"]
    h_a = results[(0, False, "none")]["active_hours"]

    def aggname(agg):
        return "집계 없음" if agg == "none" else f"K={agg[1:]} / N={N_AGG}"

    def gname(g):
        return "적용" if g else "미적용"

    e("### 스트림 규모 (오프셋 0)")
    e("")
    e("| 세션 | 태그 | 추론 수 | 관측 시간 | 라벨 1 추론 | 낙상 | 게이트 통과 추론 |")
    e("|---|---|---:|---:|---:|---:|---:|")
    for st in s0:
        e(f"| `{st.csv_file[13:-4]}` | {st.tag} | {len(st.starts)} | {len(st.starts) * 0.2:.1f} s "
          f"| {int(st.label.sum())} | {len(st.falls)} | {int(st.gate.sum())}/{len(st.starts)} "
          f"({100 * st.gate.mean():.1f}%) |")
    orphan = sum(st.n_orphan_pos for st in s0)
    e("")
    e(f"빈 방 관측 {n_inf_empty}회 = {h_e * 3600:.1f} s = {h_e:.5f} h → **오탐 1건 = {1 / h_e:.1f} FA/h** · "
      f"활동 관측 {n_inf_active}회 = {h_a * 3600:.1f} s = {h_a:.5f} h → **1건 = {1 / h_a:.1f} FA/h** · "
      f"단독 귀속이 안 돼 최대 비율 낙상에 귀속한 양성 추론 {orphan}개")
    e("")

    e("### 게이트 × 집계 — 오프셋 0")
    e("")
    e("| 게이트 | 집계 | 사건 재현율 | 빈 방 오탐 건수 | 빈 방 FA/h | 빈 방 경보 켜진 추론 | 활동 오탐 건수 "
      "| 활동 FA/h | 활동 경보 켜진 추론 | (참고) bad 제외 재현율 | (참고) 판정 +0.8 s 재현율 "
      "| (참고) 엄격 재현율 |")
    e("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for g in (False, True):
        for agg in AGGS:
            r = results[(0, g, agg)]
            e(f"| {gname(g)} | {aggname(agg)} | **{r['fall_k']}/{r['fall_n']}** = {r['fall_k'] / r['fall_n']:.3f} "
              f"| {r['empty_events']} | **{r['empty_fah']:.1f}** | {r['empty_alarm_infer']}/{r['empty_infer']} "
              f"| {r['active_events']} | **{r['active_fah']:.1f}** | {r['active_alarm_infer']}/{r['active_infer']} "
              f"| {r['nobad_k']}/{r['nobad_n']} | {r['fall_k_ext']}/{r['fall_n']} "
              f"| {r['fall_k_strict']}/{r['fall_n']} |")
    e("")

    e("### 창 단위 vs 사건 단위 (임계 0.80)")
    e("")
    e("| 게이트 | 단위 | 낙상 쪽 | 빈 방 쪽 | 활동 쪽 |")
    e("|---|---|---:|---:|---:|")
    for g in (False, True):
        w9 = win924[g]
        r0 = results[(0, g, "none")]
        e(f"| {gname(g)} | 창 · 학습 검증 924창 (stride 50 / 낙상 근처 20) "
          f"| 양성 창 {w9['pos'][0]}/{w9['pos'][1]} = {w9['pos'][0] / w9['pos'][1]:.3f} "
          f"| 발화 창 {w9['empty'][0]}/{w9['empty'][1]} = {100 * w9['empty'][0] / w9['empty'][1]:.1f}% "
          f"| 발화 창 {w9['active'][0]}/{w9['active'][1]} = {100 * w9['active'][0] / w9['active'][1]:.1f}% |")
        e(f"| {gname(g)} | 사건 · 924창 세트 · 집계 없음 "
          f"| 낙상 {ev924[g][0]}/{ev924[g][1]} = {ev924[g][0] / ev924[g][1]:.3f} | (시간축 불규칙 — FA/h 없음) | — |")
        e(f"| {gname(g)} | 창 · 200 ms 스트림 "
          f"| 양성 추론 {r0['pos_fire']}/{r0['pos_n']} = {r0['pos_fire'] / r0['pos_n']:.3f} "
          f"| 발화 추론 {r0['empty_fire_infer']}/{r0['empty_infer']} = "
          f"{100 * r0['empty_fire_infer'] / r0['empty_infer']:.1f}% "
          f"| 발화 추론 {r0['active_fire_infer']}/{r0['active_infer']} = "
          f"{100 * r0['active_fire_infer'] / r0['active_infer']:.1f}% |")
        for agg in AGGS:
            r = results[(0, g, agg)]
            e(f"| {gname(g)} | **사건 · {aggname(agg)}** "
              f"| 낙상 {r['fall_k']}/{r['fall_n']} = {r['fall_k'] / r['fall_n']:.3f} "
              f"| {r['empty_events']}건 = {r['empty_fah']:.1f} FA/h "
              f"| {r['active_events']}건 = {r['active_fah']:.1f} FA/h |")
    e("")

    e("### 판정")
    e("")
    for name, desc in (("A", "빈 방 FA/h < 1.0 그리고 활동 FA/h < 1.0"), ("B", "빈 방 FA/h < 1.0 만 (참고)")):
        r = judge[name]
        if r is None:
            e(f"- **{name}** ({desc}): 만족하는 조합 없음")
        else:
            e(f"- **{name}** ({desc}): **{aggname(r['agg'])}, 게이트 {gname(r['gate'])}** — 사건 재현율 "
              f"{r['fall_k']}/{r['fall_n']} = {r['fall_k'] / r['fall_n']:.3f} · 빈 방 {r['empty_events']}건 "
              f"({r['empty_fah']:.1f} FA/h) · 활동 {r['active_events']}건 ({r['active_fah']:.1f} FA/h)")
    e(f"- 빈 방 FA/h 최솟값: {low_empty['empty_fah']:.1f} ({aggname(low_empty['agg'])}, 게이트 "
      f"{gname(low_empty['gate'])}, 재현율 {low_empty['fall_k']}/{low_empty['fall_n']})")
    e(f"- 활동 FA/h 최솟값: {low_active['active_fah']:.1f} ({aggname(low_active['agg'])}, 게이트 "
      f"{gname(low_active['gate'])}, 재현율 {low_active['fall_k']}/{low_active['fall_n']})")
    e("")

    e("### 세션별 사건 재현율 (오프셋 0)")
    e("")
    e("| 세션 | 낙상 | 게이트 | " + " | ".join(aggname(x) for x in AGGS) + " |")
    e("|---|---:|---|" + "---:|" * len(AGGS))
    for g in (False, True):
        for st in s0:
            if not st.falls:
                continue
            cells = []
            for agg in AGGS:
                k = sum(x["detected"] for x in results[(0, g, agg)]["per_fall"] if x["csv_file"] == st.csv_file)
                cells.append(f"{k}/{len(st.falls)}")
            e(f"| `{st.csv_file[13:-4]}` | {len(st.falls)} | {gname(g)} | " + " | ".join(cells) + " |")
    e("")

    e("### 낙상별 (오프셋 0) — 감지 여부")
    e("")
    e("| 세션 | # | 마커 ms | bad | 양성 추론 | 그중 최고 점수 | 그중 게이트 통과 | 미적용: 없음 K1 K2 K3 K4 K5 "
      "| 적용: 없음 K1 K2 K3 K4 K5 |")
    e("|---|---:|---:|---|---:|---:|---:|---|---|")
    for st in s0:
        for f in st.falls:
            P = [i for i, o in enumerate(st.owner) if f["idx"] in o]
            marks = []
            for g in (False, True):
                cells = []
                for agg in AGGS:
                    pf = [x for x in results[(0, g, agg)]["per_fall"]
                          if x["csv_file"] == st.csv_file and x["fall_idx"] == f["idx"]][0]
                    cells.append("●" if pf["detected"] else "·")
                marks.append(" ".join(cells))
            e(f"| `{st.csv_file[13:-4]}` | {f['idx'] + 1} | {f['marker_ms']} | {'bad' if f['bad'] else ''} "
              f"| {len(P)} | {st.score[P].max():.4f} | {int(st.gate[P].sum())}/{len(P)} | {marks[0]} | {marks[1]} |")
    e("")

    e("### (보조) 추론 시작 오프셋별 범위 — 0/40/80/120/160 ms")
    e("")
    e("| 게이트 | 집계 | 사건 재현율 (최소 ~ 최대) | 오프셋별 재현율 | 빈 방 오탐 건수 (최소 ~ 최대) "
      "| 오프셋별 빈 방 오탐 건수 | 활동 오탐 건수 |")
    e("|---|---|---:|---|---:|---|---:|")
    for g in (False, True):
        for agg in AGGS:
            rs = [results[(p, g, agg)] for p in PHASES]
            fk = [r["fall_k"] for r in rs]
            ee = [r["empty_events"] for r in rs]
            ae = [r["active_events"] for r in rs]
            e(f"| {gname(g)} | {aggname(agg)} | {min(fk)} ~ {max(fk)} /{rs[0]['fall_n']} "
              f"| {' / '.join(str(x) for x in fk)} | {min(ee)} ~ {max(ee)} | {' / '.join(str(x) for x in ee)} "
              f"| {min(ae)} ~ {max(ae)} |")
    e("")

    text = "\n".join(L)
    print("\n" + text)
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(text, encoding="utf-8")
        print(f"[out] markdown -> {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
