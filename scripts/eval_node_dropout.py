#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_node_dropout.py — 노드 슬롯을 인위로 0으로 만들었을 때 M1 점수가 얼마나 변하는지 잰다.

학습은 하지 않는다. 체크포인트·라벨·yaml·원본 CSV 는 읽기만 한다.

질문
  런타임에서 노드가 끊기면 그 슬롯의 빈 프레임은 0으로 채워진다.
  학습 데이터에는 산발적 결측(슬롯 일부 0)만 있었고, 한 노드 슬롯 전체가 0인
  창은 없었다. 그 입력에서 점수가 버티는지, 추론을 생략해야 하는지를 측정으로 정한다.

모드
  --mode full     노드 슬롯을 창 전체에 걸쳐 0 (노드 1개·2개·3개 조합 7조건)  [기본]
  --mode partial  한 노드씩, 창 100프레임 중 일부 프레임만 남기고 나머지를 0

공통 방법
  1. 창은 `m1_fall.dataset.build_split` 으로만 만든다 (학습과 동일 전처리).
  2. 0으로 덮는 값은 로더가 결측 슬롯을 채우는 값과 같은 0 이다.
  3. 체크포인트는 strict=True 로만 로드. 출력 logit 에 sigmoid 를 여기서 한 번 적용.

── full 판정 기준 (측정 전 고정 — 숫자를 보고 바꾸지 않는다) ──────────────────────
  한 노드 0 조건 3개(노드1·노드2·노드3) **각각**에 대해
    (a) 검증셋 전체 창의 |기준점수 - 조건점수| 중앙값 < SAFE_MEDIAN_MAX
    (b) 기준 재현율@THRESHOLD - 조건 재현율@THRESHOLD   < SAFE_RECALL_DROP_MAX
  셋 모두 (a)(b) 통과 → "0 패딩으로 두어도 안전"
  하나라도 실패       → "학습 슬롯이 다 차기 전에는 추론 생략이 맞음"

── partial 방법 ──────────────────────────────────────────────────────────────────
  커버리지 c% = 창 100프레임 중 c개 프레임 위치만 남기고 나머지 위치를 0으로 덮는다
  (한 노드씩, 나머지 두 노드는 원본 그대로). 원본에도 결측이 있으므로 실제로
  실프레임이 남는 비율은 c 이하다 — 노드별 실측 커버리지를 같이 낸다.
  100% 는 추가 결측 없음 = 기준 그 자체다.

  결측 방식 (MASK_METHODS)
    scatter     (a) 무작위 분산 결측 — 빠질 프레임 위치를 창마다 무작위로 고름
    block       (b) 연속 구간 결측 — 길이 100-c 구간, 시작 위치를 창마다 무작위로 고름
    block_head  (보조) 연속 구간을 창 맨 앞(가장 과거 프레임)에 고정
    block_tail  (보조) 연속 구간을 창 맨 끝(가장 최근 프레임)에 고정
  무작위 방식은 PARTIAL_SEEDS 의 시드마다 따로 재고 판정에는 시드 중 **최악값**을 쓴다.

── partial 판정 기준 (측정 전 고정) ────────────────────────────────────────────────
  통과 = 재현율@0.80 하락 < PARTIAL_RECALL_DROP_MAX
         그리고 빈 방 신규 발화 없음
         (= 기준 점수가 0.80 미만이던 빈 방 창들의 조건 점수 최댓값 < 0.80)
  요청 원문은 "빈 방 최댓값 < 0.80" 이지만 기준(100%) 자체의 빈 방 최댓값이 0.8591 로
  이미 0.80 이상이라, 원문대로면 100% 도 실패해 답이 나오지 않는다.
  원문 기준(빈 방 전체 최댓값 < 0.80)도 같이 계산해 보고한다.

  최소 커버리지 = 100% 부터 내려가며 **연속으로** 통과한 마지막 단계 (중간에 실패하면 거기서 멈춤).
  노드 값 = (a)(b) 중 높은 쪽. 권고치 = 세 노드 값 중 가장 높은 값.
  (b-앞)(b-끝)은 판정과 별도로 같은 규칙의 최소 커버리지를 보고한다.
  (a)와 (b)의 차이가 "크다" = 최소 커버리지가 다르거나, 같은 커버리지에서
  재현율 또는 |Δ| 중앙값 차이가 BIG_DIFF 이상.

── tail 방법 (--mode tail) ─────────────────────────────────────────────────────────
  한 노드씩, 창 맨 끝(가장 최근) L 프레임만 0으로 덮는다 (L = TAIL_GAPS, 1~19).
  그 외 프레임은 원본 그대로 — 추가 결측 없음. 결정적이라 시드가 없다.
  L=10 은 partial 의 block_tail 90% 와 같은 조건이다 (값이 같아야 한다).

── tail 판정 기준 (측정 전 고정) ───────────────────────────────────────────────────
  통과 = partial 과 동일 (재현율@0.80 하락 < 0.05 그리고 기준 미발화 빈 방 최댓값 < 0.80)
  노드별 허용 공백 L_safe = L=1 부터 **연속으로** 통과한 마지막 길이 (L=1 부터 실패면 0).
  권고 N = min(노드별 L_safe) + 1  — 창 끝 공백이 N 프레임이면 기준 실패가 시작되는 길이.
  규칙 = "창 끝 N프레임 안에 결측이 있으면 추론 생략".
  원본 검증 창에 이 규칙을 적용했을 때 생략되는 창 비율도 같이 낸다 (추론 없이 계산).

  (보조) 규칙이 **통과시키는** 창에서도 다른 위치 결측이 해로운가:
  partial 의 창 단위 점수(첫 시드)를 읽고 같은 시드로 마스크를 재생성해,
  규칙 통과 창(세 노드 모두 창 끝 N 프레임에 실프레임이 있음)만 골라 같은 판정을 다시 한다.
  새 추론은 하지 않는다. 창 순서·기준 점수가 현재 로더와 일치하는지 먼저 확인한다.

사용
    $env:PYTHONPATH="src"
    .venv\\Scripts\\python.exe scripts\\eval_node_dropout.py
    .venv\\Scripts\\python.exe scripts\\eval_node_dropout.py --mode partial
    .venv\\Scripts\\python.exe scripts\\eval_node_dropout.py --mode tail
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

THRESHOLD = 0.80                 # docs/M1_HANDOFF.md 권고 운영 임계값
SAFE_MEDIAN_MAX = 0.05           # full 판정 (a)
SAFE_RECALL_DROP_MAX = 0.05      # full 판정 (b)
EMPTY_TAG = "normal_empty"       # 빈 방 세션 태그 (파일명 끝)
EXPECTED_VAL_WINDOWS = 924       # 학습 val / M1_HANDOFF.md 와 같은 모집단인지 확인용

# (이름, 0으로 만들 노드축 인덱스). 노드 n 은 인덱스 n-1 (dataset.py: frames[idx, nid-1]).
CONDITIONS = [
    ("base", ()),
    ("n1_zero", (0,)),
    ("n2_zero", (1,)),
    ("n3_zero", (2,)),
    ("n12_zero", (0, 1)),
    ("n13_zero", (0, 2)),
    ("n23_zero", (1, 2)),
    ("all_zero", (0, 1, 2)),
]
SINGLE_NODE = ("n1_zero", "n2_zero", "n3_zero")

# ── partial ──
COVERAGE_LEVELS = [100, 90, 80, 70, 60, 50, 30, 10]   # 창 100프레임 중 남기는 프레임 %
PARTIAL_SEEDS = [0, 1, 2, 3, 4]
PARTIAL_RECALL_DROP_MAX = 0.05
BIG_DIFF = 0.05
# (키, 표시 이름, 무작위 여부)
MASK_METHODS = [
    ("scatter", "(a) 무작위 분산 결측", True),
    ("block", "(b) 연속 구간 결측 · 위치 무작위", True),
    ("block_head", "(b-앞) 연속 구간 결측 · 창 앞(과거) 고정", False),
    ("block_tail", "(b-끝) 연속 구간 결측 · 창 끝(최근) 고정", False),
]
JUDGE_METHODS = ("scatter", "block")

# ── tail ──
TAIL_GAPS = list(range(1, 20))     # 창 끝에서 0으로 덮는 프레임 수 (100 Hz → 10 ms/프레임)


def tag_of(csv_file: str) -> str:
    """'csi_20260916_1356_normal_empty.csv' -> 'normal_empty'"""
    parts = Path(csv_file).stem.split("_")
    return "_".join(parts[3:]) if len(parts) > 3 else "untagged"


def load_checkpoint(model, ckpt_path: Path):
    import torch
    if not ckpt_path.exists():
        sys.exit(f"체크포인트가 없다: {ckpt_path}")
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model.load_state_dict(state, strict=True)    # strict=False 금지 (CLAUDE.md 규칙 6)
    epoch = blob.get("epoch") if isinstance(blob, dict) else "?"
    print(f"[ckpt] {ckpt_path} · strict=True 로드 성공 · 키 {len(state)}개 · epoch {epoch}")
    metrics = blob.get("metrics") if isinstance(blob, dict) else None
    return model, (metrics if isinstance(metrics, dict) else {})


def infer(model, X: np.ndarray, bs: int) -> np.ndarray:
    import torch
    out = np.empty(len(X), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(X), bs):
            logit = model(torch.from_numpy(X[i:i + bs]))    # (B,1) pre-sigmoid
            out[i:i + bs] = torch.sigmoid(logit).numpy().reshape(-1)
    return out


def dist(s: np.ndarray) -> dict:
    p = np.percentile(s, [5, 25, 50, 75, 95])
    return {"n": len(s), "mean": float(s.mean()), "min": float(s.min()),
            "p5": p[0], "p25": p[1], "p50": p[2], "p75": p[3], "p95": p[4], "max": float(s.max())}


# ── 공통 준비: 창·모델·기준 점수 ───────────────────────────────────────────────────

@dataclass
class Ctx:
    X: np.ndarray               # (N, 3, 64, 100) 원본 검증 창 — 절대 수정하지 않는다
    labels: np.ndarray          # (N,) 0/1
    groups: List[str]           # 창별 csv 파일
    tags: np.ndarray            # 창별 행동 태그
    is_fall: np.ndarray         # 낙상 라벨 창
    is_empty: np.ndarray        # 빈 방 창
    present: np.ndarray         # (N, 3, 100) 원본에서 실프레임이 있던 프레임
    model: object
    bs: int
    base: np.ndarray            # (N,) 기준 점수


def load_context(config_path: str, ckpt_path: str) -> Ctx:
    from sklearn.metrics import average_precision_score
    from m1_fall.config import load_config
    from m1_fall.dataset import build_split
    from m1_fall.model import build_model

    cfg = load_config(config_path)
    print(f"[cfg] {config_path} · labels_csv={cfg.paths.labels_csv} · n_nodes={cfg.tensor.n_nodes} "
          f"· stride={cfg.window.stride_frames} · fall_stride={cfg.window.fall_stride_frames}")
    if cfg.tensor.n_nodes != 3:
        sys.exit(f"이 측정은 3노드 전제다. n_nodes={cfg.tensor.n_nodes}")

    X, y, groups = build_split(cfg, "val")
    labels = y.astype(int)
    tags = np.array([tag_of(g) for g in groups])
    is_fall = labels == 1
    is_empty = tags == EMPTY_TAG
    print(f"[data] val 창 {len(X)}개 {tuple(X.shape[1:])} · 파일 {len(set(groups))}개 "
          f"· 낙상 라벨 {int(is_fall.sum())} · 빈 방 {int(is_empty.sum())} "
          f"({sorted({g for g, e in zip(groups, is_empty) if e})})")
    if len(X) != EXPECTED_VAL_WINDOWS:
        sys.exit(f"창 수 {len(X)} != {EXPECTED_VAL_WINDOWS}. 학습 val 과 다른 모집단이다.")

    # 원래 데이터에서 노드 슬롯이 얼마나 차 있었는지 (실프레임은 peak=1 이라 전부 0일 수 없다)
    present = np.any(X != 0, axis=2)                   # (N, 3, 100) 프레임별 실프레임 여부
    for n in range(3):
        cov = present[:, n, :].mean(axis=1)            # 창별 커버리지
        print(f"[cover] 노드{n + 1}: 창 평균 커버리지 {cov.mean() * 100:.1f}% "
              f"· 최소 {cov.min() * 100:.0f}% · 슬롯 전체가 0인 창 {int((cov == 0).sum())}개")

    model = build_model(cfg.model, cfg.tensor.n_channels, cfg.tensor.window_frames, cfg.tensor.n_nodes)
    model, ck_metrics = load_checkpoint(model, Path(ckpt_path))
    model.eval()
    bs = int(cfg.train.batch_size)
    base = infer(model, X, bs)

    # 체크포인트에 기록된 val 지표와 기준 조건이 일치하는지 (파이프라인 동일성 증거)
    base_pr = float(average_precision_score(labels, base))
    if "pr_auc" in ck_metrics:
        print(f"[check] 기준 pr_auc {base_pr:.6f} vs 체크포인트 기록 {float(ck_metrics['pr_auc']):.6f}")
    rec05 = float((base[is_fall] >= 0.5).mean())
    if "fall_recall" in ck_metrics:
        print(f"[check] 기준 recall@0.5 {rec05:.6f} vs 체크포인트 기록 {float(ck_metrics['fall_recall']):.6f}")

    return Ctx(X=X, labels=labels, groups=groups, tags=tags, is_fall=is_fall, is_empty=is_empty,
               present=present, model=model, bs=bs, base=base)


def write_window_csv(ctx: Ctx, out: str, columns: List[str], scores: dict) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_file_idx: dict = {}
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["csv_file", "tag", "window_idx", "label"] + columns)
        for i, g in enumerate(ctx.groups):
            k = per_file_idx.get(g, 0)
            per_file_idx[g] = k + 1
            w.writerow([g, ctx.tags[i], k, ctx.labels[i]] + [f"{scores[n][i]:.6f}" for n in columns])
    print(f"\n[out] 창 단위 결과 {len(ctx.groups)}행 × 점수 {len(columns)}열 -> {out_path}")


# ── full: 노드 슬롯 전체 0 ───────────────────────────────────────────────────────

def run_full(ctx: Ctx, out: str) -> None:
    from sklearn.metrics import average_precision_score

    X, labels, is_fall, is_empty = ctx.X, ctx.labels, ctx.is_fall, ctx.is_empty
    scores = {}
    for name, idx in CONDITIONS:
        if not idx:
            scores[name] = ctx.base
            continue
        Xc = X.copy()
        Xc[:, list(idx)] = 0.0
        scores[name] = infer(ctx.model, Xc, ctx.bs)
        del Xc
    base = scores["base"]

    # ── 조건별 지표 ────────────────────────────────────────────────────────────
    rows = []
    for name, idx in CONDITIONS:
        s = scores[name]
        d = np.abs(base - s)
        r = {
            "cond": name,
            "zeroed": ",".join(str(i + 1) for i in idx) or "-",
            "pr_auc": float(average_precision_score(labels, s)),
            "recall_k": int((s[is_fall] >= THRESHOLD).sum()),
            "recall_n": int(is_fall.sum()),
            "empty_k": int((s[is_empty] >= THRESHOLD).sum()),
            "empty_n": int(is_empty.sum()),
            "neg_k": int((s[~is_fall] >= THRESHOLD).sum()),
            "neg_n": int((~is_fall).sum()),
            "dist_all": dist(s), "dist_fall": dist(s[is_fall]), "dist_empty": dist(s[is_empty]),
            "d_all_med": float(np.median(d)), "d_all_p95": float(np.percentile(d, 95)),
            "d_fall_med": float(np.median(d[is_fall])), "d_fall_p95": float(np.percentile(d[is_fall], 95)),
            "d_empty_med": float(np.median(d[is_empty])), "d_empty_p95": float(np.percentile(d[is_empty], 95)),
            "signed_all_mean": float(np.mean(s - base)),
            "signed_fall_mean": float(np.mean(s[is_fall] - base[is_fall])),
            "signed_empty_mean": float(np.mean(s[is_empty] - base[is_empty])),
            # 임계값을 넘나든 창 수
            "fall_lost": int(((base >= THRESHOLD) & (s < THRESHOLD) & is_fall).sum()),
            "fall_gained": int(((base < THRESHOLD) & (s >= THRESHOLD) & is_fall).sum()),
            "empty_gained": int(((base < THRESHOLD) & (s >= THRESHOLD) & is_empty).sum()),
            "empty_lost": int(((base >= THRESHOLD) & (s < THRESHOLD) & is_empty).sum()),
        }
        r["recall"] = r["recall_k"] / r["recall_n"]
        r["empty_rate"] = r["empty_k"] / r["empty_n"]
        r["neg_rate"] = r["neg_k"] / r["neg_n"]
        rows.append(r)
    base_row = rows[0]
    for r in rows:
        r["recall_drop"] = base_row["recall"] - r["recall"]

    # ── 출력 ──────────────────────────────────────────────────────────────────
    print(f"\n[표 1] 조건별 지표 (임계 {THRESHOLD:.2f})")
    print(f"{'조건':<10}{'0노드':>7}{'pr_auc':>9}{'재현율':>16}{'하락':>8}{'빈방발화':>14}{'비낙상발화':>15}")
    for r in rows:
        print(f"{r['cond']:<10}{r['zeroed']:>7}{r['pr_auc']:>9.4f}"
              f"{r['recall_k']:>5}/{r['recall_n']} {r['recall']:.3f}{r['recall_drop']:>+8.3f}"
              f"{r['empty_k']:>5}/{r['empty_n']} {r['empty_rate'] * 100:4.1f}%"
              f"{r['neg_k']:>5}/{r['neg_n']} {r['neg_rate'] * 100:4.1f}%")

    for key, title in (("dist_all", "전체"), ("dist_fall", "낙상 라벨"), ("dist_empty", "빈 방")):
        print(f"\n[표 2] 점수 분포 — {title}")
        print(f"{'조건':<10}{'n':>5}{'mean':>8}{'min':>8}{'p5':>8}{'p25':>8}{'p50':>8}{'p75':>8}{'p95':>8}{'max':>8}")
        for r in rows:
            q = r[key]
            print(f"{r['cond']:<10}{q['n']:>5}{q['mean']:>8.4f}{q['min']:>8.4f}{q['p5']:>8.4f}"
                  f"{q['p25']:>8.4f}{q['p50']:>8.4f}{q['p75']:>8.4f}{q['p95']:>8.4f}{q['max']:>8.4f}")

    print("\n[표 3] |기준 - 조건| — 중앙값 / 95분위  (부호있는 평균: 조건-기준)")
    print(f"{'조건':<10}{'전체 med':>10}{'p95':>8}{'낙상 med':>10}{'p95':>8}{'빈방 med':>10}{'p95':>8}"
          f"{'Δ전체':>9}{'Δ낙상':>9}{'Δ빈방':>9}")
    for r in rows[1:]:
        print(f"{r['cond']:<10}{r['d_all_med']:>10.4f}{r['d_all_p95']:>8.4f}{r['d_fall_med']:>10.4f}"
              f"{r['d_fall_p95']:>8.4f}{r['d_empty_med']:>10.4f}{r['d_empty_p95']:>8.4f}"
              f"{r['signed_all_mean']:>+9.4f}{r['signed_fall_mean']:>+9.4f}{r['signed_empty_mean']:>+9.4f}")

    print(f"\n[표 4] 임계 {THRESHOLD:.2f} 를 넘나든 창")
    print(f"{'조건':<10}{'낙상 잃음':>10}{'낙상 새로':>10}{'빈방 새로':>10}{'빈방 사라짐':>12}")
    for r in rows[1:]:
        print(f"{r['cond']:<10}{r['fall_lost']:>10}{r['fall_gained']:>10}{r['empty_gained']:>10}{r['empty_lost']:>12}")

    allz = scores["all_zero"]
    print(f"\n[check] 전부 0 조건 점수: min {allz.min():.6f} max {allz.max():.6f} (입력이 같으니 상수여야 한다)")

    print(f"\n[판정] 기준: 한 노드 0 조건마다 전체 |Δ| 중앙값 < {SAFE_MEDIAN_MAX} 그리고 "
          f"재현율 하락 < {SAFE_RECALL_DROP_MAX}")
    all_pass = True
    for r in rows:
        if r["cond"] not in SINGLE_NODE:
            continue
        ok_a = r["d_all_med"] < SAFE_MEDIAN_MAX
        ok_b = r["recall_drop"] < SAFE_RECALL_DROP_MAX
        all_pass &= ok_a and ok_b
        print(f"  {r['cond']}: 중앙값 {r['d_all_med']:.4f} {'통과' if ok_a else '실패'} · "
              f"재현율 하락 {r['recall_drop']:+.4f} {'통과' if ok_b else '실패'}")
    verdict = "0 패딩으로 두어도 안전" if all_pass else "학습 슬롯이 다 차기 전에는 추론 생략이 맞음"
    print(f"  → {verdict}")

    write_window_csv(ctx, out, [n for n, _ in CONDITIONS], scores)


# ── partial: 창 안 일부 프레임만 0 ──────────────────────────────────────────────────

def keep_mask(method: str, n_win: int, n_frames: int, keep_frames: int, rng) -> np.ndarray:
    """(n_win, n_frames) bool — True 인 프레임 위치만 남긴다. 창마다 독립."""
    gap = n_frames - keep_frames
    keep = np.ones((n_win, n_frames), dtype=bool)
    if gap == 0:
        return keep
    if method == "scatter":
        drop = np.argsort(rng.random((n_win, n_frames)), axis=1)[:, :gap]
        np.put_along_axis(keep, drop, False, axis=1)
        return keep
    if method == "block":
        start = rng.integers(0, n_frames - gap + 1, size=n_win)
    elif method == "block_head":
        start = np.zeros(n_win, dtype=np.int64)
    elif method == "block_tail":
        start = np.full(n_win, n_frames - gap, dtype=np.int64)
    else:
        raise ValueError(method)
    t = np.arange(n_frames)
    return ~((t >= start[:, None]) & (t < start[:, None] + gap))


def partial_metrics(ctx: Ctx, s: np.ndarray, base_recall_k: int) -> dict:
    from sklearn.metrics import average_precision_score

    d = np.abs(ctx.base - s)
    f, e = ctx.is_fall, ctx.is_empty
    quiet = e & (ctx.base < THRESHOLD)          # 기준에서 발화하지 않던 빈 방 창
    n_fall = int(f.sum())
    rk = int((s[f] >= THRESHOLD).sum())
    return {
        "pr_auc": float(average_precision_score(ctx.labels, s)),
        "recall_k": rk,
        "recall_n": n_fall,
        "recall": rk / n_fall,
        "recall_drop": (base_recall_k - rk) / n_fall,
        "empty_fire_k": int((s[e] >= THRESHOLD).sum()),
        "empty_n": int(e.sum()),
        "empty_max": float(s[e].max()),
        "empty_quiet_n": int(quiet.sum()),
        "empty_quiet_max": float(s[quiet].max()),
        "empty_new_fire_k": int((s[quiet] >= THRESHOLD).sum()),
        "d_med": float(np.median(d)),
        "d_p95": float(np.percentile(d, 95)),
        "d_fall_med": float(np.median(d[f])),
        "d_empty_med": float(np.median(d[e])),
    }


def aggregate(recs: List[dict]) -> dict:
    """시드들 → 최악값(판정용) + 범위."""
    return {
        "n_seeds": len(recs),
        "cov_mean": float(np.mean([r["cov_mean"] for r in recs])),
        "cov_min": float(min(r["cov_min"] for r in recs)),
        "pr_auc": min(r["pr_auc"] for r in recs),
        "pr_auc_hi": max(r["pr_auc"] for r in recs),
        "recall_k": min(r["recall_k"] for r in recs),
        "recall_k_hi": max(r["recall_k"] for r in recs),
        "recall_n": recs[0]["recall_n"],
        "recall_drop": max(r["recall_drop"] for r in recs),
        "empty_fire_k": max(r["empty_fire_k"] for r in recs),
        "empty_n": recs[0]["empty_n"],
        "empty_max": max(r["empty_max"] for r in recs),
        "empty_max_lo": min(r["empty_max"] for r in recs),
        "empty_quiet_n": recs[0]["empty_quiet_n"],
        "empty_quiet_max": max(r["empty_quiet_max"] for r in recs),
        "empty_quiet_max_lo": min(r["empty_quiet_max"] for r in recs),
        "empty_new_fire_k": max(r["empty_new_fire_k"] for r in recs),
        "d_med": max(r["d_med"] for r in recs),
        "d_med_lo": min(r["d_med"] for r in recs),
        "recall_mean": float(np.mean([r["recall"] for r in recs])),
        "d_med_mean": float(np.mean([r["d_med"] for r in recs])),
    }


def passes(a: dict) -> bool:
    return a["recall_drop"] < PARTIAL_RECALL_DROP_MAX and a["empty_quiet_max"] < THRESHOLD


def passes_literal(a: dict) -> bool:
    return a["recall_drop"] < PARTIAL_RECALL_DROP_MAX and a["empty_max"] < THRESHOLD


def min_coverage(agg: dict, node: int, method: str, rule) -> int | None:
    """100% 부터 내려가며 연속으로 통과한 마지막 단계. 100% 도 실패면 None."""
    last = None
    for lv in COVERAGE_LEVELS:
        if not rule(agg[(node, method, lv)]):
            break
        last = lv
    return last


def fmt_cov(v) -> str:
    return "없음" if v is None else f"{v}%"


def run_partial(ctx: Ctx, out: str, out_windows: str, md: str | None) -> None:
    X = ctx.X
    n_win, _, _, n_frames = X.shape
    base_recall_k = int((ctx.base[ctx.is_fall] >= THRESHOLD).sum())
    print(f"\n[partial] 기준 재현율@{THRESHOLD:.2f} {base_recall_k}/{int(ctx.is_fall.sum())} · "
          f"빈 방 최댓값 {ctx.base[ctx.is_empty].max():.4f} · "
          f"기준 미발화 빈 방 {int((ctx.is_empty & (ctx.base < THRESHOLD)).sum())}창 "
          f"최댓값 {ctx.base[ctx.is_empty & (ctx.base < THRESHOLD)].max():.4f}")

    buf = X.copy()                          # 한 노드 슬롯만 바꿨다가 되돌려 쓴다
    records: List[dict] = []
    window_scores: dict = {"base": ctx.base}
    window_cols: List[str] = ["base"]
    method_idx = {m: i for i, (m, _, _) in enumerate(MASK_METHODS)}
    total = sum(len(PARTIAL_SEEDS) if rnd else 1 for _, _, rnd in MASK_METHODS) * 3 * \
        sum(1 for lv in COVERAGE_LEVELS if lv < 100)
    done = 0

    for node in range(3):
        orig_cov = ctx.present[:, node, :].mean(axis=1)
        for method, _, is_random in MASK_METHODS:
            for lv in COVERAGE_LEVELS:
                keep_frames = int(round(lv * n_frames / 100))
                if keep_frames == n_frames:
                    # 100% = 추가 결측 없음 = 기준. 추론을 다시 하지 않는다.
                    m = partial_metrics(ctx, ctx.base, base_recall_k)
                    records.append({"node": node + 1, "method": method, "coverage_pct": lv, "seed": "",
                                    "cov_mean": float(orig_cov.mean()), "cov_min": float(orig_cov.min()), **m})
                    continue
                for seed in (PARTIAL_SEEDS if is_random else [None]):
                    rng = np.random.default_rng([0 if seed is None else seed, node, lv, method_idx[method]])
                    keep = keep_mask(method, n_win, n_frames, keep_frames, rng)
                    buf[:, node] = X[:, node] * keep[:, None, :]
                    s = infer(ctx.model, buf, ctx.bs)
                    buf[:, node] = X[:, node]
                    actual = (ctx.present[:, node, :] & keep).mean(axis=1)
                    m = partial_metrics(ctx, s, base_recall_k)
                    records.append({"node": node + 1, "method": method, "coverage_pct": lv,
                                    "seed": "" if seed is None else seed,
                                    "cov_mean": float(actual.mean()), "cov_min": float(actual.min()), **m})
                    if seed is None or seed == PARTIAL_SEEDS[0]:
                        col = f"n{node + 1}_{method}_c{lv}"
                        window_scores[col] = s
                        window_cols.append(col)
                    done += 1
                    print(f"  [{done:3d}/{total}] 노드{node + 1} {method:<10} {lv:3d}% seed={seed} "
                          f"재현율 {m['recall_k']:3d}/{m['recall_n']} 빈방max {m['empty_max']:.4f} "
                          f"미발화max {m['empty_quiet_max']:.4f} |Δ|med {m['d_med']:.4f} "
                          f"실측cov {actual.mean() * 100:.1f}%", flush=True)
    del buf

    # ── 요약 CSV (시드별 전부) ──────────────────────────────────────────────────
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["node", "method", "coverage_pct", "seed", "cov_mean", "cov_min", "pr_auc",
            "recall_k", "recall_n", "recall", "recall_drop", "empty_fire_k", "empty_n", "empty_max",
            "empty_quiet_n", "empty_quiet_max", "empty_new_fire_k", "d_med", "d_p95", "d_fall_med", "d_empty_med"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in records:
            w.writerow([f"{r[c]:.6f}" if isinstance(r[c], float) else r[c] for c in cols])
    print(f"\n[out] 조건 요약 {len(records)}행 -> {out_path}")
    write_window_csv(ctx, out_windows, window_cols, window_scores)

    # ── 집계 ─────────────────────────────────────────────────────────────────
    agg = {}
    for node in (1, 2, 3):
        for method, _, _ in MASK_METHODS:
            for lv in COVERAGE_LEVELS:
                recs = [r for r in records if r["node"] == node and r["method"] == method
                        and r["coverage_pct"] == lv]
                agg[(node, method, lv)] = aggregate(recs)

    lines: List[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)

    for method, title, is_random in MASK_METHODS:
        emit(f"### {title}")
        emit()
        if is_random:
            emit(f"무작위 결측이라 시드 {len(PARTIAL_SEEDS)}개로 각각 쟀다. **표의 값은 시드 중 최악값**이다 "
                 "(pr_auc·재현율은 최소, 하락·발화·최댓값·변화량은 최대).")
            emit()
        for node in (1, 2, 3):
            emit(f"**노드{node}**")
            emit()
            emit("| 커버리지 | 실측 커버리지 평균 / 최소 | pr_auc | 재현율 @0.80 | 하락 | 빈 방 발화 "
                 "| 빈 방 최댓값 (238창) | 빈 방 최댓값 (기준 미발화 235창) | \\|Δ\\| 중앙값 | 판정 |")
            emit("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for lv in COVERAGE_LEVELS:
                a = agg[(node, method, lv)]
                ok = passes(a)
                emit(f"| {lv}% | {a['cov_mean'] * 100:.1f}% / {a['cov_min'] * 100:.0f}% | {a['pr_auc']:.4f} "
                     f"| {a['recall_k']}/{a['recall_n']} = {a['recall_k'] / a['recall_n']:.3f} "
                     f"| {a['recall_drop']:.3f} | {a['empty_fire_k']}/{a['empty_n']} "
                     f"| {a['empty_max']:.4f} | {a['empty_quiet_max']:.4f} | {a['d_med']:.4f} "
                     f"| {'통과' if ok else '**실패**'} |")
            emit()

    # 판정
    emit("### 판정 — 최소 커버리지")
    emit()
    emit("| 노드 | (a) 무작위 분산 | (b) 연속 구간 | 노드 값 = max(a,b) | (보조) b-앞 | (보조) b-끝 "
         "| (원문 기준) a | (원문 기준) b |")
    emit("|---|---:|---:|---:|---:|---:|---:|---:|")
    node_vals = []
    for node in (1, 2, 3):
        va = min_coverage(agg, node, "scatter", passes)
        vb = min_coverage(agg, node, "block", passes)
        vals = [va, vb]
        nv = None if any(v is None for v in vals) else max(vals)
        node_vals.append(nv)
        emit(f"| 노드{node} | {fmt_cov(va)} | {fmt_cov(vb)} | **{fmt_cov(nv)}** "
             f"| {fmt_cov(min_coverage(agg, node, 'block_head', passes))} "
             f"| {fmt_cov(min_coverage(agg, node, 'block_tail', passes))} "
             f"| {fmt_cov(min_coverage(agg, node, 'scatter', passes_literal))} "
             f"| {fmt_cov(min_coverage(agg, node, 'block', passes_literal))} |")
    rec = None if any(v is None for v in node_vals) else max(node_vals)
    emit()
    emit(f"권고치 (세 노드 중 가장 보수적) = **{fmt_cov(rec)}**")
    tail_vals = [min_coverage(agg, n, "block_tail", passes) for n in (1, 2, 3)]
    head_vals = [min_coverage(agg, n, "block_head", passes) for n in (1, 2, 3)]
    emit(f"(보조) b-끝만으로 같은 규칙 = **{fmt_cov(None if None in tail_vals else max(tail_vals))}** · "
         f"b-앞 = **{fmt_cov(None if None in head_vals else max(head_vals))}**")
    emit()

    # (a) vs (b)
    emit("### (a) 무작위 분산 vs (b) 연속 구간 — 같은 커버리지에서의 차이 (시드 평균)")
    emit()
    emit("| 노드 | 커버리지 | 재현율 (a) | 재현율 (b) | 차 (a−b) | \\|Δ\\| 중앙값 (a) | \\|Δ\\| 중앙값 (b) "
         "| 차 (a−b) | 미발화 빈 방 최댓값 (a) | (b) | 차 ≥ 0.05 |")
    emit("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for node in (1, 2, 3):
        for lv in COVERAGE_LEVELS:
            if lv == 100:
                continue
            a, b = agg[(node, "scatter", lv)], agg[(node, "block", lv)]
            dr = a["recall_mean"] - b["recall_mean"]
            dd = a["d_med_mean"] - b["d_med_mean"]
            big = abs(dr) >= BIG_DIFF or abs(dd) >= BIG_DIFF
            emit(f"| 노드{node} | {lv}% | {a['recall_mean']:.3f} | {b['recall_mean']:.3f} | {dr:+.3f} "
                 f"| {a['d_med_mean']:.4f} | {b['d_med_mean']:.4f} | {dd:+.4f} "
                 f"| {a['empty_quiet_max']:.4f} | {b['empty_quiet_max']:.4f} | {'**예**' if big else '아니오'} |")
    emit()

    # 시드 변동
    emit("### 시드 간 변동 (무작위 방식, 시드 최소 ~ 최대)")
    emit()
    emit("| 노드 | 방식 | 커버리지 | 재현율 k | 빈 방 최댓값 (238창) | 미발화 빈 방 최댓값 | \\|Δ\\| 중앙값 |")
    emit("|---|---|---:|---:|---:|---:|---:|")
    for node in (1, 2, 3):
        for method in JUDGE_METHODS:
            for lv in COVERAGE_LEVELS:
                if lv == 100:
                    continue
                a = agg[(node, method, lv)]
                emit(f"| 노드{node} | {method} | {lv}% | {a['recall_k']} ~ {a['recall_k_hi']} "
                     f"| {a['empty_max_lo']:.4f} ~ {a['empty_max']:.4f} "
                     f"| {a['empty_quiet_max_lo']:.4f} ~ {a['empty_quiet_max']:.4f} "
                     f"| {a['d_med_lo']:.4f} ~ {a['d_med']:.4f} |")
    emit()

    text = "\n".join(lines)
    print("\n" + text)
    if md:
        Path(md).parent.mkdir(parents=True, exist_ok=True)
        Path(md).write_text(text, encoding="utf-8")
        print(f"[out] markdown 표 -> {md}")


# ── tail: 창 끝 L 프레임만 0 ─────────────────────────────────────────────────────────

def natural_tail_run(present: np.ndarray) -> np.ndarray:
    """(N, nodes, T) → (N, nodes) 원본에서 창 끝부터 연속으로 비어 있던 프레임 수."""
    rev = present[:, :, ::-1]                       # 인덱스 0 = 창 마지막 프레임
    first_present = np.argmax(rev, axis=2)
    return np.where(rev.any(axis=2), first_present, present.shape[2])


def run_tail(ctx: Ctx, out: str, md: str | None, partial_windows: str | None) -> None:
    X = ctx.X
    n_win, n_nodes, _, n_frames = X.shape
    base_recall_k = int((ctx.base[ctx.is_fall] >= THRESHOLD).sum())
    quiet = ctx.is_empty & (ctx.base < THRESHOLD)
    print(f"\n[tail] 기준 재현율@{THRESHOLD:.2f} {base_recall_k}/{int(ctx.is_fall.sum())} · "
          f"기준 미발화 빈 방 {int(quiet.sum())}창 최댓값 {ctx.base[quiet].max():.4f}")

    buf = X.copy()
    records: List[dict] = []
    for node in range(n_nodes):
        for gap in [0] + TAIL_GAPS:
            if gap == 0:
                s = ctx.base
                changed = 0
            else:
                keep = keep_mask("block_tail", n_win, n_frames, n_frames - gap, None)
                buf[:, node] = X[:, node] * keep[:, None, :]
                s = infer(ctx.model, buf, ctx.bs)
                buf[:, node] = X[:, node]
                # 원본에서 이미 끝 gap 프레임이 전부 비어 있던 창은 입력이 바뀌지 않는다
                changed = int(ctx.present[:, node, n_frames - gap:].any(axis=1).sum())
            m = partial_metrics(ctx, s, base_recall_k)
            m["pass"] = passes(m)
            records.append({"node": node + 1, "gap_frames": gap, "gap_ms": gap * 10,
                            "changed_windows": changed, **m})
            print(f"  노드{node + 1} 공백 {gap:2d}프레임 ({gap * 10:3d} ms) 재현율 {m['recall_k']:3d}/{m['recall_n']} "
                  f"하락 {m['recall_drop']:+.4f} 미발화max {m['empty_quiet_max']:.4f} "
                  f"신규발화 {m['empty_new_fire_k']} |Δ|med {m['d_med']:.4f} "
                  f"{'통과' if m['pass'] else '실패'} (입력 바뀐 창 {changed})", flush=True)
    del buf

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["node", "gap_frames", "gap_ms", "changed_windows", "pr_auc", "recall_k", "recall_n", "recall",
            "recall_drop", "empty_fire_k", "empty_n", "empty_max", "empty_quiet_n", "empty_quiet_max",
            "empty_new_fire_k", "d_med", "d_p95", "d_fall_med", "d_empty_med", "pass"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in records:
            w.writerow([f"{r[c]:.6f}" if isinstance(r[c], float) else r[c] for c in cols])
    print(f"\n[out] 조건 요약 {len(records)}행 -> {out_path}")

    # ── 판정 ─────────────────────────────────────────────────────────────────
    by = {(r["node"], r["gap_frames"]): r for r in records}
    safe, first_fail, later_pass = {}, {}, {}
    for node in range(1, n_nodes + 1):
        last = 0
        for gap in TAIL_GAPS:
            if not by[(node, gap)]["pass"]:
                break
            last = gap
        safe[node] = last
        first_fail[node] = next((g for g in TAIL_GAPS if not by[(node, g)]["pass"]), None)
        later_pass[node] = [g for g in TAIL_GAPS if first_fail[node] is not None
                            and g > first_fail[node] and by[(node, g)]["pass"]]
    min_safe = min(safe.values())
    rec_n = min_safe + 1

    # 원본 검증 창에 규칙을 적용하면 얼마나 생략되는가 (추론 없음)
    run = natural_tail_run(ctx.present)                      # (N, nodes)
    last_present = ctx.present[:, :, ::-1]                   # 인덱스 0 = 마지막 프레임

    def skip_rate(n: int) -> tuple:
        miss_any = ~last_present[:, :, :n].all(axis=2)        # (N, nodes) 끝 n 프레임 중 결측 있음
        return miss_any.mean(axis=0), miss_any.any(axis=1).mean()

    lines: List[str] = []
    emit = lines.append

    emit("### 노드별 결과")
    emit("")
    for node in range(1, n_nodes + 1):
        emit(f"**노드{node}**")
        emit("")
        emit("| 창 끝 공백 | 입력이 바뀐 창 | 재현율 @0.80 | 하락 | 기준 미발화 빈 방 235창 최댓값 | 신규 발화 창 "
             "| \\|Δ\\| 중앙값 | 판정 |")
        emit("|---:|---:|---:|---:|---:|---:|---:|---|")
        for gap in [0] + TAIL_GAPS:
            r = by[(node, gap)]
            label = "0 (기준)" if gap == 0 else f"{gap} ({gap * 10} ms)"
            chg = "—" if gap == 0 else f"{r['changed_windows']}/{n_win}"
            verdict = "—" if gap == 0 else ("통과" if r["pass"] else "**실패**")
            emit(f"| {label} | {chg} | {r['recall_k']}/{r['recall_n']} = {r['recall']:.3f} "
                 f"| {r['recall_drop']:.3f} | {r['empty_quiet_max']:.4f} | {r['empty_new_fire_k']} "
                 f"| {r['d_med']:.4f} | {verdict} |")
        emit("")

    emit("### 판정")
    emit("")
    emit("| 노드 | 허용 공백 L_safe (연속 통과) | 처음 실패한 길이 | 첫 실패 이후 다시 통과한 길이 |")
    emit("|---|---:|---:|---|")
    for node in range(1, n_nodes + 1):
        ff = first_fail[node]
        emit(f"| 노드{node} | {safe[node]} 프레임 ({safe[node] * 10} ms) "
             f"| {'없음' if ff is None else f'{ff} 프레임 ({ff * 10} ms)'} "
             f"| {', '.join(str(g) for g in later_pass[node]) or '없음'} |")
    emit("")
    emit(f"min(L_safe) = {min_safe} → **권고 N = {rec_n}**")
    emit("")

    emit("### 원본 검증 창의 창 끝 결측 (추론 없음)")
    emit("")
    emit("| 창 끝 N 프레임 안에 결측 있음 | 노드1 | 노드2 | 노드3 | 한 노드라도 (= 규칙 적용 시 생략) |")
    emit("|---:|---:|---:|---:|---:|")
    for n in sorted({1, 2, 3, 5, 10, 19, rec_n}):
        per, anyn = skip_rate(n)
        emit(f"| {n} | {per[0] * 100:.1f}% | {per[1] * 100:.1f}% | {per[2] * 100:.1f}% | {anyn * 100:.1f}% |")
    emit("")
    emit("원본 창 끝 연속 결측 길이 분포 (노드별 창 수):")
    emit("")
    emit("| 창 끝 연속 결측 | 노드1 | 노드2 | 노드3 |")
    emit("|---:|---:|---:|---:|")
    for lo, hi, label in [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 5, "3~5"), (6, 10, "6~10"),
                          (11, 19, "11~19"), (20, n_frames, "20 이상")]:
        cnt = [int(((run[:, k] >= lo) & (run[:, k] <= hi)).sum()) for k in range(n_nodes)]
        emit(f"| {label} | {cnt[0]} | {cnt[1]} | {cnt[2]} |")
    emit("")

    # ── (보조) 규칙 통과 창 안에서의 partial 결과 ─────────────────────────────────
    def allowed_mask(present_after: np.ndarray) -> np.ndarray:
        """(N, nodes, T) 실측 존재 여부 → 규칙 통과 창 (모든 노드가 창 끝 rec_n 프레임에 실프레임)."""
        return present_after[:, :, n_frames - rec_n:].all(axis=2).all(axis=1)

    base_allowed = allowed_mask(ctx.present)
    f, e = ctx.is_fall, ctx.is_empty
    emit("### 규칙을 원본 검증 창에 적용하면 (추론 없음, 기준 점수)")
    emit("")
    emit("| 창 묶음 | 창 수 | 낙상 라벨 창 | 재현율 @0.80 | 빈 방 창 | 빈 방 발화 @0.80 |")
    emit("|---|---:|---:|---:|---:|---:|")
    for name, sel in (("전체", np.ones(n_win, bool)), ("규칙 통과 (추론)", base_allowed),
                      ("규칙 생략", ~base_allowed)):
        nf, ne = int((sel & f).sum()), int((sel & e).sum())
        kf = int((ctx.base[sel & f] >= THRESHOLD).sum())
        ke = int((ctx.base[sel & e] >= THRESHOLD).sum())
        emit(f"| {name} | {int(sel.sum())} | {nf} | {kf}/{nf} = {kf / max(nf, 1):.3f} "
             f"| {ne} | {ke}/{ne} = {100 * ke / max(ne, 1):.1f}% |")
    emit("")

    win_csv = Path(partial_windows) if partial_windows else None
    if win_csv is not None and win_csv.exists():
        with win_csv.open(encoding="utf-8") as fh:
            prow = list(csv.DictReader(fh))
        base_csv = np.array([float(r["base"]) for r in prow])
        lab_csv = np.array([int(r["label"]) for r in prow])
        aligned = (len(prow) == n_win and np.all(lab_csv == ctx.labels)
                   and np.max(np.abs(base_csv - ctx.base)) < 1e-5
                   and all(r["csv_file"] == g for r, g in zip(prow, ctx.groups)))
        print(f"[tail] partial 창 단위 CSV {win_csv} 정렬 확인: {'일치' if aligned else '불일치 — 보조 분석 생략'}")
        if aligned:
            method_idx = {m: i for i, (m, _, _) in enumerate(MASK_METHODS)}
            emit(f"### (보조) 규칙 통과 창만 골라 본 §8 결과 — 첫 시드 {PARTIAL_SEEDS[0]}, 창 끝 {rec_n}프레임 기준")
            emit("")
            emit("| 노드 | 방식 | 커버리지 | 규칙 통과 창 | 낙상 창 | 재현율 기준 → 조건 | 하락 "
                 "| 미발화 빈 방 창 | 그 최댓값 | 신규 발화 | \\|Δ\\| 중앙값 | 판정 |")
            emit("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for node in range(n_nodes):
                for method in JUDGE_METHODS:
                    for lv in COVERAGE_LEVELS:
                        if lv == 100:
                            continue
                        rng = np.random.default_rng([PARTIAL_SEEDS[0], node, lv, method_idx[method]])
                        keep = keep_mask(method, n_win, n_frames, int(round(lv * n_frames / 100)), rng)
                        after = ctx.present.copy()
                        after[:, node, :] &= keep
                        sel = allowed_mask(after)
                        s = np.array([float(r[f"n{node + 1}_{method}_c{lv}"]) for r in prow])
                        sf, sq = sel & f, sel & quiet
                        nf = int(sf.sum())
                        kb = int((ctx.base[sf] >= THRESHOLD).sum())
                        kc = int((s[sf] >= THRESHOLD).sum())
                        drop = (kb - kc) / nf if nf else float("nan")
                        qmax = float(s[sq].max()) if sq.any() else float("nan")
                        newf = int((s[sq] >= THRESHOLD).sum())
                        dmed = float(np.median(np.abs(s[sel] - ctx.base[sel]))) if sel.any() else float("nan")
                        ok = nf > 0 and sq.any() and drop < PARTIAL_RECALL_DROP_MAX and qmax < THRESHOLD
                        emit(f"| 노드{node + 1} | {'(a)' if method == 'scatter' else '(b)'} | {lv}% "
                             f"| {int(sel.sum())} | {nf} | {kb} → {kc} | {drop:.3f} | {int(sq.sum())} "
                             f"| {qmax:.4f} | {newf} | {dmed:.4f} | {'통과' if ok else '**실패**'} |")
            emit("")
    else:
        print(f"[tail] partial 창 단위 CSV 가 없어 보조 분석 생략: {win_csv}")

    text = "\n".join(lines)
    print("\n" + text)
    print(f"\n[판정] 규칙: 창 끝 {rec_n}프레임 안에 결측이 있으면 추론 생략")
    if md:
        Path(md).parent.mkdir(parents=True, exist_ok=True)
        Path(md).write_text(text, encoding="utf-8")
        print(f"[out] markdown 표 -> {md}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "partial", "tail"], default="full")
    ap.add_argument("--config", default="configs/m1_fall.yaml")
    ap.add_argument("--ckpt", default="runs/m1_base_3node_20260916.pt")
    ap.add_argument("--out", default=None,
                    help="full: 창 단위 CSV (기본 out/eval_node_dropout.csv) · "
                         "partial: 조건 요약 CSV (기본 out/eval_node_coverage.csv) · "
                         "tail: 조건 요약 CSV (기본 out/eval_node_tail_gap.csv)")
    ap.add_argument("--out-windows", default="out/eval_node_coverage_windows.csv",
                    help="partial: 창 단위 점수 CSV 를 쓸 경로 (무작위 방식은 첫 시드만) · "
                         "tail: 보조 분석에서 읽을 경로")
    ap.add_argument("--md", default=None, help="partial·tail: 문서용 markdown 표를 이 경로에도 쓴다")
    a = ap.parse_args()

    ctx = load_context(a.config, a.ckpt)
    if a.mode == "full":
        run_full(ctx, a.out or "out/eval_node_dropout.csv")
    elif a.mode == "partial":
        run_partial(ctx, a.out or "out/eval_node_coverage.csv", a.out_windows, a.md)
    else:
        # tail 의 보조 분석은 partial 이 남긴 창 단위 점수를 읽는다 (--out-windows 와 같은 파일)
        run_tail(ctx, a.out or "out/eval_node_tail_gap.csv", a.md, a.out_windows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
