#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_hardneg.py — 학습된 M1 모델의 그룹별 오탐률·재현율을 잰다.

학습은 하지 않는다. 추론과 집계만 한다.

핵심 원칙
  1. 창을 직접 만들지 않는다. `m1_fall.dataset.build_split`을 그대로 쓴다.
     전처리(epoch 격자 정렬·노드별 per-frame peak 정규화·guard 0·창 나누기)가
     학습과 한 글자라도 다르면 나온 숫자는 무효다.
  2. 체크포인트는 strict=True 로만 로드한다. strict=False 는 키가 안 맞아도
     조용히 통과해서 "사전학습 효과 없음" 같은 오진을 만든다.
  3. 모델 출력은 pre-sigmoid logit 이다. 확률로 읽으려면 여기서 sigmoid 를
     한 번 적용해야 한다 (ONNX 그래프에는 sigmoid 가 없다 — 규칙 2).
  4. configs/m1_fall.yaml 파일 자체는 건드리지 않는다. 라벨 경로만 메모리에서
     덮어쓴다.

주의: 이 스크립트는 rp5 가 아니라 학습 PC 에서 도는 것이라 torch/numpy 를 쓴다.
      `scripts/` 의 표준 라이브러리 전용 규칙은 rp5 에서도 돌아야 하는
      mark_falls · verify_sync_v2 · session_report 셋에만 해당한다.

사용
    $env:PYTHONPATH="src"
    .venv\Scripts\python.exe scripts\eval_hardneg.py --labels data/labels_eval.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# 창 태그 -> 화면에 찍을 순서. 파일명 끝의 행동 태그를 그대로 쓴다.
TAG_ORDER = [
    "hardneg_chair", "hardneg_floor", "hardneg_liedown",
    "normal_active", "normal_empty",
    "fall", "fall_longlie",
]
FALL_TAGS = {"fall", "fall_longlie"}       # 재현율로 읽는 그룹


def tag_of(csv_file: str) -> str:
    """'csi_20260916_1331_hardneg_chair.csv' -> 'hardneg_chair'"""
    parts = Path(csv_file).stem.split("_")   # csi / 20260916 / 1331 / hardneg / chair
    return "_".join(parts[3:]) if len(parts) > 3 else "untagged"


def load_checkpoint(model, ckpt_path: Path):
    """strict=True 로만 로드한다. 로드된 키 수를 찍어 증거를 남긴다."""
    import torch
    if not ckpt_path.exists():
        sys.exit(f"체크포인트가 없다: {ckpt_path}")
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model.load_state_dict(state, strict=True)    # strict=False 금지
    epoch = blob.get("epoch") if isinstance(blob, dict) else "?"
    print(f"[ckpt] {ckpt_path} · strict=True 로드 성공 · 키 {len(state)}개 · epoch {epoch}")
    if isinstance(blob, dict) and isinstance(blob.get("metrics"), dict):
        got = " ".join(f"{k}={float(v):.4f}" for k, v in blob["metrics"].items()
                       if isinstance(v, (int, float)) and k != "epoch")
        print(f"[ckpt] 학습 당시 val 지표: {got}")
    return model


def pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def summarize(scores: np.ndarray) -> dict:
    if len(scores) == 0:
        return {"n": 0, "ge50": float("nan"), "ge70": float("nan"),
                "median": float("nan"), "p90": float("nan")}
    return {
        "n": len(scores),
        "ge50": float(np.mean(scores >= 0.5)),
        "ge70": float(np.mean(scores >= 0.7)),
        "median": float(np.median(scores)),
        "p90": float(np.percentile(scores, 90)),
    }


def print_table(title: str, reading: str, rows: list) -> None:
    print(f"\n{title}")
    print(f"{'그룹':<16}{'창 수':>7}{'>=0.5':>9}{'>=0.7':>9}{'중앙값':>9}{'p90':>9}   판독")
    print("-" * 74)
    for name, s in rows:
        if s["n"] == 0:
            print(f"{name:<16}{0:>7}{'-':>9}{'-':>9}{'-':>9}{'-':>9}   (창 없음)")
            continue
        print(f"{name:<16}{s['n']:>7}{pct(s['ge50']):>9}{pct(s['ge70']):>9}"
              f"{s['median']:>9.3f}{s['p90']:>9.3f}   {reading}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m1_fall.yaml")
    ap.add_argument("--labels", default="data/labels_eval.csv",
                    help="평가용 라벨 CSV. yaml 의 paths.labels_csv 를 메모리에서 덮어쓴다.")
    ap.add_argument("--ckpt", default="runs/best.pt")
    ap.add_argument("--out", default="out/eval_hardneg.csv")
    a = ap.parse_args()

    import torch
    from m1_fall.config import load_config
    from m1_fall.dataset import build_split
    from m1_fall.model import build_model

    cfg = load_config(a.config)
    # yaml 파일은 수정하지 않는다. 메모리 위에서만 라벨 경로를 바꾼다.
    original = cfg.paths.labels_csv
    cfg.paths.labels_csv = a.labels
    print(f"[cfg] {a.config} · labels_csv: {original} -> {a.labels} (메모리에서만 교체)")
    print(f"[cfg] n_nodes={cfg.tensor.n_nodes} · raw_dir={cfg.paths.raw_dir} "
          f"· stride={cfg.window.stride_frames} · fall_stride={cfg.window.fall_stride_frames} "
          f"· halfwidth={cfg.label.fall_halfwidth_ms}ms · pos_frac={cfg.label.fall_pos_frac}")

    # 학습과 동일한 로더·전처리. 여기서 새로 만들면 숫자가 무효다.
    X, y, groups = build_split(cfg, "val")
    if len(X) == 0:
        sys.exit("창이 0개다. labels_eval.csv 의 split 이 전부 val 인지, "
                 "raw_dir 에 파일이 있는지 확인해라.")
    print(f"[data] 창 {len(X)}개 {tuple(X.shape[1:])} · 파일 {len(set(groups))}개 "
          f"· 양성(낙상) 라벨 {int(y.sum())}개")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}" + (f" -> {torch.cuda.get_device_name(0)}" if device == "cuda" else ""))

    model = build_model(cfg.model, cfg.tensor.n_channels, cfg.tensor.window_frames,
                        cfg.tensor.n_nodes)
    model = load_checkpoint(model, Path(a.ckpt)).to(device).eval()

    # 추론. 모델 출력은 pre-sigmoid logit 이므로 sigmoid 를 여기서 적용한다.
    scores = np.empty(len(X), dtype=np.float64)
    bs = int(cfg.train.batch_size)
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(device)
            logit = model(xb)                       # (B,1) pre-sigmoid
            prob = torch.sigmoid(logit)             # <-- 확률로 바꾸는 유일한 지점
            scores[i:i + bs] = prob.cpu().numpy().reshape(-1)
    print(f"[infer] sigmoid 적용 완료 · 점수 범위 {scores.min():.4f} ~ {scores.max():.4f}")

    tags = np.array([tag_of(g) for g in groups])
    files = np.array(groups)
    labels = y.astype(int)

    # 창 단위 결과 저장
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_file_idx: dict = {}
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["csv_file", "tag", "window_idx", "score", "label"])
        for f, t, s, lab in zip(files, tags, scores, labels):
            idx = per_file_idx.get(f, 0)
            per_file_idx[f] = idx + 1
            w.writerow([f, t, idx, f"{s:.6f}", int(lab)])
    print(f"[out] 창 단위 결과 {len(scores)}행 -> {out_path}")

    seen = [t for t in TAG_ORDER if (tags == t).any()]
    extra = sorted(set(tags.tolist()) - set(TAG_ORDER))
    if extra:
        print(f"[warn] TAG_ORDER 에 없는 태그: {extra} (표 맨 뒤에 붙인다)")
    seen += extra

    # 표 1 — 오탐률: 라벨 0(비낙상) 창만. 낙상 파일 안의 비낙상 구간도 여기 포함된다.
    neg_rows = [(t, summarize(scores[(tags == t) & (labels == 0)])) for t in seen]
    print_table("[표 1] 오탐률 — 라벨 0(비낙상) 창을 낙상이라고 한 비율 (낮을수록 좋다)",
                "오탐률", neg_rows)

    # 표 2 — 재현율: 라벨 1(낙상) 창만.
    pos_rows = [(t, summarize(scores[(tags == t) & (labels == 1)])) for t in seen]
    pos_rows = [(t, s) for t, s in pos_rows if s["n"] > 0]
    if pos_rows:
        print_table("[표 2] 재현율 — 라벨 1(낙상) 창을 낙상이라고 맞힌 비율 (높을수록 좋다)",
                    "재현율", pos_rows)

    # 표 3 — 파일별 (어느 세션이 문제인지 보려고)
    print("\n[표 3] 파일별")
    print(f"{'파일':<42}{'태그':<16}{'창':>5}{'양성':>5}{'>=0.7':>9}{'중앙값':>9}")
    print("-" * 90)
    for f in sorted(set(files.tolist())):
        sel = files == f
        s = summarize(scores[sel])
        print(f"{f:<42}{tag_of(f):<16}{s['n']:>5}{int(labels[sel].sum()):>5}"
              f"{pct(s['ge70']):>9}{s['median']:>9.3f}")

    print("\n주의: 위 숫자는 1인·1실·1일 데이터에서 나온 것이다. "
          "피험자·환경 일반화는 검증되지 않았다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
