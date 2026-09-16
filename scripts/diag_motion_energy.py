#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_motion_energy.py — "모델이 낙상이 아니라 움직임 강도를 본다" 가설 진단.

학습하지 않는다. 창을 다시 만들어 움직임 에너지를 재고, 이미 측정해 둔 모델
점수(out/eval_hardneg.csv)와 붙여서 비교만 한다.

검증 대상
  H1  모델 점수가 창의 움직임 강도와 강하게 상관하는가
  H2  움직임 강도 하나만으로 분류해도 모델과 비슷한 PR-AUC 가 나오는가
      (= 13만 파라미터가 한 줄 수식 이상을 못 하고 있는가)
  H3  눕기의 발동률이 낮은 게 모델이 걸러서가 아니라 실제로 천천히 누워서인가

지켜야 할 것
  1. 창은 `m1_fall.dataset.build_split` 으로 만든다. eval_hardneg.py 와 같은
     로더·같은 순서여야 점수와 짝이 맞는다. 창 수가 다르면 중단한다.
  2. PR-AUC 는 train.py 가 쓰는 것과 **같은 함수**를 쓴다
     (`sklearn.metrics.average_precision_score` — src/m1_fall/train.py:222).
  3. 에너지 정의 세 개를 전부 계산하고 전부 보고한다. 가설에 맞는 것만
     골라 적지 않는다.
  4. configs/m1_fall.yaml 파일은 건드리지 않는다. 라벨 경로만 메모리에서 바꾼다.

측정 도구 자체의 타당성 검사도 같이 한다. 로더가 결측 노드 슬롯을 0 으로
채우기 때문에 "실프레임 -> 0" 전환이 큰 변화량으로 잡힐 수 있다. 그래서
창마다 결측률(zero_frac)을 같이 재고, 에너지가 움직임이 아니라 결측을
재고 있는지 확인한다. 판정은 처음 정한 세 정의로 하고, 보정 지표(e_valid)는
그 판정을 어떻게 읽어야 하는지 밝히는 용도로만 쓴다.

사용
    $env:PYTHONPATH="src"
    .venv\Scripts\python.exe scripts\diag_motion_energy.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

TAG_ORDER = [
    "fall", "fall_longlie",
    "hardneg_liedown", "hardneg_chair", "hardneg_floor",
    "normal_active", "normal_empty",
]

# 측정 전에 못박아 둔 판정 기준. 숫자를 보고 나서 바꾸지 않는다.
H1_ACCEPT, H1_REJECT = 0.8, 0.5      # |Spearman| 0.8 이상 채택 / 0.5 미만 기각
H2_MODEL_PR_AUC = 0.472              # 학습 당시 보고된 모델 val PR-AUC
H2_ACCEPT_RATIO = 0.90               # 그 90% 이상(>=0.425)이면 채택
H3_ACCEPT_RATIO = 0.70               # 눕기 e_max 중앙값 < 낙상의 70% 면 채택

# 에너지가 결측을 재고 있다고 볼 상관 기준 (도구 타당성 검사)
CONFOUND_WARN = 0.7


def tag_of(csv_file: str) -> str:
    """'csi_20260916_1331_hardneg_chair.csv' -> 'hardneg_chair'"""
    parts = Path(csv_file).stem.split("_")
    return "_".join(parts[3:]) if len(parts) > 3 else "untagged"


def motion_energy(win: np.ndarray) -> tuple:
    """창 하나 (n_nodes, 64, 100) -> (e_mean, e_max, e_std, e_valid, zero_frac).

    시간축은 마지막이다. raw 는 ESP 에서 프레임별 peak 정규화가 걸려 있어
    절대 진폭이 이미 사라졌으므로, 프레임 간 변화량을 움직임의 대리 지표로 쓴다.

    앞의 세 개가 처음에 정한 정의이고 결과는 전부 그대로 보고한다.
      zero_frac  창 안에서 통째로 0 인 (노드, 프레임) 슬롯의 비율 = 결측률
      e_valid    양쪽 프레임이 모두 살아 있는 쌍만 골라 낸 평균 변화량
    e_valid 는 정의를 골라 쓰려고 넣은 게 아니라, e_mean 이 움직임이 아니라
    결측을 재고 있는지 확인하려고 넣은 것이다.
    """
    d = np.abs(np.diff(win, axis=-1))            # (노드, 부반송파, 99) 프레임 간 변화량
    e_mean = float(d.mean())                     # 평균 움직임
    e_max = float(d.mean(axis=(0, 1)).max())     # 프레임별 평균 중 최댓값 = 순간 최대
    e_std = float(win.std(axis=-1).mean())       # 시간축 표준편차의 노드·부반송파 평균

    present = np.abs(win).sum(axis=1) > 0        # (노드, 프레임) 실제 프레임이 있나
    zero_frac = float(1.0 - present.mean())
    pair_ok = present[:, 1:] & present[:, :-1]   # 양쪽이 다 살아 있는 프레임 쌍
    e_valid = float(d.mean(axis=1)[pair_ok].mean()) if pair_ok.any() else float("nan")
    return e_mean, e_max, e_std, e_valid, zero_frac


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """순위 상관. scipy 가 있으면 쓰고, 없으면 순위를 만들어 피어슨으로 낸다."""
    if len(a) < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).statistic)
    except ImportError:
        return float(np.corrcoef(_rankdata(a), _rankdata(b))[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    """동점을 평균 순위로 처리하는 순위 매기기 (scipy 없을 때용)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(xs):                            # 동점 구간을 평균 순위로 눌러준다
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """train.py 와 동일한 정의 (src/m1_fall/train.py:222). 양쪽 클래스가 있어야 한다."""
    from sklearn.metrics import average_precision_score
    if len(np.unique(labels)) != 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def q(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p)) if len(x) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m1_fall.yaml")
    ap.add_argument("--labels", default="data/labels_eval.csv")
    ap.add_argument("--scores", default="out/eval_hardneg.csv")
    ap.add_argument("--out", default="out/diag_motion_energy.csv")
    a = ap.parse_args()

    from m1_fall.config import load_config
    from m1_fall.dataset import build_split

    cfg = load_config(a.config)
    original = cfg.paths.labels_csv
    cfg.paths.labels_csv = a.labels              # yaml 파일은 그대로 둔다
    print(f"[cfg] labels_csv: {original} -> {a.labels} (메모리에서만 교체)")
    print(f"[cfg] n_nodes={cfg.tensor.n_nodes} · stride={cfg.window.stride_frames} "
          f"· fall_stride={cfg.window.fall_stride_frames}")

    # eval_hardneg.py 와 같은 로더·같은 순서.
    X, y, groups = build_split(cfg, "val")
    print(f"[data] 창 {len(X)}개 {tuple(X.shape[1:])} · 양성 {int(y.sum())}개")

    # ── 점수 파일 읽고 창 수·순서 확인 ────────────────────────────────────
    score_rows = list(csv.DictReader(open(a.scores, encoding="utf-8")))
    if len(score_rows) != len(X):
        sys.exit(f"중단: 창 수가 다르다. build_split={len(X)} vs {a.scores}={len(score_rows)}. "
                 "로더 설정이나 라벨 파일이 eval_hardneg.py 때와 달라졌다.")

    score_map = {(r["csv_file"], int(r["window_idx"])): float(r["score"]) for r in score_rows}
    per_file_idx: dict = {}
    keys = []
    for gfile in groups:
        idx = per_file_idx.get(gfile, 0)
        per_file_idx[gfile] = idx + 1
        keys.append((gfile, idx))

    missing = [k for k in keys if k not in score_map]
    if missing:
        sys.exit(f"중단: 조인 실패 {len(missing)}건 (예: {missing[:3]}). "
                 "창 순서가 eval_hardneg.py 때와 어긋났다.")
    scores = np.array([score_map[k] for k in keys], dtype=np.float64)
    print(f"[join] {len(scores)}창 전부 조인 성공 (실패 0건)")

    # ── 에너지 계산 ──────────────────────────────────────────────────────
    E = np.array([motion_energy(X[i]) for i in range(len(X))], dtype=np.float64)
    e_mean, e_max, e_std, e_valid, zero_frac = (E[:, 0], E[:, 1], E[:, 2], E[:, 3], E[:, 4])
    print(f"[energy] e_mean {e_mean.min():.4f}~{e_mean.max():.4f} · "
          f"e_max {e_max.min():.4f}~{e_max.max():.4f} · "
          f"e_std {e_std.min():.4f}~{e_std.max():.4f} · "
          f"e_valid {np.nanmin(e_valid):.4f}~{np.nanmax(e_valid):.4f}")
    print(f"[missing] 결측 0패딩 비율 평균 {zero_frac.mean():.3f} "
          f"(범위 {zero_frac.min():.3f}~{zero_frac.max():.3f})")

    tags = np.array([tag_of(g) for g in groups])
    labels = y.astype(int)

    # ── 저장 ─────────────────────────────────────────────────────────────
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["csv_file", "tag", "window_idx", "label", "score",
                    "e_mean", "e_max", "e_std", "e_valid", "zero_frac"])
        for (f, i), t, lab, s, em, ex, es, ev, zf in zip(
                keys, tags, labels, scores, e_mean, e_max, e_std, e_valid, zero_frac):
            w.writerow([f, t, i, int(lab), f"{s:.6f}", f"{em:.6f}", f"{ex:.6f}",
                        f"{es:.6f}", f"{ev:.6f}", f"{zf:.6f}"])
    print(f"[out] 창 단위 결과 {len(scores)}행 -> {out_path}")

    seen = [t for t in TAG_ORDER if (tags == t).any()]
    seen += sorted(set(tags.tolist()) - set(TAG_ORDER))

    # ── 표 0: 측정 도구 타당성 검사 ──────────────────────────────────────
    print("\n[표 0] 측정 도구 타당성 — 에너지가 움직임이 아니라 결측을 재고 있는가")
    print(f"{'지표':<14}{'~ 결측률 상관':>16}   판정")
    print("-" * 52)
    verdicts = {}
    for name, arr in (("e_mean", e_mean), ("e_max", e_max),
                      ("e_std", e_std), ("e_valid", e_valid)):
        r = spearman(arr, zero_frac)
        bad = abs(r) >= CONFOUND_WARN
        verdicts[name] = r
        print(f"{name:<14}{r:>+16.3f}   {'오염 — 결측 지표에 가깝다' if bad else 'OK'}")
    print(f"{'모델 점수':<14}{spearman(scores, zero_frac):>+16.3f}   (참고)")

    # ── 표 1: 순위 상관 ──────────────────────────────────────────────────
    print("\n[표 1] 모델 점수 vs 움직임 에너지 — Spearman 순위 상관")
    print(f"{'구간':<18}{'창 수':>7}{'e_mean':>10}{'e_max':>10}{'e_std':>10}{'e_valid':>10}")
    print("-" * 67)
    print(f"{'전체':<18}{len(scores):>7}"
          f"{spearman(scores, e_mean):>10.3f}{spearman(scores, e_max):>10.3f}"
          f"{spearman(scores, e_std):>10.3f}{spearman(scores, e_valid):>10.3f}")
    for t in seen:
        m = tags == t
        print(f"{t:<18}{int(m.sum()):>7}"
              f"{spearman(scores[m], e_mean[m]):>10.3f}"
              f"{spearman(scores[m], e_max[m]):>10.3f}"
              f"{spearman(scores[m], e_std[m]):>10.3f}"
              f"{spearman(scores[m], e_valid[m]):>10.3f}")

    # ── 표 2: 에너지 단독 분류 성능 ──────────────────────────────────────
    base = float(labels.mean())
    model_pr = pr_auc(labels, scores)
    print("\n[표 2] 에너지 단독 분류 PR-AUC — train.py 와 같은 average_precision_score")
    print(f"{'점수원':<24}{'PR-AUC':>10}{'모델 대비':>12}{'랜덤 대비':>12}")
    print("-" * 58)
    for name, v in (("모델 (sigmoid 점수)", model_pr),
                    ("e_mean 단독", pr_auc(labels, e_mean)),
                    ("e_max 단독", pr_auc(labels, e_max)),
                    ("e_std 단독", pr_auc(labels, e_std)),
                    ("e_valid 단독(보정)", pr_auc(labels, e_valid))):
        print(f"{name:<24}{v:>10.4f}{v / model_pr:>11.2f}x{v / base:>11.2f}x")
    print(f"{'랜덤 기준선(양성비)':<24}{base:>10.4f}")
    print(f"참고: 학습 당시 모델 val PR-AUC = {H2_MODEL_PR_AUC}. 이번 평가셋은 하드네거가"
          f" 더해져 양성비({base:.3f})가 달라 같은 자가 아니다.")

    # ── 표 3: 그룹별 에너지 분포 ─────────────────────────────────────────
    print("\n[표 3] 그룹별 움직임 에너지 분포 (중앙값 [25% ~ 75%])")
    print(f"{'그룹':<18}{'창수':>6}{'e_mean':>26}{'e_max':>26}{'결측률':>8}")
    print("-" * 85)
    for t in seen:
        m = tags == t
        s1 = f"{np.median(e_mean[m]):.4f} [{q(e_mean[m],25):.4f}~{q(e_mean[m],75):.4f}]"
        s2 = f"{np.median(e_max[m]):.4f} [{q(e_max[m],25):.4f}~{q(e_max[m],75):.4f}]"
        print(f"{t:<18}{int(m.sum()):>6}{s1:>26}{s2:>26}{np.median(zero_frac[m]):>8.3f}")
    pos = labels == 1
    print(f"{'(낙상 라벨 창)':<18}{int(pos.sum()):>6}"
          f"{np.median(e_mean[pos]):>26.4f}{np.median(e_max[pos]):>26.4f}"
          f"{np.median(zero_frac[pos]):>8.3f}")

    # ── 판정 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("판정 — 기준은 측정 전에 정해둔 것이고, 처음 정한 세 정의로만 판정한다")
    print("=" * 72)

    rhos = {"e_mean": spearman(scores, e_mean), "e_max": spearman(scores, e_max),
            "e_std": spearman(scores, e_std)}
    best_name = max(rhos, key=lambda k: abs(rhos[k]))
    best_rho = rhos[best_name]
    h1 = "채택" if abs(best_rho) >= H1_ACCEPT else ("기각" if abs(best_rho) < H1_REJECT else "판단보류")
    print(f"H1 점수 ~ 움직임 강도 : |rho| 최대 {abs(best_rho):.3f} ({best_name}, 부호 "
          f"{'+' if best_rho > 0 else '-'}) -> **{h1}**  (>={H1_ACCEPT} 채택 / <{H1_REJECT} 기각)")

    prs = {"e_mean": pr_auc(labels, e_mean), "e_max": pr_auc(labels, e_max),
           "e_std": pr_auc(labels, e_std)}
    best_pr = max(prs.values())
    thr = H2_MODEL_PR_AUC * H2_ACCEPT_RATIO
    h2 = "채택" if best_pr >= thr else "기각"
    print(f"H2 에너지 단독 충분   : 최고 PR-AUC {best_pr:.4f} vs 기준 {thr:.4f} -> **{h2}**")

    lie = tags == "hardneg_liedown"
    if lie.any() and pos.any():
        lie_med, fall_med = float(np.median(e_max[lie])), float(np.median(e_max[pos]))
        ratio = lie_med / fall_med if fall_med else float("nan")
        h3 = "채택" if ratio < H3_ACCEPT_RATIO else "기각"
        print(f"H3 눕기는 실제로 느림 : e_max 중앙 눕기 {lie_med:.4f} / 낙상 {fall_med:.4f} "
              f"= {ratio:.2f} -> **{h3}**  (<{H3_ACCEPT_RATIO} 채택)")
    else:
        print("H3 : 눕기 또는 낙상 창이 없어 판정 불가")

    worst = max(abs(verdicts[k]) for k in ("e_mean", "e_max", "e_std"))
    if worst >= CONFOUND_WARN:
        print(f"\n⚠ 경고: 에너지 지표가 결측률과 |rho| {worst:.3f} 로 상관한다. "
              "위 판정은 '가설이 틀렸다'가 아니라\n"
              "  '이 에너지 지표로는 가설을 검증하지 못했다'로 읽어야 한다. "
              "표 0 과 문서를 같이 볼 것.")

    print("\n주의: 1인·1실·1일 데이터다. 이 진단은 이 데이터 안에서만 유효하다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
