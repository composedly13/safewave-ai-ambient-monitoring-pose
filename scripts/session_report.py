#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
session_report.py — 수집 세션 기록을 검증하고 노드별 커버리지를 계산한다.

M1은 5노드 전부에 같은 라벨(fall_ts_ms)을 붙인다. 세션 단위 라벨이라
낙상 지점에서 먼 노드·차폐된 노드에도 양성이 붙는데, 그 노드는 신호가
없으므로 라벨 잡음이 된다. M2에는 SNR 게이트가 있지만 M1에는 없다.

수집 시점에 노드 배치와 낙상 지점을 기록해두면 사후에 노드별 recall을
뽑아 기여 없는 노드를 걸러낼 수 있다. 기록을 안 하면 되돌릴 방법이 없다.

    python3 session_report.py                          # 검증 + 커버리지 요약
    python3 session_report.py --plan 200 50            # 수집 계획표 생성
    python3 session_report.py --join labels_self.csv   # 라벨에 노드거리 조인

의존성 없음. 표준 라이브러리만 (rp5에 아무것도 설치할 필요 없음).
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

DEFAULT_LAYOUT = "data/node_layout.csv"
DEFAULT_LOG = "data/session_log.csv"

# 차폐 등급 -> 신호 기여 기대치. 사후 필터 후보이지 확정 규칙이 아니다.
OCCLUSION_RANK = {"clear": 0, "furniture": 1, "door": 2, "wall": 3}
NEAR_M = 3.0          # 이 거리 안 + clear 이면 '기여 기대' 노드
EXPECTED_NODES = 5

OK, BAD, WARN = "  [o]", "  [X]", "  [!]"


class Report:
    def __init__(self):
        self.fatal = 0
        self.warn = 0

    def ok(self, m):
        print(f"{OK} {m}")

    def bad(self, m):
        print(f"{BAD} {m}")
        self.fatal += 1

    def warn_(self, m):
        print(f"{WARN} {m}")
        self.warn += 1


# ── 읽기 ──────────────────────────────────────────────────────────

def read_csv(path, rep):
    if not os.path.exists(path):
        rep.bad(f"{path} 없음")
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    live = [r for r in rows if not (r.get("notes", "") or "").startswith("예시행")]
    dropped = len(rows) - len(live)
    if dropped:
        rep.warn_(f"{os.path.basename(path)}: 예시행 {dropped}줄 무시 (삭제 권장)")
    return live


def parse_layouts(rows, rep):
    """layout_id -> {node_id: (x, y, z)}"""
    out = defaultdict(dict)
    for r in rows:
        try:
            lid = r["layout_id"].strip()
            nid = str(int(r["node_id"]))
            out[lid][nid] = (float(r["x_m"]), float(r["y_m"]), float(r.get("z_m") or 0.0))
        except (KeyError, ValueError) as exc:
            rep.bad(f"node_layout 행 파싱 실패 ({exc}): {r}")
    for lid, nodes in out.items():
        if len(nodes) != EXPECTED_NODES:
            rep.bad(f"배치 {lid}: 노드 {len(nodes)}개 (기대 {EXPECTED_NODES})")
        else:
            rep.ok(f"배치 {lid}: 노드 5개 좌표 기록됨")
    return out


def parse_occlusion(field, rep, sid):
    """'1:clear;2:wall;...' -> {node_id: grade}"""
    out = {}
    for part in (field or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            rep.bad(f"{sid}: occlusion_by_node 형식 오류 '{part}' (기대 '1:clear')")
            continue
        n, g = part.split(":", 1)
        g = g.strip().lower()
        if g not in OCCLUSION_RANK:
            rep.bad(f"{sid}: 알 수 없는 차폐 등급 '{g}' (허용 {list(OCCLUSION_RANK)})")
            continue
        out[n.strip()] = g
    return out


def dist(a, b):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


# ── 검증 + 커버리지 ────────────────────────────────────────────────

def analyze(layout_path, log_path):
    rep = Report()
    layouts = parse_layouts(read_csv(layout_path, rep), rep)
    sessions = read_csv(log_path, rep)
    if not sessions:
        rep.warn_("session_log.csv 에 세션이 없음 — 수집 전이면 정상")
        return rep, [], layouts

    print()
    enriched = []
    seen = set()
    for s in sessions:
        sid = (s.get("session_id") or "?").strip()
        if sid in seen:
            rep.bad(f"session_id 중복: {sid}")
        seen.add(sid)

        lid = (s.get("layout_id") or "").strip()
        if lid not in layouts:
            rep.bad(f"{sid}: 배치 {lid or '(빈칸)'} 가 node_layout.csv 에 없음")
            continue
        nodes = layouts[lid]

        try:
            spot = (float(s["spot_x_m"]), float(s["spot_y_m"]), 0.0)
        except (KeyError, ValueError):
            rep.bad(f"{sid}: spot_x_m / spot_y_m 미기입 — 노드 거리 계산 불가")
            continue

        occ = parse_occlusion(s.get("occlusion_by_node"), rep, sid)
        missing = sorted(set(nodes) - set(occ))
        if missing:
            rep.bad(f"{sid}: 노드 {missing} 차폐 미기입")

        per_node = {}
        for nid, xyz in nodes.items():
            d = dist(spot, xyz)
            g = occ.get(nid, "")
            per_node[nid] = (round(d, 2), g)
        contributing = [n for n, (d, g) in per_node.items() if d <= NEAR_M and g == "clear"]

        if not contributing:
            rep.bad(f"{sid}: 기여 기대 노드 0개 — 낙상 지점이 모든 노드에서 멀거나 가려짐")
        elif len(contributing) == EXPECTED_NODES:
            rep.warn_(f"{sid}: 5노드 전부 근접·clear — 먼 노드 표본이 안 모임")

        vp = (s.get("verify_pass") or "").strip().lower()
        if vp not in ("y", "yes", "pass", "o", "1"):
            rep.warn_(f"{sid}: verify_pass 미기입 — verify_sync_v2 통과 전")

        s["_per_node"] = per_node
        s["_contrib"] = contributing
        enriched.append(s)

    coverage(enriched)
    return rep, enriched, layouts


def coverage(sessions):
    if not sessions:
        return
    print("\n=== 클래스별 이벤트 수 ===")
    by_class = defaultdict(int)
    for s in sessions:
        try:
            by_class[(s.get("class") or "?").strip().upper()] += int(s.get("n_events") or 0)
        except ValueError:
            pass
    for k in sorted(by_class):
        print(f"  {k:10s} {by_class[k]:5d}")
    fall = by_class.get("FALL", 0)
    hard = by_class.get("HARDNEG", 0)
    if fall and hard / max(fall, 1) < 0.5:
        print(f"  [!] 하드네거티브가 낙상의 {hard/fall*100:.0f}% — 04 실측상 비슷한 비중까지 올려야 함")

    print("\n=== 노드별 거리 분포 (양성 라벨이 붙는 노드) ===")
    buckets = defaultdict(lambda: defaultdict(int))
    for s in sessions:
        if (s.get("class") or "").strip().upper() != "FALL":
            continue
        n_ev = int(s.get("n_events") or 0)
        for nid, (d, g) in s["_per_node"].items():
            b = "≤2m" if d <= 2 else ("2-4m" if d <= 4 else ">4m")
            if g != "clear":
                b += f"/{g}"
            buckets[nid][b] += n_ev
    for nid in sorted(buckets):
        parts = ", ".join(f"{k} {v}" for k, v in sorted(buckets[nid].items()))
        print(f"  node {nid}: {parts}")
    print("\n  거리·차폐대별 표본이 한쪽으로 쏠리면 노드별 recall 비교가 불가능하다.")

    print("\n=== 피험자 / 배치 다양성 ===")
    subj = {(s.get("subject_id") or "?").strip() for s in sessions}
    lays = {(s.get("layout_id") or "?").strip() for s in sessions}
    print(f"  피험자 {len(subj)}명 {sorted(subj)}")
    print(f"  배치   {len(lays)}종 {sorted(lays)}")
    if len(subj) < 5:
        print("  [!] 03 CV에서 피험자 간 변동이 구성 효과의 4배 — 5~8명 목표")
    if len(lays) < 2:
        print("  [!] 사전학습 데이터는 환경이 E22 하나뿐 — 배치 2~3종 필요")


# ── 계획표 생성 ────────────────────────────────────────────────────

FALL_DIRS = [("front", "전방"), ("back", "후방"), ("left", "좌측"), ("right", "우측")]
START_POSES = [("stand", "서기"), ("sit", "앉기"), ("walk", "걷다가")]
HARDNEGS = [
    "빠르게앉기-의자털썩", "빠르게앉기-침대쓰러지듯", "빠르게앉기-바닥주저앉기",
    "눕기", "쭈그려물건줍기", "물건떨어뜨리기", "비틀거리기",
]


def make_plan(n_fall, n_hardneg, per_session, out_path):
    rows = []
    i = 0
    combos = [(d, p) for d in FALL_DIRS for p in START_POSES]
    need = math.ceil(n_fall / per_session)
    for k in range(need):
        d, p = combos[k % len(combos)]
        i += 1
        rows.append(dict(session_id=f"F{i:03d}", csv_prefix="", date="", layout_id="L1",
                         subject_id="", **{"class": "FALL"},
                         scenario=f"낙상-{d[1]}-{p[1]}", fall_dir=d[0], start_pose=p[0],
                         spot_x_m="", spot_y_m="", n_events=per_session,
                         occlusion_by_node="", operator="", lag_ms=300, verify_pass="", notes=""))
    need_h = math.ceil(n_hardneg / per_session)
    for k in range(need_h):
        sc = HARDNEGS[k % len(HARDNEGS)]
        i += 1
        rows.append(dict(session_id=f"H{i:03d}", csv_prefix="", date="", layout_id="L1",
                         subject_id="", **{"class": "HARDNEG"},
                         scenario=sc, fall_dir="", start_pose="",
                         spot_x_m="", spot_y_m="", n_events=per_session,
                         occlusion_by_node="", operator="", lag_ms=300, verify_pass="", notes=""))
    cols = ["session_id", "csv_prefix", "date", "layout_id", "subject_id", "class", "scenario",
            "fall_dir", "start_pose", "spot_x_m", "spot_y_m", "n_events",
            "occlusion_by_node", "operator", "lag_ms", "verify_pass", "notes"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    mins = len(rows) * 2
    print(f"[plan] {out_path}: 녹화 {len(rows)}건 "
          f"(FALL {need} / HARDNEG {need_h}) · 이벤트 {n_fall}+{n_hardneg}")
    print(f"[plan] 녹화당 2분 가정 시 순수 수집 {mins}분 ≈ {mins/60:.1f}시간 "
          f"(설치·휴식 제외) · 5노드 데이터 약 {mins/60*5.8:.0f}GB")
    print("[plan] 피험자별로 이 표를 복제해라. layout_id 는 배치를 바꿀 때만 새로 딴다.")


# ── 라벨 조인 ──────────────────────────────────────────────────────

def join_labels(labels_path, sessions):
    """labels_self.csv 각 행에 세션의 노드별 거리/차폐를 붙여 출력한다."""
    if not os.path.exists(labels_path):
        print(f"[join] {labels_path} 없음")
        return
    by_prefix = {(s.get("csv_prefix") or "").strip(): s for s in sessions if s.get("csv_prefix")}
    if not by_prefix:
        print("[join] session_log 에 csv_prefix 가 비어 있어 조인 불가")
        return
    out = labels_path.replace(".csv", "_nodes.csv")
    with open(labels_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    hit = 0
    for r in rows:
        stem = (r.get("csv_file") or "").rsplit(".", 1)[0]
        s = next((v for k, v in by_prefix.items() if stem.startswith(k)), None)
        if not s:
            r["node_dist"] = ""
            r["node_occl"] = ""
            continue
        hit += 1
        r["node_dist"] = ";".join(f"{n}:{d}" for n, (d, _) in sorted(s["_per_node"].items()))
        r["node_occl"] = ";".join(f"{n}:{g}" for n, (_, g) in sorted(s["_per_node"].items()))
    cols = list(rows[0].keys()) if rows else []
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[join] {out}: {hit}/{len(rows)} 행에 노드 거리·차폐 부착")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default=DEFAULT_LAYOUT)
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--plan", nargs=2, type=int, metavar=("N_FALL", "N_HARDNEG"),
                    help="수집 계획표 생성 후 종료")
    ap.add_argument("--per-session", type=int, default=4, help="녹화 1건당 이벤트 수 (기본 4)")
    ap.add_argument("--plan-out", default="data/session_plan.csv")
    ap.add_argument("--join", metavar="LABELS_CSV", help="라벨에 노드 거리/차폐 조인")
    a = ap.parse_args()

    if a.plan:
        make_plan(a.plan[0], a.plan[1], a.per_session, a.plan_out)
        return 0

    print("=== 수집 기록 검증 ===")
    rep, sessions, _ = analyze(a.layout, a.log)
    if a.join:
        print()
        join_labels(a.join, sessions)

    print(f"\n치명 {rep.fatal} · 경고 {rep.warn}")
    if rep.fatal:
        print("치명 항목을 고치기 전에는 본 수집 데이터를 신뢰하지 마라.")
        return 1
    print("통과.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
