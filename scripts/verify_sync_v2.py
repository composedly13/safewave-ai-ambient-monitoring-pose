#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_sync.py - 수집 세트를 검증하고 labels.csv 줄들을 만든다.

실제 수집 파일을 보고 다시 쓴 버전이야. 두 가지가 이전 가정과 달랐어.

  1. ts_ms 는 epoch 가 아니다
     ts_ms 는 장치 시계고, epoch 는 stream_id 앞부분에 들어 있어
     ("1781165460004-0" 형태). dataset.py 는 fall_ts_ms 를 ts_ms 컬럼과
     비교하니까, 마커(epoch)를 ts_ms 도메인으로 변환해야 해.

     오프셋 = median(stream_id_epoch - ts_ms) 를 파일에서 직접 계산한다.
     실측에서 44분에 19ms(약 7ppm)만 드리프트해서 충분히 안정적이야.
     시계를 맞출 필요가 없다는 뜻이고, 이게 훨씬 튼튼해.

  2. 수집기가 60초마다 파일을 쪼갠다
     4분을 찍으면 파일이 4개 나와. 마커를 epoch 범위로 각 파일에 배분한다.

사용
    python3 verify_sync.py --markers markers/run01.markers.csv --rawdir data/raw
    python3 verify_sync.py --rawdir data/raw --normal-only

의존성 없음. 표준 라이브러리만.
"""

import argparse
import csv
import glob
import os
import statistics
import sys

FS_HZ = 100.0
FS_TOL = 0.15
GAP_FACTOR = 5
HALFWIDTH_MS = 1000
WRAP = 2 ** 32          # ts_ms 는 Unix ms 의 하위 32비트 (펌웨어 packet.h)
                        # 49.7일마다 4294967295 -> 0 으로 돌아간다

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


def epoch_of(stream_id):
    """'1781165460004-0' -> 1781165460004"""
    s = str(stream_id).split("-")[0].strip()
    return int(s) if s.isdigit() else None


def scan(path, rep):
    per_node, offs, eps = {}, [], []
    with open(path, encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        try:
            header = next(rdr)
        except StopIteration:
            rep.bad(f"{os.path.basename(path)}: 빈 파일")
            return None
        cols = {c.strip(): i for i, c in enumerate(header)}

        need = ["stream_id", "node_id", "ts_ms"]
        miss = [c for c in need if c not in cols]
        if miss:
            rep.bad(f"{os.path.basename(path)}: 필수 컬럼 없음 {miss}")
            return None
        raw_missing = [f"raw_{i}" for i in range(64) if f"raw_{i}" not in cols]
        if raw_missing:
            rep.bad(f"{os.path.basename(path)}: raw_ 컬럼 {len(raw_missing)}개 없음")

        i_s, i_n, i_t = cols["stream_id"], cols["node_id"], cols["ts_ms"]
        for row in rdr:
            if len(row) <= max(i_s, i_n, i_t):
                continue
            try:
                ts = int(float(row[i_t]))
            except ValueError:
                continue
            per_node.setdefault(row[i_n].strip(), []).append(ts)
            ep = epoch_of(row[i_s])
            if ep is not None:
                eps.append(ep)
                offs.append(ep - ts)

    if not per_node:
        rep.bad(f"{os.path.basename(path)}: 타임스탬프를 못 읽음")
        return None
    if offs:
        k = round(statistics.median(offs) / WRAP)
        skew = k * WRAP - statistics.median(offs)
    else:
        skew = None
    return {
        "name": os.path.basename(path),
        "nodes": per_node,
        "epochs": eps,
        "offset": statistics.median(offs) if offs else None,
        "off_spread": (max(offs) - min(offs)) if offs else None,
        "skew": skew,
    }


def check_file(info, rep):
    nodes = info["nodes"]
    all_ts = [t for v in nodes.values() for t in v]
    dur = (max(all_ts) - min(all_ts)) / 1000.0
    line = f"{info['name']}  노드 {len(nodes)}  {dur:.1f}초  {len(all_ts):,}행"

    expect = 1000.0 / FS_HZ
    fails = []
    for n in sorted(nodes):
        ts = sorted(nodes[n])
        if len(ts) < 10:
            fails.append(f"node {n} 샘플 {len(ts)}개")
            continue
        drops = [(a, b) for a, b in zip(ts[:-1], ts[1:]) if b < a - WRAP // 2]
        if drops:
            fails.append(f"node {n} ts_ms 랩어라운드 {len(drops)}회 "
                         f"(32비트 한계 초과 — 이 세션은 버려)")
        d = [b - a for a, b in zip(ts, ts[1:]) if b > a]
        if not d:
            fails.append(f"node {n} ts 증가 안 함")
            continue
        med = statistics.median(d)
        fs = 1000.0 / med if med else 0
        if abs(fs - FS_HZ) / FS_HZ > FS_TOL:
            fails.append(f"node {n} {fs:.1f}Hz")
        gaps = [x for x in d if x > expect * GAP_FACTOR]
        lost = sum(gaps) / 1000.0
        if gaps and lost > dur * 0.05:
            fails.append(f"node {n} 끊김 {lost:.1f}초")

    if info["offset"] is None:
        rep.bad(line + "  - stream_id 에서 epoch 를 못 읽음")
    elif info["off_spread"] is not None and info["off_spread"] > 1000:
        rep.warn_(line + f"  - 오프셋 흔들림 {info['off_spread']}ms")
    if fails:
        rep.bad(line + "  - " + " / ".join(fails))
    else:
        rep.ok(line)
    return dur


def read_markers(path):
    falls, bads = [], []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ts, kind = int(r["ts_ms"]), r["kind"]
            if kind == "fall":
                falls.append(ts)
            elif kind == "bad":
                bads.append(ts)
    return sorted(falls), sorted(bads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rawdir", default="data/raw")
    ap.add_argument("--pattern", default="csi_*.csv")
    ap.add_argument("--markers", help="mark_falls.py 출력 (epoch ms)")
    ap.add_argument("--normal-only", action="store_true",
                    help="마커 없이 전부 NORMAL 로 처리")
    ap.add_argument("--lag-ms", type=int, default=300,
                    help="키를 누르기까지의 반응 지연. 마커에서 뺀다.")
    ap.add_argument("--halfwidth", type=int, default=HALFWIDTH_MS)
    ap.add_argument("--out", help="labels 조각을 쓸 경로")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.rawdir, a.pattern)))
    if not files:
        sys.exit(f"수집 파일이 없어: {a.rawdir}/{a.pattern}")
    if not a.markers and not a.normal_only:
        sys.exit("--markers 를 주거나 --normal-only 를 붙여")

    rep = Report()
    print(f"\n{len(files)}개 파일  {a.rawdir}\n" + "-" * 62)

    infos = []
    for p in files:
        info = scan(p, rep)
        if info:
            check_file(info, rep)
            infos.append(info)
    if not infos:
        print("\n읽을 수 있는 파일이 없어.")
        return 1

    offs = [i["offset"] for i in infos if i["offset"] is not None]
    if offs:
        print("-" * 62)
        drift = max(offs) - min(offs)
        span = (max(max(i["epochs"]) for i in infos)
                - min(min(i["epochs"]) for i in infos)) / 1000.0
        print(f"  장치시계 오프셋 중앙 {int(statistics.median(offs))}"
              f"  (= {round(statistics.median(offs)/WRAP)} x 2^32 - skew)")
        sk = [i["skew"] for i in infos if i.get("skew") is not None]
        if sk:
            m = statistics.median(sk)
            print(f"  ESP-RPi 시계 스큐 {m:.0f}ms")
            if abs(m) > HALFWIDTH_MS:
                print(f"    · halfwidth {HALFWIDTH_MS}ms 보다 커. 파일에서 뽑은")
                print(f"      오프셋으로 변환하니 문제없지만, ts_ms 를 벽시계로")
                print(f"      착각해서 쓰면 양성 구간이 통째로 빗나가.")
        msg = f"  파일 간 드리프트 {drift}ms / {span:.0f}초"
        if span > 0:
            msg += f"  ({drift/span:.1f}ms per sec)"
        print(msg)
        if drift > 2000:
            rep.warn_("오프셋 드리프트가 커. 파일별로 따로 변환하니 문제는 없지만 확인해둬.")

    falls, bads = ([], [])
    if a.markers:
        if not os.path.exists(a.markers):
            sys.exit(f"마커 파일이 없어: {a.markers}")
        falls, bads = read_markers(a.markers)
        falls = [t - a.lag_ms for t in falls]
        print("-" * 62)
        line = f"  마커 {len(falls)}건 (반응 지연 -{a.lag_ms}ms 적용)"
        if bads:
            line += f" · 나쁜 시도 {len(bads)}건"
        print(line)

    rows, assigned = [], set()
    for info in infos:
        lo, hi = min(info["epochs"]), max(info["epochs"])
        off = info["offset"]
        mine = [t for t in falls if lo <= t <= hi]
        for t in mine:
            assigned.add(t)
        ts_field = ""
        label = "NORMAL"
        if mine and off is not None:
            label = "FALL"
            ts_field = ";".join(str(int(t - off)) for t in mine)
        note = f"self;{len(mine)}fall" if mine else "self"
        rows.append([info["name"], label, ts_field, "", "self", note])

    orphan = [t for t in falls if t not in assigned]
    if orphan:
        rep.bad(f"마커 {len(orphan)}건이 어떤 파일의 시간 범위에도 안 들어가. "
                f"수집기가 꺼진 사이에 눌렀거나 시계가 다른 기기야.")

    print("-" * 62)
    n_fall = sum(1 for r in rows if r[1] == "FALL")
    print(f"  FALL {n_fall}개 파일 · NORMAL {len(rows)-n_fall}개 파일")

    if rep.fatal:
        print(f"\n실패 {rep.fatal}건. labels 줄을 만들지 않을게.")
        return 1

    out_lines = [",".join(r) for r in rows]
    tail = f" (경고 {rep.warn}건)" if rep.warn else ""
    print(f"\n통과{tail}. labels.csv 에 붙일 줄:\n")
    for l in out_lines[:12]:
        print("  " + l)
    if len(out_lines) > 12:
        print(f"  ... 외 {len(out_lines)-12}줄")

    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["csv_file", "session_label", "fall_ts_ms",
                        "split", "source", "notes"])
            w.writerows(rows)
        print(f"\n{a.out} 에 저장했어.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
