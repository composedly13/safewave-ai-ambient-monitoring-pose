#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mark_falls.py — 낙상 순간을 rp5 수집기와 같은 시계로 찍는다.

수집기를 띄운 뒤 같은 기기(rp5)에서 이 스크립트를 함께 돌린다.
낙상이 일어나는 순간 스페이스바(또는 f)를 누르면 epoch ms가 기록된다.

    python3 mark_falls.py --session csi_20260801_1430.csv

끝나면 markers/csi_20260801_1430.markers.csv 가 생기고,
verify_sync.py 가 이걸 labels.csv 한 줄로 바꿔준다.

의존성 없음. 표준 라이브러리만 쓴다 (rp5에 아무것도 설치할 필요 없음).
"""

import argparse
import csv
import os
import sys
import time

# ── 시계 ─────────────────────────────────────────────────────────
# epoch ms 를 기록한다. 수집 CSV 의 ts_ms 는 장치 시계라 값이 다르지만,
# stream_id 앞부분에 같은 epoch 가 들어 있어서(예: "1781165460004-0")
# verify_sync.py 가 파일에서 오프셋을 계산해 자동 변환해줘.
# 실측 드리프트가 44분에 19ms 라 시계를 맞출 필요가 없다.
def now_ms() -> int:
    return time.time_ns() // 1_000_000


# ── 키 입력 ──────────────────────────────────────────────────────
class KeyReader:
    """엔터 없이 한 글자를 읽는다. tty가 아니면 줄 단위로 폴백."""

    def __enter__(self):
        self.raw = False
        try:
            import termios, tty  # noqa: F401

            self.termios = termios
            self.fd = sys.stdin.fileno()
            if os.isatty(self.fd):
                self.saved = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
                self.raw = True
        except Exception:
            pass
        return self

    def read(self) -> str:
        if self.raw:
            return sys.stdin.read(1)
        line = sys.stdin.readline()
        return line[:1] if line else "q"

    def __exit__(self, *exc):
        if self.raw:
            self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.saved)


BANNER = """
┌───────────────────────────────────────────────────────────┐
│  낙상 마커                                                │
│                                                           │
│   space / f   낙상 순간 기록                              │
│   x           직전 마커 취소                              │
│   b           나쁜 시도 표시 (분석에서 제외)              │
│   q           종료 후 저장                                │
└───────────────────────────────────────────────────────────┘
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True,
                    help="대응하는 수집 CSV 파일명 (예: csi_20260801_1430.csv)")
    ap.add_argument("--outdir", default="markers")
    ap.add_argument("--probe", action="store_true",
                    help="현재 epoch ms만 출력하고 종료")
    a = ap.parse_args()

    if a.probe:
        print(now_ms())
        return 0

    os.makedirs(a.outdir, exist_ok=True)
    stem = a.session[:-4] if a.session.endswith(".csv") else a.session
    path = os.path.join(a.outdir, f"{stem}.markers.csv")
    if os.path.exists(path):
        print(f"이미 있는 파일이야: {path}")
        print("덮어쓰지 않고 멈출게. 세션 이름을 바꾸거나 기존 파일을 옮겨.")
        return 1

    marks = []          # [(ts_ms, kind)]
    t0 = now_ms()
    print(BANNER)
    print(f"세션 {a.session}   시작 {t0}")
    print("수집기가 이미 돌고 있는지 확인하고 시작해.\n")

    with KeyReader() as kr:
        while True:
            try:
                ch = kr.read()
            except KeyboardInterrupt:
                ch = "q"
            if not ch:
                continue
            c = ch.lower()

            if c in (" ", "f"):
                ts = now_ms()
                marks.append((ts, "fall"))
                n = sum(1 for _, k in marks if k == "fall")
                print(f"  ● 낙상 #{n}   ts_ms={ts}   (+{(ts - t0) / 1000:.1f}s)")

            elif c == "b":
                ts = now_ms()
                marks.append((ts, "bad"))
                print(f"  ○ 나쁜 시도   ts_ms={ts}")

            elif c == "x":
                if marks:
                    ts, kind = marks.pop()
                    print(f"  ✕ 취소   {kind} ts_ms={ts}")
                else:
                    print("  ✕ 취소할 마커가 없어")

            elif c == "q":
                break

    t1 = now_ms()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts_ms", "kind"])
        w.writerow([t0, "session_start"])
        for ts, kind in marks:
            w.writerow([ts, kind])
        w.writerow([t1, "session_end"])

    n_fall = sum(1 for _, k in marks if k == "fall")
    n_bad = sum(1 for _, k in marks if k == "bad")
    print(f"\n저장: {path}")
    print(f"낙상 {n_fall}건 · 나쁜 시도 {n_bad}건 · 길이 {(t1 - t0) / 60000:.1f}분")
    print(f"\n다음: python3 verify_sync.py --session {a.session}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
