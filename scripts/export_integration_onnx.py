#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_integration_onnx.py — 통합 레포(jinsan02/safewave-ai-ambient-monitoring)
배포용 ONNX 를 만든다.

이 레포의 기본 산출물(`models/m1_base_3node_20260916.onnx`)과 **일부러 다르게**
만든 파일이다. 둘 다 유지하며, 기본 산출물은 건드리지 않는다.

두 가지를 바꾼다:

  1) 첫 conv 의 입력 채널을 3 -> 5 로 **0 패딩**
     통합 레포는 5노드 슬롯으로 입력을 만든다. 가중치 (out, 3, kh, kw) 를
     (out, 5, kh, kw) 로 늘리고 채널 3·4 를 전부 0 으로 둔다. conv 는 입력
     채널에 대한 합이므로, 0 가중치 채널은 무엇이 들어와도 출력에 기여하지
     않는다 -> 3채널 모델과 **수학적으로 동일**하다. bias 는 그대로.

  2) 그래프 끝에 **Sigmoid 를 붙인다** (출력 이름 `fall_score`, 범위 0~1)

     ⚠️ 이 레포의 기본 규칙은 "ONNX 에 Sigmoid 를 넣지 마라" 다
        (`CLAUDE.md` 규칙 2 — rp5 가 sigmoid 를 하므로 이중 적용을 막으려는 것).
        **이 파일은 그 규칙의 의도적 예외다.** 통합 레포의
        `ai/experts/m1_wifi_pose.py` 가

            score = float(output_arr.reshape(-1)[0])
            score = np.clip(score, 0.0, 1.0)

        로 받는다. sigmoid 를 적용하지 않고 로짓을 그대로 자르므로, 그래프에
        sigmoid 가 없으면 음수 로짓이 전부 0.0 으로, 양수는 1.0 근처로 뭉개져
        임계값이 무의미해진다. **그래서 여기서는 넣어야 한다.**
        나중에 "규칙 2 위반"으로 보고 빼지 마라 — 빼면 조용히 망가진다.

사용
    $env:PYTHONPATH="src"
    .venv\\Scripts\\python.exe scripts\\export_integration_onnx.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CONV1_WEIGHT_KEY = "convs.0.net.0.weight"     # M1FallNet.convs[0].net[0] = Conv2d
INPUT_NAME = "csi_data"
OUTPUT_NAME = "fall_score"                    # 확률 (sigmoid 적용 후)
OPSET = 17
TARGET_NODES = 5                              # 통합 레포 슬롯 수


class SigmoidWrapper(nn.Module):
    """로짓을 내는 M1FallNet 에 sigmoid 를 붙여 0~1 확률로 만든다."""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.core(x))     # (B, 1) 0~1


def pad_conv1_channels(state_dict: dict, n_from: int, n_to: int) -> dict:
    """첫 conv 의 입력 채널을 n_from -> n_to 로 0 패딩한 state_dict 를 돌려준다."""
    sd = {k: v.clone() for k, v in state_dict.items()}
    w = sd[CONV1_WEIGHT_KEY]                   # (out, n_from, kh, kw)
    if w.shape[1] != n_from:
        raise ValueError(f"{CONV1_WEIGHT_KEY} 의 입력 채널이 {w.shape[1]} 이다 "
                         f"(기대 {n_from}). 체크포인트가 예상과 다르다.")
    out_c, _, kh, kw = w.shape
    padded = torch.zeros((out_c, n_to, kh, kw), dtype=w.dtype)
    padded[:, :n_from] = w                     # 나머지 채널은 0 으로 남는다
    sd[CONV1_WEIGHT_KEY] = padded
    print(f"[pad] {CONV1_WEIGHT_KEY}: {tuple(w.shape)} -> {tuple(padded.shape)} "
          f"(채널 {n_from}..{n_to - 1} 은 전부 0)")
    print(f"[pad] 패딩된 채널의 가중치 절댓값 합 = {float(padded[:, n_from:].abs().sum()):.1f} "
          f"(0 이어야 한다)")
    return sd


def verify_equivalence(m3: nn.Module, m5: nn.Module, n_trials: int = 5) -> float:
    """채널 3·4 에 잡음을 넣어도 3채널 모델과 출력이 같은지 확인한다.

    conv 가 입력 채널 합이고 해당 가중치가 0 이므로 **비트 단위로 같아야** 한다.
    0 이 아니면 패딩이 잘못된 것이므로 호출 쪽에서 중단한다.
    """
    rng = np.random.default_rng(0)
    worst = 0.0
    m3.eval(); m5.eval()
    with torch.no_grad():
        for i in range(n_trials):
            x3 = torch.from_numpy(rng.standard_normal((2, 3, 64, 100)).astype(np.float32))
            noise = torch.from_numpy(
                (rng.standard_normal((2, 2, 64, 100)) * 100.0).astype(np.float32))
            x5 = torch.cat([x3, noise], dim=1)          # 채널 3·4 = 큰 잡음
            d = float((m3(x3) - m5(x5)).abs().max())
            worst = max(worst, d)
            print(f"[verify] 시행 {i + 1}/{n_trials}: 잡음 크기 "
                  f"{float(noise.abs().max()):7.1f} · max|diff| = {d:.3e}")
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m1_fall.yaml")
    ap.add_argument("--ckpt", default="runs/m1_base_3node_20260916.pt")
    ap.add_argument("--out", default="models/integration/m1_wifi_pose.onnx")
    a = ap.parse_args()

    from m1_fall.config import load_config
    from m1_fall.model import build_model

    cfg = load_config(a.config)
    n_from = cfg.tensor.n_nodes                     # 3
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = blob["state_dict"] if "state_dict" in blob else blob
    print(f"[ckpt] {a.ckpt} · epoch {blob.get('epoch')} · 키 {len(sd)}개")

    # ── [1] 3채널 원본 + 5채널 0패딩 모델 ────────────────────────────────
    m3 = build_model(cfg.model, cfg.tensor.n_channels, cfg.tensor.window_frames, n_from)
    m3.load_state_dict(sd, strict=True)              # strict=True (규칙 6)
    m3.eval()
    print(f"[load] 3채널 모델 strict=True 로드 성공")

    sd5 = pad_conv1_channels(sd, n_from, TARGET_NODES)
    m5_core = build_model(cfg.model, cfg.tensor.n_channels, cfg.tensor.window_frames,
                          TARGET_NODES)
    m5_core.load_state_dict(sd5, strict=True)
    m5_core.eval()
    print(f"[load] {TARGET_NODES}채널 모델 strict=True 로드 성공")

    # ── [2] 등가성 검증 — 0 이 아니면 중단 ───────────────────────────────
    worst = verify_equivalence(m3, m5_core)
    print(f"[verify] 최댓값 max|diff| = {worst:.3e}")
    if worst != 0.0:
        sys.exit(f"중단: 등가성 검증 실패. max|diff| = {worst:.3e} (0 이어야 한다). "
                 "패딩된 채널이 출력에 영향을 주고 있다.")
    print("[verify] 비트 단위로 동일 — 채널 3·4 는 출력에 전혀 영향을 주지 않는다")

    # ── [3] Sigmoid 부착 ────────────────────────────────────────────────
    model = SigmoidWrapper(m5_core).eval()
    print(f"[wrap] Sigmoid 부착 -> 출력 '{OUTPUT_NAME}' 범위 0~1 "
          "(통합 레포가 sigmoid 를 하지 않으므로 의도적으로 넣는다)")

    # ── [4] 내보내기 ────────────────────────────────────────────────────
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, TARGET_NODES, cfg.tensor.n_channels, cfg.tensor.window_frames)
    torch.onnx.export(
        model, (dummy,), str(out),
        export_params=True, opset_version=OPSET, do_constant_folding=True,
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        dynamo=False,
    )
    print(f"[export] {out} (opset {OPSET}, in={INPUT_NAME}{tuple(dummy.shape)}, "
          f"out={OUTPUT_NAME})")

    # ── [5] 검증 ────────────────────────────────────────────────────────
    import onnx
    import onnxruntime as ort
    g = onnx.load(str(out)).graph
    ops = [n.op_type for n in g.node]
    print(f"[check] Sigmoid 노드 {ops.count('Sigmoid')}개 (1 이어야 한다)")
    if ops.count("Sigmoid") != 1:
        sys.exit(f"중단: Sigmoid 노드가 {ops.count('Sigmoid')}개다.")

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    worst_score = 0.0
    lo, hi = 1.0, 0.0
    for i in range(5):
        x3 = rng.standard_normal((4, 3, 64, 100)).astype(np.float32)
        x5 = np.concatenate([x3, rng.standard_normal((4, 2, 64, 100)).astype(np.float32) * 50],
                            axis=1)
        onnx_score = np.asarray(sess.run(None, {INPUT_NAME: x5})[0]).reshape(-1)
        with torch.no_grad():
            ref = torch.sigmoid(m3(torch.from_numpy(x3))).numpy().reshape(-1)
        d = float(np.max(np.abs(onnx_score - ref)))
        worst_score = max(worst_score, d)
        lo, hi = min(lo, float(onnx_score.min())), max(hi, float(onnx_score.max()))
        print(f"[check] 시행 {i + 1}/5: max|onnx - sigmoid(3채널 로짓)| = {d:.3e}")
    print(f"[check] 출력 범위 {lo:.6f} ~ {hi:.6f} -> [0,1] "
          f"{'충족' if 0.0 <= lo and hi <= 1.0 else '위반'}")
    print(f"[check] 최댓값 max|diff| = {worst_score:.3e} -> 기준 1e-5 "
          f"{'충족' if worst_score < 1e-5 else '미달'}")
    if not (0.0 <= lo and hi <= 1.0) or worst_score >= 1e-5:
        sys.exit("중단: 검증 실패.")
    print("[done] 통합 레포 배포용 ONNX 준비 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
