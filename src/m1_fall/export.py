"""Export trained PyTorch model -> ONNX (opset 17) + emit the rp5 patch bundle.

Produces:
  1. {export.onnx_path}  (runs/m1_wifi_pose.onnx)        — the deliverable rp5 loads
  2. parity check: torch fall_logit vs onnxruntime fall_logit within tolerance
  3. {paths.out_dir}/rp5_patch/export_m1_wifi_pose_onnx.py
        self-contained rp5-side export script (inline CNN-GRU == M1FallNet, loads
        this repo's state_dict), dummy (1,n_nodes,64,100) -> fall_logit (B,1), opset 17
  4. {paths.out_dir}/rp5_patch/PREPROCESS_AND_EXPORT.md
        the rp5-side _preprocess + export-script notes (5-node input)

Input contract: (B, n_nodes, 64, 100) — nodes on the conv input-channel axis
(early fusion), never concatenated onto the subcarrier axis.
The model emits (B,1)=fall_logit (pre-sigmoid); NO Sigmoid node is baked into the
graph — rp5 applies sigmoid on its side (CLAUDE.md 규칙 2).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import Config, load_config


def _torch():
    import torch
    return torch


def _load_model(cfg: Config, ckpt: str):
    torch = _torch()
    from .model import build_model
    # n_nodes MUST match the checkpoint: conv1's input-channel count is part of the
    # weight shape, so a single-node ckpt cannot load into a 5-node model (and the
    # strict=True below makes that failure loud rather than silent — 규칙 6).
    model = build_model(cfg.model, cfg.tensor.n_channels, cfg.tensor.window_frames,
                        cfg.tensor.n_nodes)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=True)
    model.eval()
    return model


def _input_shape(cfg: Config, batch: int) -> tuple:
    """(B, n_nodes, 64, 100) — the one input shape this repo exports."""
    return (batch, cfg.tensor.n_nodes, cfg.tensor.n_channels, cfg.tensor.window_frames)


def export_onnx(cfg: Config, model, onnx_path: Path) -> None:
    torch = _torch()
    # HARD CONTRACT: (1, n_nodes, 64, 100). Node/subcarrier/time axes are all fixed;
    # only batch is dynamic.
    dummy = torch.randn(*_input_shape(cfg, 1), dtype=torch.float32)
    dynamic = {cfg.export.input_name: {0: "batch"},
               cfg.export.output_name: {0: "batch"}}
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (dummy,), str(onnx_path),
        export_params=True, opset_version=cfg.export.opset, do_constant_folding=True,
        input_names=[cfg.export.input_name], output_names=[cfg.export.output_name],
        dynamic_axes=dynamic,
        dynamo=False,   # torch>=2.11: legacy TorchScript exporter (stable for opset 17).
    )
    print(f"[export] wrote {onnx_path} (opset {cfg.export.opset}, "
          f"in={cfg.export.input_name}{tuple(dummy.shape)}, "
          f"out={cfg.export.output_name}=fall_logit (B,1))")


def assert_no_sigmoid(onnx_path: Path) -> None:
    """규칙 2: the graph must END at fall_logit. A Sigmoid here is applied twice."""
    import onnx
    graph = onnx.load(str(onnx_path)).graph
    offenders = [n.name or n.op_type for n in graph.node if n.op_type == "Sigmoid"]
    if offenders:
        raise RuntimeError(
            f"ONNX graph contains {len(offenders)} Sigmoid node(s): {offenders[:3]}. "
            "rp5 applies sigmoid itself — baking one in double-applies it and collapses "
            "scores into 0.5~0.73, making the 0.7 threshold meaningless (CLAUDE.md 규칙 2)."
        )
    print("[check] Sigmoid 노드 없음 (규칙 2 준수)")


def parity_check(cfg: Config, model, onnx_path: Path, atol: float = 1e-4) -> bool:
    torch = _torch()
    import onnxruntime as ort
    x = np.random.randn(*_input_shape(cfg, 2)).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x)).numpy()      # (2, 1)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = np.asarray(sess.run(None, {cfg.export.input_name: x})[0])
    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    ok = max_diff <= atol
    print(f"[parity] input {x.shape} | max|torch-onnx|={max_diff:.2e} "
          f"(atol={atol:.0e}) -> {'OK' if ok else 'FAIL'}")
    print(f"[parity] sample torch logit={torch_out.reshape(-1)[0]:.4f}  "
          f"onnx logit={onnx_out.reshape(-1)[0]:.4f}")
    return ok


def emit_rp5_patch(cfg: Config, ckpt: str, patch_dir: Path) -> None:
    """Write a self-contained rp5-side export script + the preprocess/export notes."""
    patch_dir.mkdir(parents=True, exist_ok=True)
    m = cfg.model
    n_nodes = cfg.tensor.n_nodes
    script = f'''"""M1 ONNX export for rp5 — REPLACES the random DTPoseModel.

Self-contained: inline CNN-GRU matching the safewave-m1 M1FallNet, loads the trained
state_dict, exports (1, {n_nodes}, {cfg.tensor.n_channels}, {cfg.tensor.window_frames}) -> fall_logit (B,1) (opset {cfg.export.opset}).
Point --ckpt at best.pt from the M1 training repo.

INPUT IS 5-NODE EARLY FUSION: the node axis is the conv INPUT-CHANNEL axis.
Nodes are NOT concatenated onto the subcarrier axis (the forbidden 192 legacy).
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn

N_NODES = {n_nodes}            # HARD CONTRACT — node axis = conv input channels
N_SUBCARRIERS = {cfg.tensor.n_channels}     # HARD CONTRACT — 64 per node (NOT the legacy 192)
N_FRAMES = {cfg.tensor.window_frames}         # HARD CONTRACT — 100 frames (1 s @ 100 Hz), fixed
CONV_CHANNELS = {m.conv_channels}
GRU_HIDDEN = {m.gru_hidden}
GRU_LAYERS = {m.gru_layers}
DROPOUT = {m.dropout}


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),   # halves subcarrier axis, time untouched
        )

    def forward(self, x):
        return self.net(x)


class M1FallNet(nn.Module):
    def __init__(self):
        super().__init__()
        chans = [N_NODES, *CONV_CHANNELS]       # first conv fuses all nodes at once
        self.convs = nn.Sequential(*[ConvBlock(chans[i], chans[i + 1]) for i in range(len(CONV_CHANNELS))])
        freq_out = N_SUBCARRIERS // (2 ** len(CONV_CHANNELS))
        feat_dim = CONV_CHANNELS[-1] * freq_out
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=GRU_HIDDEN, num_layers=GRU_LAYERS,
                          batch_first=True, dropout=DROPOUT if GRU_LAYERS > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(GRU_HIDDEN, 1))

    def forward(self, x):                       # x: (B, {n_nodes}, {cfg.tensor.n_channels}, {cfg.tensor.window_frames})
        x = self.convs(x)                       # (B, C, F', T)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])         # (B, 1) fall_logit, pre-sigmoid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best.pt from the M1 training repo")
    ap.add_argument("--out", default="volumes/models/m1_wifi_pose_onnx/m1_wifi_pose.onnx")
    ap.add_argument("--opset", type=int, default={cfg.export.opset})
    a = ap.parse_args()

    model = M1FallNet()
    state = torch.load(a.ckpt, map_location="cpu")
    # strict=True: a ckpt trained at a different node count must fail loudly here.
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=True)
    model.eval()

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, N_NODES, N_SUBCARRIERS, N_FRAMES)   # (1,{n_nodes},{cfg.tensor.n_channels},{cfg.tensor.window_frames})
    kw = dict(export_params=True, opset_version=a.opset, do_constant_folding=True,
              input_names=["{cfg.export.input_name}"], output_names=["{cfg.export.output_name}"],
              dynamic_axes={{"{cfg.export.input_name}": {{0: "batch"}},
                            "{cfg.export.output_name}": {{0: "batch"}}}})
    # torch>=2.9 defaults to the dynamo exporter, which needs the extra `onnxscript`
    # package; the legacy TorchScript path is what this contract was validated on.
    # torch<2.6 has no `dynamo` kwarg at all -> TypeError, so fall back.
    try:
        torch.onnx.export(model, (dummy,), str(out), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, (dummy,), str(out), **kw)
    print(f"[M1] exported {{out}} (in=(1,{n_nodes},{cfg.tensor.n_channels},{cfg.tensor.window_frames}), out=fall_logit (B,1))")


if __name__ == "__main__":
    main()
'''
    (patch_dir / "export_m1_wifi_pose_onnx.py").write_text(script, encoding="utf-8")

    notes = f'''# rp5 inference-side edits (ship together with the new ONNX)

> ⚠️ **rp5 미검증 — 적용 전 반드시 `develop`을 직접 읽어라.**
> 이 문서는 이 학습 레포가 **내보내는 계약**을 근거로 자동 생성된 것이고,
> rp5의 **실제 코드를 읽고 쓴 것이 아니다.** 2026-08-25 정찰(T1)에서 rp5
> (`192.168.1.2`)에 접속하지 못했고, 그 뒤 5노드로 전환하면서 확인 범위가 더 넓어졌다.
> 상세: `docs/archive/RP5_DEVELOP_AUDIT.md`.
> 아래는 **확정된 작업 목록이 아니라 확인해야 할 질문 목록**이다.

## 이 레포가 내보내는 계약 (확정)

```
INPUT   {cfg.export.input_name}    (B, {n_nodes}, {cfg.tensor.n_channels}, {cfg.tensor.window_frames})   float32
        = 노드 {n_nodes} × 서브캐리어 {cfg.tensor.n_channels} × {cfg.tensor.window_frames}프레임 (1초 @ 100Hz)
        노드축은 conv 입력채널(조기융합). 서브캐리어축 192 concat 아님.
OUTPUT  {cfg.export.output_name}  (B, 1)   pre-sigmoid logit
        그래프에 Sigmoid 노드 없음 — rp5가 sigmoid를 한다.
opset   {cfg.export.opset} · 동적 축은 batch 뿐
```

## rp5에서 확인해야 할 것 (전부 미검증)

### Q1. `ai/experts/m1_wifi_pose.py._preprocess` 가 `(1,{n_nodes},{cfg.tensor.n_channels},{cfg.tensor.window_frames})`을 만드는가?

입력은 노드마다 {cfg.tensor.n_channels} 서브캐리어이고, **노드는 채널축으로 쌓는다.**
`block_raw`는 ESP에서 이미 per-frame peak 정규화돼 있으므로 **재정규화 금지,
절대진폭 복원 금지** (노드별로 독립 정규화된 상태 그대로 써야 한다).

기대 형태:
```python
N_NODES = {n_nodes}          # 노드축 = 채널축
N_SUBCARRIERS = {cfg.tensor.n_channels}   # 노드당 {cfg.tensor.n_channels} (레거시 192 아님)
N_FRAMES = {cfg.tensor.window_frames}       # 1초 @ 100Hz 고정
...
def _preprocess(self, signal_data):
    data = np.asarray(signal_data, dtype=np.float32)
    # -> (1, {n_nodes}, {cfg.tensor.n_channels}, {cfg.tensor.window_frames})
```

> 과거 스냅샷(2026-06-05)은 `192 * 100`을 하드코딩했고, `CLAUDE.md`의 함정 표는
> "develop은 ONNX에서 shape을 직접 읽으므로 수정 불필요"라고 적고 있다. **서로 모순이다.**
> 직접 읽어서 판정하라. shape을 ONNX에서 읽더라도, **노드를 어떤 순서·어떤 축으로
> 채우는지**는 별개 문제다.

### Q2. `ai/main.py`의 롤링 버퍼가 {n_nodes}노드를 **무엇으로 시간 정렬**하는가?  ← 가장 위험

이 레포는 **`stream_id`의 epoch**(rp5 수신 시각 = 유일한 공통 시계)로 100Hz 그리드를
깔고 노드를 ±5ms 최근접 슬롯에 맞춘다. `ts_ms`는 **보드마다 다른 장치 시계**(32비트 랩)라
노드 간 정렬에 쓰면 노드들이 서로 어긋난다.

rp5가 `ts_ms`로 정렬한다면 **shape은 맞는데 내용이 틀린 입력**이 들어간다 —
추론이 실패하지 않고 조용히 엉뚱한 점수를 낸다. 이게 가장 잡기 어려운 실패다.

### Q3. 노드가 빠진 시점에 rp5는 무엇을 채우는가?

이 레포 로더는 **0으로 패딩**한다. rp5가 직전 프레임을 복제하거나, 노드가 하나라도
없으면 추론을 건너뛴다면 학습 분포와 추론 분포가 갈린다.

### Q4. `_infer_onnx`가 sigmoid를 **몇 번** 하는가?

이 레포는 `{cfg.export.output_name}`을 **pre-sigmoid**로 내보낸다 (`CLAUDE.md` 규칙 3).
- rp5가 sigmoid를 **안 하면**: logit이 그대로 점수가 된다. 2026-06-05 스냅샷의
  `_infer_onnx`에는 sigmoid가 없고 `np.clip(score, 0.0, 1.0)`만 있었다 — 그 경우
  음수 logit은 전부 `0.0`, 양수는 `1.0` 근처로 잘려 **임계 0.7이 무의미**해진다.
- sigmoid가 **두 번** 걸리면: 출력이 0.5~0.73으로 뭉개진다 (`CLAUDE.md` 규칙 2).

어느 쪽인지 확인하기 전에는 **ONNX에 Sigmoid를 넣지도 빼지도 말고 현행(미포함)을 유지**하라.

## rp5 export 스크립트 교체

`scripts/export_m1_wifi_pose_onnx.py`를 이 폴더의 `export_m1_wifi_pose_onnx.py`로 교체.
기존 스크립트 대비 달라지는 점:
  - `dummy_input` (1, 1, 192, 100) -> (1, {n_nodes}, {cfg.tensor.n_channels}, {cfg.tensor.window_frames})
  - 랜덤 초기화 `DTPoseModel` -> 인라인 CNN-GRU (M1FallNet), 이 레포의 학습된
    `state_dict`를 `--ckpt best.pt`로 로드 (strict=True)
  - 출력 이름 `{cfg.export.output_name}`, opset {cfg.export.opset}, 동적 축은 batch 뿐

생성 근거 체크포인트: `{ckpt}`
'''
    (patch_dir / "PREPROCESS_AND_EXPORT.md").write_text(notes, encoding="utf-8")
    print(f"[patch] wrote rp5 bundle -> {patch_dir}/")


def export(config_path: str, ckpt: str) -> None:
    cfg = load_config(config_path)
    model = _load_model(cfg, ckpt)

    onnx_path = Path(cfg.export.onnx_path)
    export_onnx(cfg, model, onnx_path)
    assert_no_sigmoid(onnx_path)
    ok = parity_check(cfg, model, onnx_path)
    emit_rp5_patch(cfg, ckpt, Path(cfg.paths.out_dir) / "rp5_patch")
    if not ok:
        raise RuntimeError("ONNX parity check failed — torch and onnxruntime disagree")
    print("[export] done - m1_wifi_pose.onnx ready for rp5 (rp5 쪽 동작은 미검증)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m1_fall.yaml")
    p.add_argument("--ckpt", required=True, help="path to trained checkpoint (.pt)")
    a = p.parse_args()
    export(a.config, a.ckpt)
