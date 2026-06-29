"""Export trained PyTorch model -> ONNX (opset 17) + emit the rp5 patch bundle.

Produces:
  1. {export.onnx_path}  (runs/m1_wifi_pose.onnx)        — the deliverable rp5 loads
  2. parity check: torch fall_logit vs onnxruntime fall_logit within tolerance
  3. {paths.out_dir}/rp5_patch/export_m1_wifi_pose_onnx.py
        self-contained rp5-side export script (inline CNN-GRU == M1FallNet, loads
        this repo's state_dict), dummy (1,1,64,100) -> fall_logit (B,1), opset 17
  4. {paths.out_dir}/rp5_patch/PREPROCESS_AND_EXPORT.md
        the _preprocess (192 -> 64) + export-script before/after notes

The model emits (B,1)=fall_logit (pre-sigmoid); rp5 _infer_onnx else-branch reads
output.reshape(-1)[0] then sigmoids it.
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
    model = build_model(cfg.model)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model.eval()
    return model


def export_onnx(cfg: Config, model, onnx_path: Path) -> None:
    torch = _torch()
    # HARD CONTRACT: (1, 1, 64, 100). Time axis is fixed at 100 (no dynamic time).
    dummy = torch.randn(1, 1, cfg.tensor.n_channels, cfg.tensor.window_frames, dtype=torch.float32)
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
          f"in={cfg.export.input_name}, out={cfg.export.output_name}=fall_logit (B,1))")


def parity_check(cfg: Config, model, onnx_path: Path, atol: float = 1e-4) -> bool:
    torch = _torch()
    import onnxruntime as ort
    x = np.random.randn(2, 1, cfg.tensor.n_channels, cfg.tensor.window_frames).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x)).numpy()      # (2, 1)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = np.asarray(sess.run(None, {cfg.export.input_name: x})[0])
    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    ok = max_diff <= atol
    print(f"[parity] max|torch-onnx|={max_diff:.2e} (atol={atol:.0e}) -> {'OK' if ok else 'FAIL'}")
    print(f"[parity] sample torch logit={torch_out.reshape(-1)[0]:.4f}  "
          f"onnx logit={onnx_out.reshape(-1)[0]:.4f}")
    return ok


def emit_rp5_patch(cfg: Config, ckpt: str, patch_dir: Path) -> None:
    """Write a self-contained rp5-side export script + the preprocess/export notes."""
    patch_dir.mkdir(parents=True, exist_ok=True)
    m = cfg.model
    script = f'''"""M1 ONNX export for rp5 — REPLACES the random DTPoseModel.

Self-contained: inline CNN-GRU matching the safewave-m1 M1FallNet, loads the trained
state_dict, exports (1, 1, {cfg.tensor.n_channels}, {cfg.tensor.window_frames}) -> fall_logit (B,1) (opset {cfg.export.opset}).
Point --ckpt at best.pt from the M1 training repo.
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn

N_SUBCARRIERS = {cfg.tensor.n_channels}    # HARD CONTRACT — 64 (NOT the legacy 192)
N_FRAMES = {cfg.tensor.window_frames}          # HARD CONTRACT — 100 frames (1 s @ 100 Hz), fixed
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
        chans = [1, *CONV_CHANNELS]
        self.convs = nn.Sequential(*[ConvBlock(chans[i], chans[i + 1]) for i in range(len(CONV_CHANNELS))])
        freq_out = N_SUBCARRIERS // (2 ** len(CONV_CHANNELS))
        feat_dim = CONV_CHANNELS[-1] * freq_out
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=GRU_HIDDEN, num_layers=GRU_LAYERS,
                          batch_first=True, dropout=DROPOUT if GRU_LAYERS > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(GRU_HIDDEN, 1))

    def forward(self, x):                       # x: (B, 1, 64, 100)
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
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model.eval()

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 1, N_SUBCARRIERS, N_FRAMES)   # (1,1,64,100) — NOT (1,1,192,100)
    torch.onnx.export(model, (dummy,), str(out), export_params=True, opset_version=a.opset,
                      do_constant_folding=True, input_names=["{cfg.export.input_name}"],
                      output_names=["{cfg.export.output_name}"],
                      dynamic_axes={{"{cfg.export.input_name}": {{0: "batch"}},
                                    "{cfg.export.output_name}": {{0: "batch"}}}})
    print(f"[M1] exported {{out}} (out=fall_logit (B,1))")


if __name__ == "__main__":
    main()
'''
    (patch_dir / "export_m1_wifi_pose_onnx.py").write_text(script, encoding="utf-8")

    notes = f'''# rp5 inference-side edits (ship together with the new ONNX)

Two edits switch rp5 from the random `DTPoseModel` to this trained CNN-GRU.
(Unlike M2, no main.py buffer change is needed — M1 infers on a single
1-node x 100-frame window already assembled upstream.)

## 1. ai/experts/m1_wifi_pose.py._preprocess  (192 -> 64 subcarriers)

The model input is (1, 1, {cfg.tensor.n_channels}, {cfg.tensor.window_frames}). block_raw is already per-frame
peak-normalized on the ESP, so DO NOT re-normalize and DO NOT reconstruct absolute
amplitude. Only the subcarrier count changes: the legacy PulseFi 192 -> 64.

Before:
```python
N_SUBCARRIERS = 192
...
def _preprocess(self, signal_data):
    data = np.asarray(signal_data, dtype=np.float32)
    # ... reshape / pad to (1, 1, 192, 100) ...
```

After:
```python
N_SUBCARRIERS = {cfg.tensor.n_channels}     # HARD CONTRACT: block_raw 64 subcarriers (was 192)
N_FRAMES = {cfg.tensor.window_frames}         # fixed 1 s @ 100 Hz
...
def _preprocess(self, signal_data):
    data = np.asarray(signal_data, dtype=np.float32)
    # ... reshape / pad to (1, 1, 64, 100) ...
```
Everything downstream is identical — `_infer_onnx` already feeds name
`{cfg.export.input_name}` and reads the else-branch output as `out.reshape(-1)[0]`
(then sigmoid), which is exactly our `{cfg.export.output_name}` (B,1).

## 2. scripts/export_m1_wifi_pose_onnx.py

Replace with `export_m1_wifi_pose_onnx.py` in this folder. Changes vs the old script:
  - `dummy_input` (1, 1, 192, 100) -> (1, 1, {cfg.tensor.n_channels}, {cfg.tensor.window_frames})
  - the random `DTPoseModel` -> inline CNN-GRU (M1FallNet) loading this repo's
    trained `state_dict` (`--ckpt best.pt`)
  - output name `{cfg.export.output_name}`, opset {cfg.export.opset}, batch-only dynamic axis
'''
    (patch_dir / "PREPROCESS_AND_EXPORT.md").write_text(notes, encoding="utf-8")
    print(f"[patch] wrote rp5 bundle -> {patch_dir}/")


def export(config_path: str, ckpt: str) -> None:
    cfg = load_config(config_path)
    model = _load_model(cfg, ckpt)

    onnx_path = Path(cfg.export.onnx_path)
    export_onnx(cfg, model, onnx_path)
    ok = parity_check(cfg, model, onnx_path)
    emit_rp5_patch(cfg, ckpt, Path(cfg.paths.out_dir) / "rp5_patch")
    if not ok:
        raise RuntimeError("ONNX parity check failed — torch and onnxruntime disagree")
    print("[export] done - m1_wifi_pose.onnx ready for rp5")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m1_fall.yaml")
    p.add_argument("--ckpt", required=True, help="path to trained checkpoint (.pt)")
    a = p.parse_args()
    export(a.config, a.ckpt)
