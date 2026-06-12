"""Export trained PyTorch model -> ONNX (opset 17).  [STUB — feature/m1-fall PR]

Planned:
  - load best checkpoint state_dict into model.build_model(cfg.model)
  - torch.onnx.export with dummy (1,1,64,100), input name cfg.export.input_name
    ('csi_data'), output name cfg.export.output_name ('fall_logit'), opset 17
  - parity check: torch logits vs onnxruntime logits within tolerance
  - emit the rp5 patch (192 -> 64 in _preprocess + export script dummy_input,
    and load THIS state_dict instead of the random DTPoseModel)
"""
from __future__ import annotations

import argparse

from .config import load_config


def export(config_path: str, ckpt: str) -> None:
    cfg = load_config(config_path)
    raise NotImplementedError(
        "export.py is a scaffold stub — implemented in the feature/m1-fall PR. "
        f"(target: {cfg.export.onnx_path}, opset={cfg.export.opset}, "
        f"in={cfg.export.input_name}, out={cfg.export.output_name})"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/m1_fall.yaml")
    p.add_argument("--ckpt", required=True, help="path to trained checkpoint (.pt)")
    a = p.parse_args()
    export(a.config, a.ckpt)
