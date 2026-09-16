"""M1 ONNX export for rp5 — REPLACES the random DTPoseModel.

Self-contained: inline CNN-GRU matching the safewave-m1 M1FallNet, loads the trained
state_dict, exports (1, 5, 64, 100) -> fall_logit (B,1) (opset 17).
Point --ckpt at best.pt from the M1 training repo.

INPUT IS 5-NODE EARLY FUSION: the node axis is the conv INPUT-CHANNEL axis.
Nodes are NOT concatenated onto the subcarrier axis (the forbidden 192 legacy).
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn

N_NODES = 5            # HARD CONTRACT — node axis = conv input channels
N_SUBCARRIERS = 64     # HARD CONTRACT — 64 per node (NOT the legacy 192)
N_FRAMES = 100         # HARD CONTRACT — 100 frames (1 s @ 100 Hz), fixed
CONV_CHANNELS = [16, 32, 64]
GRU_HIDDEN = 64
GRU_LAYERS = 1
DROPOUT = 0.3


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

    def forward(self, x):                       # x: (B, 5, 64, 100)
        x = self.convs(x)                       # (B, C, F', T)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])         # (B, 1) fall_logit, pre-sigmoid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best.pt from the M1 training repo")
    ap.add_argument("--out", default="volumes/models/m1_wifi_pose_onnx/m1_wifi_pose.onnx")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    model = M1FallNet()
    state = torch.load(a.ckpt, map_location="cpu")
    # strict=True: a ckpt trained at a different node count must fail loudly here.
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=True)
    model.eval()

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, N_NODES, N_SUBCARRIERS, N_FRAMES)   # (1,5,64,100)
    kw = dict(export_params=True, opset_version=a.opset, do_constant_folding=True,
              input_names=["csi_data"], output_names=["fall_logit"],
              dynamic_axes={"csi_data": {0: "batch"},
                            "fall_logit": {0: "batch"}})
    # torch>=2.9 defaults to the dynamo exporter, which needs the extra `onnxscript`
    # package; the legacy TorchScript path is what this contract was validated on.
    # torch<2.6 has no `dynamo` kwarg at all -> TypeError, so fall back.
    try:
        torch.onnx.export(model, (dummy,), str(out), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, (dummy,), str(out), **kw)
    print(f"[M1] exported {out} (in=(1,5,64,100), out=fall_logit (B,1))")


if __name__ == "__main__":
    main()
