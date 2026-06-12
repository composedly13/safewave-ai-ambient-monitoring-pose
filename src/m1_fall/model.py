"""M1 fall-detection model: CNN-GRU (db_spec M1 = CNN-GRU family).

Input  : (B, 1, 64, 100)  — 64 subcarriers × 100 frames, per-frame peak-normalized.
Flow   : Conv2d stack pools the SUBCARRIER axis (spatial channel/freq pattern) while
         PRESERVING the time axis → GRU over time captures the abrupt broadband change
         of a fall → FC → single pre-sigmoid logit.
Output : (B, 1)  fall_logit  (sigmoid applied downstream / in rp5).

Kept small for RPi5 CPU inference (target < a few ms).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelCfg


class ConvBlock(nn.Module):
    """Conv over (subcarrier, time); MaxPool halves subcarrier axis, time untouched."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # (freq//2, time)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class M1FallNet(nn.Module):
    def __init__(self, cfg: ModelCfg, n_subcarriers: int = 64, n_frames: int = 100):
        super().__init__()
        chans = [1, *cfg.conv_channels]
        self.convs = nn.Sequential(*[ConvBlock(chans[i], chans[i + 1]) for i in range(len(cfg.conv_channels))])

        # subcarrier axis halves once per conv block; time axis is preserved.
        freq_out = n_subcarriers // (2 ** len(cfg.conv_channels))
        feat_dim = cfg.conv_channels[-1] * freq_out

        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.gru_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 64, 100)
        x = self.convs(x)                 # (B, C, F', T)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (B, T, C*F') — sequence over time
        out, _ = self.gru(x)              # (B, T, H)
        logit = self.head(out[:, -1, :])  # (B, 1) — last timestep
        return logit


def build_model(cfg: ModelCfg, n_subcarriers: int = 64, n_frames: int = 100) -> M1FallNet:
    return M1FallNet(cfg, n_subcarriers, n_frames)


if __name__ == "__main__":
    # Sanity: shapes + param count.
    from .config import load_config
    cfg = load_config("configs/m1_fall.yaml")
    net = build_model(cfg.model)
    x = torch.zeros(4, 1, cfg.tensor.n_channels, cfg.tensor.window_frames)
    y = net(x)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"input  {tuple(x.shape)}  ->  output {tuple(y.shape)}")
    print(f"params: {n_params:,}")
