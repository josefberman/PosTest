"""Lightweight Transformer encoder for per-event position regression."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerPath(nn.Module):
    def __init__(self, input_dim: int = 12, d_model: int = 64, nhead: int = 4, nlayers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128, batch_first=True, dropout=0.1
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=nlayers)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        z = self.pos(z)
        z = self.enc(z)
        return self.head(z)


def train_transformer(
    obs_df,
    true_df,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
) -> TransformerPath:
    from path_estimation.nn.dataset import TrajectoryDataset

    ds = TrajectoryDataset(obs_df, true_df)
    x, y = ds[0]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    model = TransformerPath(input_dim=x.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_transformer_at_times(
    model: TransformerPath,
    obs_df,
    true_df,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    from path_estimation.nn.dataset import TrajectoryDataset
    from path_estimation.io import align_times_to_true

    ds = TrajectoryDataset(obs_df, true_df)
    x, _ = ds[0]
    x = x.unsqueeze(0).to(device)
    pred = model(x).squeeze(0).cpu().numpy()
    t_ev = ds.times
    times_s, _ = align_times_to_true(true_df)
    east = np.interp(times_s, t_ev, pred[:, 0])
    north = np.interp(times_s, t_ev, pred[:, 1])
    return times_s, np.column_stack([east, north])
