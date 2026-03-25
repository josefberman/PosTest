"""LSTM regressor: observation sequence -> per-event (east, north)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LSTMPath(nn.Module):
    def __init__(self, input_dim: int = 12, hidden: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, num_layers, batch_first=True, dropout=0.0
        )
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.lstm(x)
        return self.head(o)


def train_lstm(
    obs_df,
    true_df,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-2,
) -> LSTMPath:
    from path_estimation.nn.dataset import TrajectoryDataset

    ds = TrajectoryDataset(obs_df, true_df)
    x, y = ds[0]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    model = LSTMPath(input_dim=x.shape[-1]).to(device)
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
def predict_lstm_at_times(
    model: LSTMPath,
    obs_df,
    true_df,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate per-event predictions to 1 Hz true timeline."""
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
