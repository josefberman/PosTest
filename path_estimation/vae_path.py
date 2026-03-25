"""Conditional VAE: pooled observation features -> 1 Hz trajectory (supervised on same run)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from path_estimation.io import align_times_to_true
from path_estimation.nn.dataset import build_feature_matrix
from path_estimation.types import EstimationResult


class PathVAE(nn.Module):
    def __init__(self, in_dim: int, T: int, latent: int = 32) -> None:
        super().__init__()
        self.T = T
        self.enc = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 64))
        self.fc_mu = nn.Linear(64, latent)
        self.fc_logvar = nn.Linear(64, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, 128),
            nn.ReLU(),
            nn.Linear(128, T * 2),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * lv)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        out = self.dec(z).view(-1, self.T, 2)
        return out, mu, lv


def _pooled_features(obs_df: pd.DataFrame, dim_pad: int) -> np.ndarray:
    feats, _ = build_feature_matrix(obs_df)
    vec = feats.mean(axis=0)
    if len(vec) < dim_pad:
        vec = np.pad(vec, (0, dim_pad - len(vec)))
    return vec[:dim_pad].astype(np.float32)


def estimate_vae(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G,
    rng: np.random.Generator,
    *,
    device: torch.device | None = None,
    epochs: int = 15,
    latent: int = 24,
    max_len: int = 256,
) -> EstimationResult:
    times_s, true_xy = align_times_to_true(true_df)
    T_full = len(times_s)
    stride = max(1, int(np.ceil(T_full / max_len)))
    times_ds = times_s[::stride]
    true_ds = true_xy[::stride]
    T = len(times_ds)
    dim_pad = 12
    x_vec = torch.from_numpy(_pooled_features(obs_df, dim_pad)).unsqueeze(0)
    y = torch.from_numpy(true_ds.astype(np.float32)).unsqueeze(0)
    if device is None:
        device = torch.device("cpu")
    x_vec = x_vec.to(device)
    y = y.to(device)

    model = PathVAE(in_dim=dim_pad, T=T, latent=latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred, mu, lv = model(x_vec)
        rec = torch.nn.functional.mse_loss(pred, y)
        kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        loss = rec + 1e-3 * kl
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred, mu, lv = model(x_vec)
        est = pred.squeeze(0).cpu().numpy()
        sig = float(torch.exp(0.5 * lv).mean().cpu())
        sigma = np.full(T_full, sig, dtype=float)

    east = np.interp(times_s, times_ds, est[:, 0])
    north = np.interp(times_s, times_ds, est[:, 1])

    return EstimationResult(
        times_s=times_s,
        east_m=east,
        north_m=north,
        std_east_m=sigma,
        std_north_m=sigma.copy(),
        meta={"method": "vae_path"},
    )
