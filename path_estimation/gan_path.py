"""Lightweight conditional generator (GAN-style: G + D) for 1 Hz paths from pooled obs features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from path_estimation.io import align_times_to_true
from path_estimation.nn.dataset import build_feature_matrix
from path_estimation.types import EstimationResult


class Generator(nn.Module):
    def __init__(self, in_dim: int, T: int, noise: int = 16) -> None:
        super().__init__()
        self.T = T
        self.net = nn.Sequential(
            nn.Linear(in_dim + noise, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, T * 2),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, z], dim=-1)).view(-1, self.T, 2)


class Discriminator(nn.Module):
    def __init__(self, T: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(T * 2, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, path: torch.Tensor) -> torch.Tensor:
        return self.net(path.view(path.size(0), -1))


def _pooled(obs_df: pd.DataFrame, dim: int) -> np.ndarray:
    feats, _ = build_feature_matrix(obs_df)
    v = feats.mean(axis=0)
    if len(v) < dim:
        v = np.pad(v, (0, dim - len(v)))
    return v[:dim].astype(np.float32)


def estimate_gan(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G,
    rng: np.random.Generator,
    *,
    device: torch.device | None = None,
    epochs: int = 5,
    noise_dim: int = 16,
    max_len: int = 256,
) -> EstimationResult:
    times_s, true_xy = align_times_to_true(true_df)
    T_full = len(times_s)
    stride = max(1, int(np.ceil(T_full / max_len)))
    times_ds = times_s[::stride]
    true_ds = true_xy[::stride]
    T = len(times_ds)
    dim_in = 12
    x_vec = torch.from_numpy(_pooled(obs_df, dim_in)).unsqueeze(0)
    center = true_ds.mean(axis=0, keepdims=True).astype(np.float32)
    y = torch.from_numpy((true_ds - center).astype(np.float32)).unsqueeze(0)
    if device is None:
        device = torch.device("cpu")
    x_vec = x_vec.to(device)
    y = y.to(device)

    gen = Generator(dim_in, T, noise=noise_dim).to(device)
    disc = Discriminator(T).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        z = torch.randn(1, noise_dim, device=device)
        fake = gen(x_vec, z)
        opt_d.zero_grad()
        loss_d_real = bce(disc(y), torch.ones(1, 1, device=device))
        loss_d_fake = bce(disc(fake.detach()), torch.zeros(1, 1, device=device))
        (loss_d_real + loss_d_fake).backward()
        opt_d.step()

        z2 = torch.randn(1, noise_dim, device=device)
        fake2 = gen(x_vec, z2)
        opt_g.zero_grad()
        loss_g = bce(disc(fake2), torch.ones(1, 1, device=device))
        loss_g.backward()
        opt_g.step()

    gen.eval()
    with torch.no_grad():
        samples = []
        for _ in range(8):
            z = torch.randn(1, noise_dim, device=device)
            samples.append(gen(x_vec, z).squeeze(0).cpu().numpy())
        arr = np.stack(samples, axis=0)
        est = np.mean(arr, axis=0) + center
        spread = np.std(arr, axis=0)
        std_e_ds = np.maximum(spread[:, 0], 1e-3)
        std_n_ds = np.maximum(spread[:, 1], 1e-3)

    east = np.interp(times_s, times_ds, est[:, 0])
    north = np.interp(times_s, times_ds, est[:, 1])
    if not np.all(np.isfinite(east)):
        east = np.full(T_full, float(center[0, 0]))
        north = np.full(T_full, float(center[0, 1]))
    std_e = np.interp(times_s, times_ds, std_e_ds)
    std_n = np.interp(times_s, times_ds, std_n_ds)

    return EstimationResult(
        times_s=times_s,
        east_m=east,
        north_m=north,
        std_east_m=std_e,
        std_north_m=std_n,
        meta={"method": "gan_path", "n_mc_samples": 8},
    )
