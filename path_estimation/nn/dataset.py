"""Tensor dataset: observation sequence -> 1 Hz positions (supervised)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from path_estimation.io import align_times_to_true


def build_feature_matrix(obs_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Per-event features (fixed 12-D) and timestamps."""
    obs_df = obs_df.sort_values("timestamp_s").reset_index(drop=True)
    t = obs_df["timestamp_s"].to_numpy(float)
    t_max = float(np.max(t)) if len(t) else 1.0
    feats: list[list[float]] = []
    for _, row in obs_df.iterrows():
        tn = float(row["timestamp_s"]) / max(t_max, 1.0)
        src = row["source_type"]
        oh = [0.0, 0.0, 0.0]
        pad = [0.0] * 8
        if src == "gps":
            oh = [1.0, 0.0, 0.0]
            pad[0] = float(row["gps_x"])
            pad[1] = float(row["gps_y"])
        elif src == "circle":
            oh = [0.0, 1.0, 0.0]
            pad[0] = float(row["circle_x"])
            pad[1] = float(row["circle_y"])
            pad[2] = float(row["circle_r"])
        else:
            oh = [0.0, 0.0, 1.0]
            pad[0] = float(row["cell_tower_x"])
            pad[1] = float(row["cell_tower_y"])
            pad[2] = float(row["cell_r_min"])
            pad[3] = float(row["cell_r_max"])
            pad[4] = float(row["cell_theta_start"])
            pad[5] = float(row["cell_theta_end"])
        feats.append([tn] + oh + pad)
    return np.asarray(feats, dtype=np.float32), t


def interpolate_truth_to_events(
    t_ev: np.ndarray, true_df: pd.DataFrame
) -> np.ndarray:
    """True (east, north) linearly interpolated at event times."""
    tt, xy = align_times_to_true(true_df)
    ex = np.interp(t_ev, tt, true_df["true_x"].to_numpy(float))
    ny = np.interp(t_ev, tt, true_df["true_y"].to_numpy(float))
    return np.column_stack([ex, ny])


class TrajectoryDataset(Dataset):
    """Single-sequence dataset: full observation sequence, targets at each event."""

    def __init__(self, obs_df: pd.DataFrame, true_df: pd.DataFrame) -> None:
        self.feats, self.times = build_feature_matrix(obs_df)
        self.targets = interpolate_truth_to_events(self.times, true_df)

    def __len__(self) -> int:
        return max(1, len(self.feats))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.feats).float()
        y = torch.from_numpy(self.targets.astype(np.float32))
        return x, y
