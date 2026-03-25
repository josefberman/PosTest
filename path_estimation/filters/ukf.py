"""Unscented Kalman filter (constant-velocity) — GPS linear measurement via filterpy UKF."""

from __future__ import annotations

import numpy as np
import pandas as pd
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

from path_estimation.io import align_times_to_true
from path_estimation.types import EstimationResult


def estimate_ukf_fused(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G,
    rng: np.random.Generator,
    *,
    sigma_acc: float = 0.4,
    sigma_gps: float = 6.0,
) -> EstimationResult:
    """UKF with GPS-only updates (same model as KF; UT for covariance propagation)."""
    times_s, _ = align_times_to_true(true_df)
    events = obs_df.sort_values("timestamp_s").reset_index(drop=True)
    if events.empty:
        raise ValueError("No observations.")
    if events[events["source_type"] == "gps"].empty:
        raise ValueError("UKF requires at least one GPS observation.")

    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        return F @ x

    def hx(x: np.ndarray) -> np.ndarray:
        return np.array([x[0], x[1]], dtype=float)

    def Qm(dt: float) -> np.ndarray:
        q = sigma_acc**2
        return np.diag([0.25 * q * dt**4] * 2 + [q * dt**2] * 2)

    points = MerweScaledSigmaPoints(n=4, alpha=0.3, beta=2.0, kappa=0.0)
    ukf = UKF(dim_x=4, dim_z=2, dt=1.0, fx=fx, hx=hx, points=points)
    ukf.Q = Qm(1.0)
    ukf.R = np.eye(2) * (sigma_gps**2)

    t_ev = events["timestamp_s"].to_numpy(float)
    g0 = events[events["source_type"] == "gps"].iloc[0]
    ukf.x = np.array([float(g0["gps_x"]), float(g0["gps_y"]), 0.0, 0.0])
    ukf.P = np.eye(4) * 100.0

    traj_t: list[float] = []
    traj_xy: list[np.ndarray] = []
    traj_std: list[np.ndarray] = []

    t_prev = float(t_ev[0])
    traj_t.append(t_prev)
    traj_xy.append(ukf.x[:2].copy())
    traj_std.append(np.sqrt(np.maximum(np.diag(ukf.P[:2, :2]), 0.0)))

    for k in range(1, len(t_ev)):
        t = float(t_ev[k])
        dt = max(t - t_prev, 1e-3)
        ukf.Q = Qm(dt)
        ukf.predict(dt=dt)
        row = events.iloc[k]
        if row["source_type"] == "gps":
            z = np.array([float(row["gps_x"]), float(row["gps_y"])], dtype=float)
            ukf.update(z)
        std_pair = np.sqrt(np.maximum(np.diag(ukf.P[:2, :2]), 0.0))
        traj_t.append(t)
        traj_xy.append(ukf.x[:2].copy())
        traj_std.append(std_pair)
        t_prev = t

    t_keys = np.asarray(traj_t, dtype=float)
    xs = np.vstack(traj_xy)
    east = np.interp(times_s, t_keys, xs[:, 0])
    north = np.interp(times_s, t_keys, xs[:, 1])
    std_arr = np.vstack(traj_std)
    std_e = np.interp(times_s, t_keys, std_arr[:, 0])
    std_n = np.interp(times_s, t_keys, std_arr[:, 1])

    return EstimationResult(
        times_s=times_s,
        east_m=east,
        north_m=north,
        std_east_m=std_e,
        std_north_m=std_n,
        meta={"method": "ukf_gps_cv"},
    )
