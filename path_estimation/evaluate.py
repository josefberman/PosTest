"""Run estimators, compute metrics, save JSON summary and figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from path_estimation.filters import (
    estimate_ekf_fused,
    estimate_kf_gps,
    estimate_particle_filter,
    estimate_ukf_fused,
)
from path_estimation.graph_stitch import estimate_graph_stitch
from path_estimation.gnn.estimate import estimate_gnn
from path_estimation.gan_path import estimate_gan
from path_estimation.graph_utils import get_projected_graph
from path_estimation.hmm_map_match import estimate_hmm_map_match
from path_estimation.io import align_times_to_true, load_observations_csv, load_true_path_csv
from path_estimation.metrics import compute_all_metrics
from path_estimation.nn.lstm_model import predict_lstm_at_times, train_lstm
from path_estimation.nn.transformer_model import predict_transformer_at_times, train_transformer
from path_estimation.types import EstimationResult
from path_estimation.vae_path import estimate_vae
from path_estimation.viz import plot_estimation_enu, plot_estimation_map

EstimatorFn = Callable[..., EstimationResult]

METHOD_REGISTRY: Dict[str, EstimatorFn] = {
    "dijkstra": lambda o, t, G, r: estimate_graph_stitch(o, t, G, r, mode="dijkstra"),
    "astar": lambda o, t, G, r: estimate_graph_stitch(o, t, G, r, mode="astar"),
    "hmm": estimate_hmm_map_match,
    "kf": estimate_kf_gps,
    "ekf": estimate_ekf_fused,
    "ukf": estimate_ukf_fused,
    "particle": estimate_particle_filter,
    "lstm": None,  # special-cased (torch)
    "transformer": None,
    "gnn": estimate_gnn,
    "vae": estimate_vae,
    "gan": estimate_gan,
}


def _estimate_lstm(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G,
    rng: np.random.Generator,
    device: Optional[str] = None,
) -> EstimationResult:
    dev = torch_device(device)
    model = train_lstm(obs_df, true_df, dev, epochs=20)
    times_s, xy = predict_lstm_at_times(model, obs_df, true_df, dev)
    return EstimationResult(
        times_s=times_s,
        east_m=xy[:, 0],
        north_m=xy[:, 1],
        meta={"method": "lstm"},
    )


def _estimate_transformer(
    obs_df: pd.DataFrame,
    true_df: pd.DataFrame,
    G,
    rng: np.random.Generator,
    device: Optional[str] = None,
) -> EstimationResult:
    dev = torch_device(device)
    model = train_transformer(obs_df, true_df, dev, epochs=25)
    times_s, xy = predict_transformer_at_times(model, obs_df, true_df, dev)
    return EstimationResult(
        times_s=times_s,
        east_m=xy[:, 0],
        north_m=xy[:, 1],
        meta={"method": "transformer"},
    )


def torch_device(name: Optional[str] = None):
    import torch as _torch

    if name:
        return _torch.device(name)
    return _torch.device("cuda" if _torch.cuda.is_available() else "cpu")


def run_evaluation(
    observations_csv: Path,
    true_path_csv: Path,
    output_dir: Path,
    methods: Optional[List[str]] = None,
    *,
    plot: bool = True,
    plot_map: bool = False,
    device: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, Dict]:
    """Run selected methods; write ``metrics.json`` and figures under ``output_dir``."""
    obs_df = load_observations_csv(observations_csv)
    true_df = load_true_path_csv(true_path_csv)
    times_s, true_xy = align_times_to_true(true_df)
    G = get_projected_graph()
    rng = np.random.default_rng(seed)

    if methods is None:
        methods = list(METHOD_REGISTRY.keys())

    out: Dict[str, Dict] = {}
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    dev = torch_device(device)

    for name in methods:
        name = name.strip().lower()
        try:
            if name == "lstm":
                res = _estimate_lstm(obs_df, true_df, G, rng, device=device)
            elif name == "transformer":
                res = _estimate_transformer(obs_df, true_df, G, rng, device=device)
            elif name == "gnn":
                res = estimate_gnn(obs_df, true_df, G, rng, device=dev)
            elif name == "vae":
                res = estimate_vae(obs_df, true_df, G, rng, device=dev)
            elif name == "gan":
                res = estimate_gan(obs_df, true_df, G, rng, device=dev)
            else:
                fn = METHOD_REGISTRY.get(name)
                if fn is None:
                    raise KeyError(f"Unknown method: {name}")
                res = fn(obs_df, true_df, G, rng)
        except Exception as exc:
            out[name] = {"error": str(exc)}
            continue

        est_xy = np.column_stack((res.east_m, res.north_m))
        m = compute_all_metrics(true_xy, est_xy)
        m["meta"] = res.meta
        out[name] = m

        if plot:
            plot_estimation_enu(
                true_df,
                res,
                obs_df,
                fig_dir / f"{name}_path_enu.png",
                title=f"{name.upper()} — estimated vs true (ENU)",
                show_observations=True,
            )
        if plot_map:
            plot_estimation_map(
                true_df,
                res,
                fig_dir / f"{name}_path_map.png",
                title=f"{name.upper()} — map",
            )

    summary_path = output_dir / "metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)

    return out


def print_summary(summary: Dict[str, Dict]) -> None:
    for k, v in summary.items():
        if "error" in v:
            print(f"{k}: ERROR — {v['error']}")
        else:
            rmse = v.get("rmse_m", float("nan"))
            print(f"{k}: RMSE={rmse:.3f} m  MAE={v.get('mae_m', float('nan')):.3f} m")
