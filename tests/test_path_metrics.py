"""Tests for trajectory metrics."""

import numpy as np

from path_estimation.metrics import compute_all_metrics, discrete_frechet, rmse_euclidean


def test_rmse_zero():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    m = rmse_euclidean(xy, xy)
    assert m < 1e-9


def test_frechet_identical():
    p = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert discrete_frechet(p, p) < 1e-9


def test_compute_all_metrics_runs():
    t = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    e = t + 0.1
    out = compute_all_metrics(t, e, max_points_frechet_dtw=50)
    assert "rmse_m" in out and out["rmse_m"] > 0
