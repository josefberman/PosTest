"""Reusable plotting helpers for synthetic path/observation visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from pyproj import Transformer
except ImportError:  # pragma: no cover
    Transformer = None  # type: ignore


def _lon_lat_to_web_mercator(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project WGS84 lon/lat (degrees) to Web Mercator (EPSG:3857) meters."""
    if Transformer is None:
        raise ImportError("plotting on maps requires the 'pyproj' package.")
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = tr.transform(
        np.asarray(lon_deg, dtype=float),
        np.asarray(lat_deg, dtype=float),
    )
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _extent_with_padding(
    xs: np.ndarray,
    ys: np.ndarray,
    pad_frac: float = 0.08,
) -> Tuple[float, float, float, float]:
    """Return ``(xmin, xmax, ymin, ymax)`` with relative padding."""
    if xs.size == 0 or ys.size == 0:
        return -1000.0, 1000.0, -1000.0, 1000.0
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    return (
        xmin - pad_frac * dx,
        xmax + pad_frac * dx,
        ymin - pad_frac * dy,
        ymax + pad_frac * dy,
    )


def _try_add_basemap(ax: plt.Axes, *, alpha: float = 0.85) -> bool:
    """Overlay OpenStreetMap tiles if ``contextily`` is available."""
    try:
        import contextily as ctx
    except ImportError:
        return False
    try:
        ctx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=ctx.providers.OpenStreetMap.Mapnik,
            alpha=alpha,
        )
        return True
    except Exception:
        return False


def plot_true_path(df: pd.DataFrame, output_path: Path, show: bool = False) -> None:
    """Plot only the ground-truth trajectory in local meters.

    Args:
        df: DataFrame with ``true_x`` and ``true_y`` (e.g. dense 1 Hz path or samples).
        output_path: Destination image path.
        show: If True, display the figure interactively.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df["true_x"], df["true_y"], color="black", linewidth=1.4, label="True path")
    ax.set_title("Ground Truth Path")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)


def plot_observations_only(df: pd.DataFrame, output_path: Path, show: bool = False) -> None:
    """Plot only observed points by source type.

    Args:
        df: Dataset DataFrame containing source-specific observation columns.
        output_path: Destination image path.
        show: If True, display the figure interactively.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    gps = df[df["source_type"] == "gps"]
    cir = df[df["source_type"] == "circle"]
    cel = df[df["source_type"] == "cell_sector"]

    if not gps.empty:
        ax.scatter(gps["gps_x"], gps["gps_y"], s=12, alpha=0.75, label="GPS")
    if not cir.empty:
        ax.scatter(cir["circle_x"], cir["circle_y"], s=16, alpha=0.65, label="Circle center")
    if not cel.empty:
        ax.scatter(
            cel["cell_tower_x"],
            cel["cell_tower_y"],
            s=28,
            alpha=0.85,
            marker="^",
            label="Cell tower (used)",
        )

    ax.set_title("Observations Only (No True Path)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    if show:
        plt.show()
    plt.close(fig)


def plot_true_path_on_map(
    track_df: pd.DataFrame,
    output_path: Path,
    *,
    show_basemap: bool = True,
    show_path: bool = True,
    title: str = "Ground truth path (London, OSM)",
    show: bool = False,
) -> None:
    """Plot the dense true path on a Web Mercator map (optionally with OSM tiles).

    Args:
        track_df: DataFrame with columns ``lon``, ``lat`` (WGS84 degrees), e.g. from
            ``*_true_path_track.csv``.
        output_path: Destination image path.
        show_basemap: If True, try to draw OpenStreetMap tiles under the path.
        show_path: If True, draw the polyline.
        title: Figure title.
        show: If True, display the figure interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    lon = track_df["lon"].to_numpy(dtype=float)
    lat = track_df["lat"].to_numpy(dtype=float)
    xm, ym = _lon_lat_to_web_mercator(lon, lat)
    if show_path:
        ax.plot(xm, ym, color="crimson", linewidth=2.0, label="True path", zorder=3)
    xmin, xmax, ymin, ymax = _extent_with_padding(xm, ym)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    if show_basemap:
        if not _try_add_basemap(ax):
            ax.set_facecolor("#e8e4dc")
    ax.set_xlabel("Easting (Web Mercator, m)")
    ax.set_ylabel("Northing (Web Mercator, m)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_observations_on_map(
    df: pd.DataFrame,
    output_path: Path,
    *,
    show_basemap: bool = True,
    show_gps: bool = True,
    show_circle: bool = True,
    show_cell_tower: bool = True,
    title: str = "Observations (London, OSM)",
    show: bool = False,
) -> None:
    """Plot observation layers on a map (GPS / circle / synthetic cell tower), no path.

    Args:
        df: Event DataFrame with ``gps_lon``/``gps_lat``, ``circle_lon``/``circle_lat``,
            ``cell_tower_lon``/``cell_tower_lat`` as applicable per ``source_type``.
        output_path: Destination image path.
        show_basemap: If True, try OpenStreetMap tiles.
        show_gps: Layer toggle for GPS fixes.
        show_circle: Layer toggle for circle centers.
        show_cell_tower: Layer toggle for (synthetic) cell tower positions.
        title: Figure title.
        show: If True, display interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    gps = df[df["source_type"] == "gps"] if show_gps else pd.DataFrame()
    cir = df[df["source_type"] == "circle"] if show_circle else pd.DataFrame()
    cel = df[df["source_type"] == "cell_sector"] if show_cell_tower else pd.DataFrame()

    if not gps.empty:
        gx, gy = _lon_lat_to_web_mercator(
            gps["gps_lon"].to_numpy(dtype=float),
            gps["gps_lat"].to_numpy(dtype=float),
        )
        ax.scatter(gx, gy, s=14, alpha=0.85, c="tab:blue", label="GPS", zorder=4)
        xs.append(gx)
        ys.append(gy)
    if not cir.empty:
        cx, cy = _lon_lat_to_web_mercator(
            cir["circle_lon"].to_numpy(dtype=float),
            cir["circle_lat"].to_numpy(dtype=float),
        )
        ax.scatter(cx, cy, s=22, alpha=0.8, c="tab:orange", label="Circle center", zorder=4)
        xs.append(cx)
        ys.append(cy)
    if not cel.empty:
        tx, ty = _lon_lat_to_web_mercator(
            cel["cell_tower_lon"].to_numpy(dtype=float),
            cel["cell_tower_lat"].to_numpy(dtype=float),
        )
        ax.scatter(tx, ty, s=36, alpha=0.9, c="tab:green", marker="^", label="Cell tower (synth)", zorder=4)
        xs.append(tx)
        ys.append(ty)

    if xs:
        all_x = np.concatenate(xs)
        all_y = np.concatenate(ys)
        xmin, xmax, ymin, ymax = _extent_with_padding(all_x, all_y)
    else:
        xmin, xmax, ymin, ymax = -1000.0, 1000.0, -1000.0, 1000.0
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    if show_basemap:
        if not _try_add_basemap(ax):
            ax.set_facecolor("#e8e4dc")
    ax.set_xlabel("Easting (Web Mercator, m)")
    ax.set_ylabel("Northing (Web Mercator, m)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_map_with_layers(
    track_df: pd.DataFrame,
    events_df: pd.DataFrame,
    output_path: Path,
    *,
    show_basemap: bool = True,
    show_true_path: bool = True,
    show_gps: bool = True,
    show_circle: bool = True,
    show_cell_tower: bool = True,
    title: str = "Path and observations (London, OSM)",
    show: bool = False,
) -> None:
    """Combined map: optional OSM basemap, true path, and observation layers.

    Args:
        track_df: Dense path with ``lon``, ``lat`` (WGS84).
        events_df: Per-event rows with geo columns for observations.
        output_path: Destination image path.
        show_basemap: If True, try OpenStreetMap tiles.
        show_true_path: Layer toggle for the ground-truth polyline.
        show_gps: Layer toggle for GPS.
        show_circle: Layer toggle for circle centers.
        show_cell_tower: Layer toggle for synthetic cell towers.
        title: Figure title.
        show: If True, display interactively.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    if show_true_path and not track_df.empty:
        lon = track_df["lon"].to_numpy(dtype=float)
        lat = track_df["lat"].to_numpy(dtype=float)
        xm, ym = _lon_lat_to_web_mercator(lon, lat)
        ax.plot(xm, ym, color="crimson", linewidth=2.0, label="True path", zorder=3)
        xs.append(xm)
        ys.append(ym)

    gps = events_df[events_df["source_type"] == "gps"] if show_gps else pd.DataFrame()
    cir = events_df[events_df["source_type"] == "circle"] if show_circle else pd.DataFrame()
    cel = events_df[events_df["source_type"] == "cell_sector"] if show_cell_tower else pd.DataFrame()

    if not gps.empty:
        gx, gy = _lon_lat_to_web_mercator(
            gps["gps_lon"].to_numpy(dtype=float),
            gps["gps_lat"].to_numpy(dtype=float),
        )
        ax.scatter(gx, gy, s=14, alpha=0.85, c="tab:blue", label="GPS", zorder=4)
        xs.append(gx)
        ys.append(gy)
    if not cir.empty:
        cx, cy = _lon_lat_to_web_mercator(
            cir["circle_lon"].to_numpy(dtype=float),
            cir["circle_lat"].to_numpy(dtype=float),
        )
        ax.scatter(cx, cy, s=22, alpha=0.8, c="tab:orange", label="Circle center", zorder=4)
        xs.append(cx)
        ys.append(cy)
    if not cel.empty:
        tx, ty = _lon_lat_to_web_mercator(
            cel["cell_tower_lon"].to_numpy(dtype=float),
            cel["cell_tower_lat"].to_numpy(dtype=float),
        )
        ax.scatter(tx, ty, s=36, alpha=0.9, c="tab:green", marker="^", label="Cell tower (synth)", zorder=4)
        xs.append(tx)
        ys.append(ty)

    if xs:
        all_x = np.concatenate(xs)
        all_y = np.concatenate(ys)
        xmin, xmax, ymin, ymax = _extent_with_padding(all_x, all_y)
    else:
        xmin, xmax, ymin, ymax = -1000.0, 1000.0, -1000.0, 1000.0
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    if show_basemap:
        if not _try_add_basemap(ax):
            ax.set_facecolor("#e8e4dc")
    ax.set_xlabel("Easting (Web Mercator, m)")
    ax.set_ylabel("Northing (Web Mercator, m)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
