"""Real walking routes on OpenStreetMap pedestrian network (central London).

Uses OSMnx to download/cache a walk graph, samples shortest paths between random
nodes until length is sufficient, then walks along edge geometries at a fixed
speed. Positions are returned in the same local ENU meter frame as
:mod:`geo_reference` (east/north relative to the London anchor).

If OSMnx is unavailable or routing fails, callers should fall back to synthetic
polylines.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from geo_reference import lon_lat_to_local_enu_meters

# Central London — Westminster / City / South Bank (walkable, dense streets)
_BBOX_WEST = -0.135
_BBOX_EAST = -0.085
_BBOX_SOUTH = 51.485
_BBOX_NORTH = 51.515

_CACHE_DIR = Path(__file__).resolve().parent / "cache"
_GRAPHML_NAME = "london_walk_bbox.graphml"


def _cumdist_xy(xy: np.ndarray) -> np.ndarray:
    """Cumulative distance along a vertex chain (first point at 0)."""
    if len(xy) < 2:
        return np.zeros(len(xy), dtype=float)
    seg = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def _truncate_polyline(xy: np.ndarray, max_len_m: float) -> np.ndarray:
    """Truncate ``xy`` so total arc length is at most ``max_len_m``."""
    c = _cumdist_xy(xy)
    if c[-1] <= max_len_m + 1e-6:
        return xy
    j = int(np.searchsorted(c, max_len_m, side="right") - 1)
    j = max(0, min(j, len(xy) - 2))
    denom = c[j + 1] - c[j]
    if denom < 1e-9:
        return xy[: j + 1]
    a = (max_len_m - c[j]) / denom
    last = (1.0 - a) * xy[j] + a * xy[j + 1]
    return np.vstack([xy[: j + 1], last])


def _positions_at_distances(xy: np.ndarray, cumdist: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Interpolate positions at arc lengths ``s`` (meters) along ``xy``."""
    s = np.asarray(s, dtype=float)
    s = np.clip(s, 0.0, cumdist[-1])
    j = np.searchsorted(cumdist, s, side="right") - 1
    j = np.clip(j, 0, len(xy) - 2)
    denom = cumdist[j + 1] - cumdist[j]
    denom = np.where(denom < 1e-9, 1.0, denom)
    a = (s - cumdist[j]) / denom
    return (1.0 - a)[:, None] * xy[j] + a[:, None] * xy[j + 1]


def _polyline_xy_from_route(G, route: List) -> np.ndarray:
    """Build dense (x, y) in projected CRS from a node route."""
    coords: List[Tuple[float, float]] = []
    for u, v in zip(route[:-1], route[1:]):
        if not G.has_edge(u, v):
            continue
        ed = G[u][v]
        best = min(ed.values(), key=lambda d: d.get("length", float("inf")))
        geom = best.get("geometry")
        xu, yu = G.nodes[u]["x"], G.nodes[u]["y"]
        xv, yv = G.nodes[v]["x"], G.nodes[v]["y"]
        if geom is not None:
            seg = list(geom.coords)
        else:
            seg = [(xu, yu), (xv, yv)]
        if not coords:
            coords.extend(seg)
        else:
            if len(seg) and seg[0] == coords[-1]:
                coords.extend(seg[1:])
            else:
                coords.extend(seg)
    if len(coords) < 2:
        return np.zeros((0, 2), dtype=float)
    arr = np.asarray(coords, dtype=float)
    # Remove consecutive duplicates
    keep = np.ones(len(arr), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(arr, axis=0)) > 1e-6, axis=1)
    return arr[keep]


def _proj_xy_to_enu_columns(xs: np.ndarray, ys: np.ndarray, crs) -> np.ndarray:
    """Projected CRS sample points -> (east_m, north_m) columns."""
    from pyproj import Transformer

    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(np.asarray(xs), np.asarray(ys))
    east, north = lon_lat_to_local_enu_meters(np.asarray(lon), np.asarray(lat))
    return np.column_stack((east, north))


def _load_walk_graph():
    """Load or download OSM walk graph; return projected MultiDiGraph."""
    import networkx as nx
    import osmnx as ox

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / _GRAPHML_NAME

    if cache_file.exists():
        G = ox.load_graphml(cache_file)
    else:
        G = ox.graph_from_bbox(
            bbox=(_BBOX_WEST, _BBOX_SOUTH, _BBOX_EAST, _BBOX_NORTH),
            network_type="walk",
            simplify=True,
        )
        ox.save_graphml(G, cache_file)

    Gp = ox.project_graph(G.copy())
    if Gp.number_of_nodes() < 10:
        raise RuntimeError("OSM walk graph has too few nodes.")
    return Gp


def _find_route(
    G,
    target_length_m: float,
    rng: np.random.Generator,
    path_kind: str,
) -> Optional[List]:
    """Return a list of node IDs whose path length (m) is at least ``target_length_m``."""
    import networkx as nx

    nodes = list(G.nodes)
    if len(nodes) < 2:
        return None

    candidates: List[Tuple[List, float, int]] = []
    n_trial = 450
    for _ in range(n_trial):
        u, v = rng.choice(nodes, size=2, replace=False)
        try:
            path = nx.shortest_path(G, u, v, weight="length")
        except nx.NetworkXNoPath:
            continue
        plen = nx.shortest_path_length(G, u, v, weight="length")
        if plen >= target_length_m * 0.88:
            candidates.append((path, plen, len(path)))

    if not candidates:
        return None

    if path_kind == "complex":
        candidates.sort(key=lambda x: (-x[2], -x[1]))
    else:
        candidates.sort(key=lambda x: (x[2], -x[1]))

    return candidates[0][0]


def positions_enu_along_osm_walk(
    times_s: np.ndarray,
    rng: np.random.Generator,
    path_kind: str,
    walk_speed_mps: float = 1.35,
) -> Optional[np.ndarray]:
    """Sample positions along a real London walking route (local ENU meters).

    Args:
        times_s: Timestamps (seconds) at which to evaluate position.
        rng: Random generator (controls start/end pair).
        path_kind: ``simple`` prefers fewer corners; ``complex`` prefers more corners.
        walk_speed_mps: Constant speed along the path.

    Returns:
        Array of shape ``(len(times_s), 2)`` with columns ``(east_m, north_m)``, or
        ``None`` if routing failed.
    """
    times_s = np.asarray(times_s, dtype=float)
    duration = float(np.max(times_s)) if times_s.size else 0.0
    target_len_m = duration * walk_speed_mps * 1.02

    try:
        G = _load_walk_graph()
    except Exception as exc:
        warnings.warn(f"Could not load OSM walk graph: {exc}", UserWarning)
        return None

    route = _find_route(G, target_len_m, rng, path_kind)
    if route is None:
        warnings.warn(
            "Could not find a long enough London street route; use synthetic path.",
            UserWarning,
        )
        return None

    xy_proj = _polyline_xy_from_route(G, route)
    if len(xy_proj) < 2:
        return None

    xy_proj = _truncate_polyline(xy_proj, target_len_m)
    cum = _cumdist_xy(xy_proj)
    if cum[-1] < target_len_m * 0.5:
        return None

    crs = G.graph.get("crs")
    if crs is None:
        return None

    cum_proj = _cumdist_xy(xy_proj)
    s_query = np.clip(times_s * walk_speed_mps, 0.0, cum_proj[-1])
    xy_samp = _positions_at_distances(xy_proj, cum_proj, s_query)
    return _proj_xy_to_enu_columns(xy_samp[:, 0], xy_samp[:, 1], crs)
