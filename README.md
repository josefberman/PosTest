# PosTest

Trajectory reconstruction from **mixed asynchronous observations** (GPS points, circular uncertainty regions, cellular sector annuli) in a **local East–North–Up (ENU)** frame, evaluated against **1 Hz ground truth**. The repo includes **synthetic data generation**, a **`path_estimation`** library with many estimators, and a **`method_eval`** harness for multi-run benchmarks.

---

## Repository layout

| Path | Role |
|------|------|
| `path_estimation/` | Installable-style package: I/O, filters (KF/EKF/UKF/particle), graph map-matching, PyTorch LSTM/Transformer, GNN, metrics, plotting, CLI (`python -m path_estimation`). |
| `generate_synthetic_datasets.py` | Builds paired `*_observations.csv` + `*_true_path.csv` and optional map figures (OSM-backed when available). |
| `geo_reference.py`, `london_street_path.py`, `plotting_utils.py` | Reference frame, optional London OSM ground paths, shared plotting helpers. |
| `method_eval/` | Batch script `run_method_evaluation.py`: many runs, per-method CSVs, per-metric bar charts. |
| `tests/` | `pytest` tests for metrics and graph shortest paths. |
| `requirements.txt` | Python dependencies. |

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Heavy optional stack: **OSMnx** (street graphs), **PyTorch** / **PyTorch Geometric** (neural methods), **contextily** (basemap tiles). Neural methods fall back or skip if CUDA is unavailable; use `--device cpu` to force CPU.

---

## Coordinate system

Positions are in **local ENU meters** (`east`, `north`) with a fixed reference (see `geo_reference.py` and CSV metadata such as `reference_origin_lat` / `reference_origin_lon`). Longitude/latitude columns are derived for mapping. **ENU** is a flat tangent-plane approximation suitable for city-scale paths.

---

## Data files

- **Observations** (`*_observations.csv`): time-ordered rows with `source_type` in `gps`, `circle`, `cell_sector` and source-specific columns (`gps_x`/`gps_y`, circle center + radius, cell tower + annulus + sector angles, etc.).
- **True path** (`*_true_path.csv`): 1 Hz `timestamp_s`, `true_x`, `true_y`, `lon`, `lat`, plus metadata.

Loaders and alignment helpers live in `path_estimation/io.py`.

**Tiny examples:** `path_estimation/sample_observations.csv` and `path_estimation/sample_true_path.csv` (paired 61 s run).

---

## Synthetic datasets

Generates one dataset (or use CLI args for duration, seed, output dir):

```bash
python generate_synthetic_datasets.py \
  --output-dir ./data/run_01 \
  --dataset-id run_01 \
  --duration-s 3600 \
  --seed 42
```

When **OSMnx** can fetch a London pedestrian graph, ground truth may follow a **real OSM walk**; otherwise a **synthetic polyline** is used. Outputs include CSVs and PNGs (true path, observations, optional basemap layers).

---

## Path estimation (single evaluation)

```bash
python -m path_estimation \
  --observations path/to/run_observations.csv \
  --true-path path/to/run_true_path.csv \
  --output-dir path_estimation_runs/run1 \
  --methods dijkstra,kf,ekf,ukf,particle,lstm,transformer,gnn \
  --seed 42 \
  --device cpu
```

- **`--no-plots`**: skip per-method ENU figures (metrics still computed).
- **`--map-plots`**: also write Web Mercator figures (may download tiles).

**Outputs:** `metrics.json` (per method: RMSE, MAE, axis MAE, Hausdorff, discrete Fréchet, DTW, path lengths, length ratio, endpoint error, plus `meta`), and `figures/<method>_path_enu.png` (true vs estimated; optional observation overlays). No uncertainty ellipses are drawn on ENU plots.

**Without a true path file**, use the Python API **`path_estimation.estimate_paths_only(observations_csv, road_graph, methods, ...)`**, which returns `EstimationResult` objects (no RMSE). Supervised methods (`lstm`, `transformer`, `gnn`) require **`evaluate_path_estimation`** with ground truth.

### Implemented methods (summary)

| ID | Idea |
|----|------|
| `dijkstra`, `astar` | Snap observations to projected OSM graph nodes; shortest-path stitch; timing along polyline. |
| `hmm` | Map-matching / Viterbi-style discrete state sequence over candidate nodes. |
| `kf` | 4D constant-velocity Kalman filter; **fused** updates from GPS (tight) and weaker position cues from circle/cell centers. |
| `ekf` | Extended KF with nonlinear geometry for mixed observations. |
| `ukf` | Unscented Kalman filter with fused measurement schedule (e.g. via `filterpy`). |
| `particle` | Bootstrap particle filter with mixed likelihoods. |
| `lstm`, `transformer` | Supervised sequence models on the observation stream; trained on the same run, then interpolated to 1 Hz. |
| `gnn` | Graph neural network over an OSM subgraph + decode to path. |

Details and defaults are in `path_estimation/README.md` and the source modules under `path_estimation/`.

---

## Metrics

All trajectories are resampled or aligned to the true 1 Hz timeline before comparison. Implemented in `path_estimation/metrics.py`:

- **Pointwise:** RMSE, MAE (Euclidean and per-axis).
- **Sequence:** symmetric Hausdorff (subsampled), discrete Fréchet, DTW (subsampled for cost).
- **Global:** path length, **length ratio** (estimated / true), **endpoint error** (start + end).

---

## Method evaluation (batch)

`method_eval/run_method_evaluation.py` generates **N** independent synthetic runs (`run_1` … `run_N`), runs **`run_evaluation`** on each, and aggregates results:

- Per run: `method_eval/run_<i>/` — synthetic CSVs, `metrics.json`, `figures/`.
- Aggregated: `method_eval/<method>.csv` — metrics as rows, `run_1` … `run_N` as columns.
- Plots: `method_eval/metric_bars/<metric>.png` — one bar chart per metric (methods on x-axis, mean ± std across runs).

Example:

```bash
python method_eval/run_method_evaluation.py --n-runs 10 --duration-s 3600 --device cpu
```

Flags include `--data-seed`, `--eval-seed`, `--methods`, `--no-plots` (skip per-run evaluation figures), `--no-bar-plot` (skip bar charts).

---

## Tests

```bash
python -m pytest tests/test_path_metrics.py tests/test_graph_shortest.py -q
```

---

## License

See [LICENSE](LICENSE) (MIT).

---

## See also

- `path_estimation/README.md` — CLI options and method table (package-focused).
- `path_estimation/metrics.py` — metric definitions.
- `path_estimation/evaluate.py` — method registry and orchestration.
