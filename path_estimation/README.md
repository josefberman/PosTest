# Path estimation

Reconstructs a trajectory (local ENU meters) from mixed asynchronous observations (`gps`, `circle`, `cell_sector`) and compares against `*_true_path.csv` (1 Hz).

## CLI

```bash
python -m path_estimation \
  --observations data/dataset_observations.csv \
  --true-path data/dataset_true_path.csv \
  --output-dir path_estimation_runs/run1 \
  --seed 42 \
  --device cpu
```

- `--methods`: comma-separated list (default: all implemented methods).
- `--no-plots`: skip PNG figures.
- `--map-plots`: also write Web Mercator basemap figures (may fetch tiles).

Outputs:

- `metrics.json` — per-method scores (RMSE, MAE, Hausdorff, discrete Fréchet, DTW, length ratio, endpoint error, …).
- `figures/<method>_path_enu.png` — ENU overlay (true vs estimated; optional σ ellipses for probabilistic outputs).

## Methods

| ID | Description |
|----|-------------|
| `dijkstra` | Snap observations to OSM nodes; stitch shortest paths; uniform speed along polyline. |
| `astar` | Same as Dijkstra with `networkx.astar_path` (heuristic). |
| `hmm` | Viterbi–style map match over k-nearest nodes per observation. |
| `kf` | 4D constant-velocity Kalman filter; GPS updates only. |
| `ekf` | EKF with GPS + circle radius + cell radial cues. |
| `ukf` | Unscented Kalman filter (GPS) via `filterpy`. |
| `particle` | Bootstrap particle filter with mixed likelihoods. |
| `lstm` | LSTM on observation sequence; supervised fit on the same run. |
| `transformer` | Small Transformer encoder. |
| `gnn` | GCN node classifier on OSM subgraph; guided snaps + stitch. |
| `vae` | Conditional VAE on pooled observations → downsampled path + interpolation. |
| `gan` | Conditional generator + discriminator; MC samples for spread. |

## Metrics (see `metrics.py`)

- Pointwise: RMSE, MAE (Euclidean and per-axis).
- Sequence: Hausdorff (subsampled), discrete Fréchet, DTW (subsampled).
- Global: path length ratio, combined endpoint error.

## Dependencies

See project `requirements.txt` (`scipy`, `filterpy`, `torch`, `torch-geometric`, …).

## Tests

```bash
python -m pytest tests/test_path_metrics.py tests/test_graph_shortest.py -q
```
