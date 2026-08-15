# PRM + A* Path Planning on 2D Occupancy Maps

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python implementation of sampling-based path planning on binary image maps. The primary planner combines a Probabilistic Roadmap (PRM), KDTree neighbor lookup, and A* search; the repository also includes a goal-biased RRT baseline, CLI and Streamlit interfaces, reproducible benchmarks, and automated tests.

## Key engineering highlights

- End-to-end PRM pipeline: seeded free-space sampling, start/goal anchors, KDTree neighbor candidates, collision-checked roadmap construction, and A* with Euclidean costs and heuristic.
- Conservative supercover-style raster collision checking for integer PRM nodes and floating-point RRT nodes, including corner contacts, boundary-aligned segments, and half-integer endpoints.
- Shared map thresholding, square-kernel pixel clearance, descriptive planning-point validation, and collision-safe shortcut smoothing.
- **73 automated tests** covering algorithms, edge cases, benchmark methodology, and end-to-end integration.
- Deterministic 500-sample roadmaps produced **2,233–5,312 accepted edges**, with **8–355 unique A* expanded nodes** and **1.5–19.6 ms A* search time** across four scenarios.
- PRM + A* and RRT were evaluated over **60 matched-seed planning attempts**; PRM produced shorter successful paths in all three tested comparison scenarios.
- Reproducible CLI and Streamlit workflows with pytest CI on Python 3.10 and 3.11.

## Planning pipeline

```text
grayscale image
    -> shared binary threshold (0 obstacle, 255 free)
    -> optional square-kernel clearance
    -> validate start and goal
    -> sample PRM configurations
    -> insert start/goal anchors
    -> KDTree neighbor candidates
    -> collision-checked undirected roadmap
    -> A* roadmap search
    -> optional collision-safe shortcut smoothing
```

## Engineering details

- PRM samples distinct valid free-space pixels using a local seeded NumPy generator.
- Start and goal are inserted as graph anchors and receive a wider neighbor-candidate query (`3 * k`); every proposed edge still passes the shared collision checker.
- A* discards stale heap entries and reports unique valid graph vertices expanded. It returns a shortest path on the constructed roadmap, not necessarily the globally shortest continuous-space path.
- Clearance erodes free space using a square `(2c + 1) x (2c + 1)` kernel. It is a conservative pixel-grid margin, not an exact circular robot footprint.
- Shortcut smoothing removes waypoints only when the replacement segment is collision-free. The app exposes smoothing for PRM + A* only.
- RRT is a seeded comparison baseline without rewiring or smoothing; its nearest-neighbor implementation is intentionally not optimized.

This is ordinary PRM, not PRM*. Finite roadmaps can be disconnected; under standard assumptions, PRM is probabilistically complete as sampling increases but is not globally optimal.

## Example outputs

| PRM path on Maze 1 Hard | PRM + A* and RRT comparison |
| --- | --- |
| ![PRM path on Maze 1 Hard](assets/prm_maze_hard.png) | ![PRM and RRT comparison](assets/prm_vs_rrt_maze_hard.png) |

## Repository structure

```text
app.py                         Streamlit interface
src/                           planners and shared map/collision utilities
scripts/run_planner.py         PRM + A* command-line entry point
scripts/benchmark.py           deterministic PRM benchmark
scripts/compare_algorithms.py  repeated PRM-versus-RRT comparison
tests/                         unit, regression, methodology, and integration tests
examples/                      synthetic occupancy maps
outputs/                       tracked benchmark CSVs
assets/                        selected result figures
.github/workflows/ci.yml       pytest CI for Python 3.10 and 3.11
```

## Installation

```bash
git clone https://github.com/ayushgoel001/prm-astar-path-planning.git
cd prm-astar-path-planning
pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

## Usage

Run PRM + A* from the command line:

```bash
python scripts/run_planner.py --map examples/maze_1.png --nodes 500 --neighbors 20 --start 5,235 --goal 350,450 --clearance 3 --seed 42 --smooth --no-show
```

Coordinates use **`(row, column)`** order. `--nodes` specifies sampled PRM configurations; start and goal are added separately. `--no-show` suppresses the display window, and `--save PATH` writes a result image.

Run the Streamlit interface:

```bash
streamlit run app.py
```

The app supports built-in or uploaded maps, validated start/goal coordinates, PRM + A* or RRT planning, configurable clearance and seeds, and downloadable result figures.

## Tests and CI

```bash
python -m pytest tests/ -v
```

The 73-test suite covers shared binarization, clearance, planning-point validation, conservative collision traversal, PRM construction, A*, RRT, smoothing, benchmark aggregation, and the complete PRM-to-A* pipeline. GitHub Actions runs the suite on Python 3.10 and 3.11.

## Deterministic PRM benchmark

`scripts/benchmark.py` runs one unsmoothed PRM + A* plan per scenario with seed 42, 500 sampled configurations, `k = 20`, and clearance 3. Adding start and goal produces 502 graph vertices.

| Scenario | Samples | Graph vertices | Edges | A* expanded nodes | Path length (px) | A* search (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Maze 1 Easy | 500 | 502 | 3908 | 8 | 220.1 | 1.8 |
| Maze 1 Hard | 500 | 502 | 3867 | 43 | 626.8 | 1.5 |
| Maze 2 Grid | 500 | 502 | 2233 | 355 | 824.9 | 19.6 |
| Maze 3 Obstacles | 500 | 502 | 5312 | 92 | 637.8 | 11.1 |

Complete build/search timing data is available in [outputs/benchmark_results.csv](outputs/benchmark_results.csv); timings are machine-specific.

## Matched-seed PRM versus RRT comparison

`scripts/compare_algorithms.py` runs 10 fresh attempts per algorithm and scenario using matched seeds 42–51: 60 attempts across three scenarios. PRM runtime includes sampling, roadmap construction, anchor connection, and A*; RRT runtime includes complete planning. Runtime statistics include all attempts, while path-length statistics include successful runs only. Neither planner is smoothed.

PRM produced shorter successful paths across all three tested scenarios, while RRT showed lower single-query runtime in this implementation. Results describe this implementation and fixed configuration, not universal planner performance. Full success-rate, runtime, path-length, and graph/tree-size data—including failed attempts—is available in [outputs/comparison_results.csv](outputs/comparison_results.csv).

Run both reports with:

```bash
python scripts/benchmark.py
python scripts/compare_algorithms.py
```

## Limitations

- Static 2D binary occupancy maps and straight-line local motion only.
- Square raster clearance rather than exact robot geometry or dynamics.
- Finite-sample PRM roadmaps can be disconnected, particularly around narrow passages.
- The CLI and app rebuild a roadmap for each query rather than caching a multi-query roadmap.
- Conservative Python collision traversal dominates measured roadmap-construction time.
- The RRT baseline rebuilds a KDTree for nearest-node queries and is not optimized.
- Neither ordinary PRM nor baseline RRT guarantees continuous-space global optimality.

See [docs/algorithm_explanation.md](docs/algorithm_explanation.md) for implementation details and algorithm guarantees.

## License

Released under the [MIT License](LICENSE).
