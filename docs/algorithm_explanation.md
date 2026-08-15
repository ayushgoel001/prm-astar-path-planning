# Algorithm Explanation

## Problem and map model

The project plans point-to-point paths on static 2D occupancy images. Coordinates use `(row, column)` order and the internal binary representation is:

```text
0   = obstacle
255 = free space
```

File loading and Streamlit uploads share one binarization rule: grayscale values at or below 127 become obstacles, and values above 127 become free space.

Start and goal coordinates are validated before planning. A point is rejected with a descriptive error if it is outside the map, occupied in the original map, or removed by the selected clearance.

## Pixel-grid obstacle clearance

Clearance is applied by eroding white free space with an all-ones square kernel:

```python
kernel_size = 2 * clearance + 1
safe_map = cv2.erode(obstacle_map, np.ones((kernel_size, kernel_size), np.uint8))
```

This is equivalent to conservative obstacle inflation on the pixel grid. The margin is square, so it must not be interpreted as an exact circular robot radius or an exact continuous configuration-space model. Sampling and collision checks use the clearance-applied map; validation also retains the original map so errors can distinguish an obstacle from a clearance violation.

## Probabilistic Roadmap

The primary planner is an ordinary Probabilistic Roadmap (PRM):

```text
1. Uniformly sample N distinct valid free-space pixels.
2. Insert start and goal as additional anchor vertices.
3. Build a KDTree over all vertices.
4. Query nearby candidate neighbors for each vertex.
5. Reject every candidate edge that fails collision checking.
6. Add accepted edges in both directions.
7. Run A* from the start anchor to the goal anchor.
```

Sampling uses `numpy.random.default_rng(seed)` for reproducibility and does not modify NumPy's global random state. A configurable border margin excludes image-edge samples independently of obstacle clearance.

Start and goal are not included in the requested sample count. For example, 500 sampled configurations plus two anchors produce 502 total graph vertices. Anchors receive a wider neighbor-candidate query (`3 * k`, capped by graph size), but every anchor connection passes the same collision checker as other roadmap edges.

A finite roadmap can remain disconnected, especially around narrow passages. Under standard PRM assumptions, the probability of finding a feasible path approaches one as the number of samples increases: ordinary PRM is probabilistically complete. It is not globally optimal, and this implementation is not PRM*.

## KDTree neighbor candidates

`scipy.spatial.KDTree` accelerates nearest-neighbor candidate lookup compared with an explicit all-pairs scan. The tree only proposes nearby vertices; it does not determine whether an edge is valid. Collision checking remains mandatory.

KDTree performance depends on the data, dimension, query size, and library implementation. The project therefore does not rely on a universal worst-case speed claim.

## Conservative raster collision checking

The shared collision checker uses a supercover-style traversal. Raster cells are modeled as unit squares centered on integer pixel coordinates. The segment is parameterized over `t` in `[0, 1]` and processes row/column boundary events in order until the next event lies beyond the segment. Endpoint-adjacent cells are checked explicitly, so an endpoint on a half-integer boundary or corner terminates safely without weakening coverage.

At an exact grid-corner crossing, both side-adjacent cells and the diagonal cell are checked. A segment on a cell boundary checks cells on both sides. Out-of-bounds endpoints or traversed cells are collisions. These conservative rules prevent thin obstacles from being skipped by rounded samples and support both integer PRM vertices and floating-point RRT vertices.

This is a raster occupancy test, not exact continuous computational geometry. Its cost scales with the number of grid cells touched by the segment.

## A* roadmap search

Each roadmap edge is weighted by Euclidean distance, and A* uses Euclidean distance to the goal as its heuristic:

```text
edge(u, v) = ||position(u) - position(v)||2
h(v)       = ||position(v) - position(goal)||2
```

Heap entries retain their path cost so stale entries can be discarded. A vertex is expanded only once after a valid non-stale pop. The reported `expanded_nodes` metric is the number of unique valid graph vertices expanded, including the goal when it is reached.

Because Euclidean distance is consistent with Euclidean edge weights, A* returns a shortest path on the constructed roadmap. That result is not necessarily the globally shortest path in continuous free space: roadmap sampling and connectivity constrain the available routes.

## Collision-safe shortcut smoothing

The optional greedy smoother looks for non-adjacent path waypoints that can be connected directly. It removes intermediate waypoints only when the proposed shortcut passes the shared collision checker. Empty, one-point, and two-point paths are unchanged; nontrivial smoothing preserves the start and goal and cannot increase Euclidean path length.

The Streamlit and CLI smoothing option applies to PRM + A*. The benchmark and PRM-versus-RRT comparison report unsmoothed planner paths.

## RRT baseline

The comparison planner is a seeded, goal-biased Rapidly-exploring Random Tree:

```text
1. Initialize the tree at start.
2. Randomly sample a point, with occasional goal samples.
3. Find the nearest current tree node.
4. Extend by at most the configured step size.
5. Add the node only if the extension is collision-free.
6. Connect to goal when it is within the goal radius and the final edge is free.
```

If start and goal are the same valid configuration, RRT immediately returns that one-point path. The implementation is a comparison baseline: it has no rewiring, path smoothing, or optimized nearest-neighbor structure. Its measured performance describes this code on the tested maps and seeds, not RRT implementations in general.

## Implementation-aware complexity

Let `N` be the graph-vertex count, `k` the requested candidate-neighbor count, `E` the accepted undirected edges, and `C` the cell-traversal cost of one proposed edge.

| Operation | Practical cost description |
| --- | --- |
| Binary mask and clearance | Linear in image pixels, multiplied by morphology implementation factors |
| Free-position collection and sampling | Scans the image, then selects the requested samples |
| KDTree build and queries | Commonly near `N log N` for this low-dimensional use, but data- and implementation-dependent |
| Roadmap edge validation | Approximately `N * k` candidate checks plus wider anchor queries; each check costs `C` |
| A* | Heap-based graph search, commonly described as `O((N + E) log N)` |
| RRT | Scales with attempted iterations; this baseline rebuilds a KDTree for each nearest-node query and performs segment checks |

In the measured Python implementation, conservative collision checking across many candidate edges makes roadmap construction the practical bottleneck; A* search is much smaller in the recorded deterministic runs.

## Benchmark semantics

`scripts/benchmark.py` runs the complete PRM + A* pipeline on fixed scenarios with seed 42. It records sampled configurations, total graph vertices, accepted edges, unique A* expansions, path status and length, roadmap build time, search time, and total time.

`scripts/compare_algorithms.py` performs 10 fresh runs per algorithm and scenario using matched seeds 42-51. PRM time includes sampling, roadmap construction, anchor connection, and A*. RRT time includes complete planning. Runtime statistics use all attempted runs; path-length statistics use successful runs only. Both scripts validate result sanity before writing:

```text
outputs/benchmark_results.csv
outputs/comparison_results.csv
```

Timing values are machine- and run-specific. The comparison is an implementation benchmark, not a universal ranking of PRM and RRT.

## Limitations

- Static 2D binary maps only; no terrain costs or moving obstacles.
- Point planning with no robot dynamics, turning limits, or velocity constraints.
- Square raster clearance rather than exact robot geometry.
- Straight-line local connections checked conservatively on pixels.
- Finite PRM roadmaps may be disconnected and are rebuilt for each current CLI or app query.
- Ordinary PRM and baseline RRT do not provide continuous-space optimality.
