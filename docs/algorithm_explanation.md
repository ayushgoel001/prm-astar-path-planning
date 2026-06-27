# Algorithm Explanation

## 1. Problem Statement

Given a 2D binary image representing a robot environment, where black pixels denote obstacles and white pixels denote free space, the objective is to find a collision-free path from a start coordinate to a goal coordinate.

This is a standard motion-planning problem in robotics. In this project, the environment is represented as a 2D occupancy map and the planner searches for a feasible path while avoiding obstacles and unsafe regions.

---

## 2. Map Representation

The planner works with binary maps:

```text
0   = obstacle
255 = free space
```

All input maps are loaded as grayscale images and thresholded into binary obstacle maps. This keeps the planning pipeline simple and consistent across generated maps and uploaded maps.

Before planning, obstacle clearance can be applied to the map. This creates a safer planning region by shrinking the available free space around obstacles.

---

## 3. Obstacle Clearance

A robot is not a point mass. If the robot has a physical radius, it should not pass directly next to walls or obstacles. To model this safety margin, the project applies obstacle clearance before sampling nodes or checking roadmap edges.

This is implemented using morphological erosion on the free-space map:

```python
kernel = np.ones((2 * clearance + 1, 2 * clearance + 1), dtype=np.uint8)
safe_map = cv2.erode(obstacle_map, kernel)
```

Eroding the white free-space region is equivalent to inflating the black obstacle region. Planning is then performed on the clearance-applied map.

---

## 4. Start and Goal Validation

Before building the roadmap, the start and goal coordinates are validated against both the original map and the clearance-applied map.

A coordinate is rejected if:

```text
1. It lies outside the map bounds.
2. It lies inside an obstacle in the original map.
3. It becomes invalid after obstacle clearance is applied.
```

This prevents misleading planning failures where the algorithm appears unable to find a path, but the actual issue is an invalid start or goal point.

---

## 5. Probabilistic Roadmap Method

Probabilistic Roadmap Method is a sampling-based planner. It approximates the connectivity of the free space by constructing a graph of collision-free sampled nodes.

In this project, PRM is used as the primary planner.

### Roadmap Construction

```text
1. Sample N nodes from free-space pixels.
2. Add start and goal nodes to the sampled node set.
3. Build a KD-Tree over all nodes.
4. For each node, query its k nearest neighbors.
5. Collision-check each candidate edge.
6. Add an undirected edge if the segment is collision-free.
```

The roadmap is undirected because robot motion between two collision-free points is reversible in this map representation. Therefore, when an edge `u -> v` is valid, the reverse edge `v -> u` is also added.

### Query Phase

After the roadmap is built, A* search is used to find a path from the start node to the goal node.

```text
1. Start and goal are inserted into the roadmap.
2. A* searches over the roadmap graph.
3. The final output is a sequence of roadmap nodes forming a collision-free path.
```

### Properties

| Property        | Description                                                                             |
| --------------- | --------------------------------------------------------------------------------------- |
| Completeness    | Probabilistically complete under suitable sampling assumptions                          |
| Optimality      | Not guaranteed optimal; path quality depends on sampling density and graph connectivity |
| Query type      | Suitable for multi-query settings when the roadmap is reused                            |
| Best suited for | Static environments with repeated planning queries                                      |

---

## 6. Free-Space Sampling

The function `sample_free_space()` samples nodes uniformly from valid free-space pixels.

It uses a `border_margin` parameter to avoid sampling too close to the image boundary. This is separate from obstacle clearance.

```text
obstacle clearance -> avoids obstacles and walls
border margin      -> avoids sampling directly on image edges
```

The sampler uses a local NumPy random generator:

```python
rng = np.random.default_rng(seed)
chosen_indices = rng.choice(len(valid_positions), size=num_samples, replace=False)
```

This keeps sampling reproducible without modifying NumPy's global random state.

---

## 7. KD-Tree Nearest-Neighbor Search

After sampling the nodes, the planner must decide which nearby nodes should be connected.

A naive approach would compare every node with every other node, which is expensive for large node sets. This project uses `scipy.spatial.KDTree` for nearest-neighbor lookup.

```python
kdtree = KDTree(nodes)
_, standard_neighbors = kdtree.query(nodes, k=k_neighbors + 1)
```

The extra neighbor is requested because the closest point to a node is the node itself. The implementation removes this self-neighbor before checking candidate edges.

Start and goal nodes are treated as anchor nodes and are queried with a wider neighbor search. This improves their chance of connecting to the roadmap, especially in constrained map regions.

---

## 8. Collision Checking

For every candidate roadmap edge, the planner checks whether the straight-line segment between the two nodes intersects an obstacle.

The segment is sampled at approximately one-pixel resolution:

```python
distance = np.linalg.norm(node2 - node1)
num_checks = max(int(np.ceil(distance)) + 1, 2)
coords = np.rint(np.linspace(node1, node2, num=num_checks)).astype(int)
```

The sampled coordinates are then checked against the obstacle map using vectorized NumPy indexing.

An edge is considered invalid if:

```text
1. Any sampled point lies outside the map.
2. Any sampled point lies on an obstacle pixel.
```

This makes long-edge collision checking more reliable than using a fixed number of samples for every edge.

---

## 9. A* Search

A* is an informed graph-search algorithm used to find a low-cost path in a weighted graph.

In this project:

```text
graph nodes = PRM sampled nodes
graph edges = collision-free roadmap connections
edge cost   = Euclidean distance between connected nodes
heuristic   = Euclidean distance to the goal
```

### A* Update Rule

For each neighbor of the current node, the algorithm checks whether a cheaper path to that neighbor has been found.

```python
new_cost = cost_so_far[current] + distance(current, neighbor)

if new_cost < cost_so_far.get(neighbor, float("inf")):
    cost_so_far[neighbor] = new_cost
    priority = new_cost + heuristic(neighbor, goal)
    came_from[neighbor] = current
```

This update rule is essential. A node should be updated if a better path to it is discovered.

### Heuristic

The heuristic is Euclidean distance:

```text
h(node, goal) = sqrt((node.row - goal.row)^2 + (node.col - goal.col)^2)
```

This is a natural heuristic for 2D path planning because the straight-line distance is a lower bound on the path distance in an obstacle-filled map.

---

## 10. Path Smoothing

The path returned by A* is a sequence of roadmap nodes. Since the roadmap is sampled, the path may contain unnecessary intermediate waypoints.

The project includes shortcut-based smoothing. It tries to connect non-adjacent points directly and removes intermediate nodes if the shortcut is collision-free.

```text
original path:  start -> a -> b -> c -> goal
smoothed path:  start -> b -> goal
```

The smoothed path is accepted only when the replacement segment is collision-free.

---

## 11. RRT Comparison

Rapidly-exploring Random Tree is included as a comparison algorithm.

RRT is a single-query tree-based planner. It grows a tree from the start node by sampling random points and extending toward them.

```text
1. Initialize tree with the start node.
2. Sample a random point, occasionally biased toward the goal.
3. Find the nearest tree node.
4. Extend from the nearest node toward the sample.
5. Add the new node if the extension is collision-free.
6. Stop when the tree reaches the goal region.
```

### PRM and RRT Comparison

| Aspect           | PRM + A*                              | RRT                                        |
| ---------------- | ------------------------------------- | ------------------------------------------ |
| Planner type     | Roadmap-based                         | Tree-based                                 |
| Query type       | Better suited for reusable roadmaps   | Single-query                               |
| Path quality     | Often shorter on the tested maps      | Often longer without rewiring or smoothing |
| Runtime behavior | Roadmap construction can be expensive | Can be fast in open spaces                 |
| Best suited for  | Static maps with repeated queries     | Single-query exploration                   |

The project includes `scripts/compare_algorithms.py` to compare both approaches on the same maps and start-goal pairs.

---

## 12. Complexity Analysis

Let:

```text
N = number of PRM nodes
k = number of nearest neighbors attempted per node
C = cost of collision checking one edge
V = number of graph vertices
E = number of graph edges
H, W = map height and width
```

| Operation                             | Approximate Complexity                                    |
| ------------------------------------- | --------------------------------------------------------- |
| Free-space mask construction          | O(H * W)                                                  |
| Sampling from valid free-space pixels | O(N) after valid positions are collected                  |
| KD-Tree construction                  | O(N log N) average case                                   |
| Nearest-neighbor queries              | Approximately O(N log N) average case for fixed k         |
| Roadmap edge validation               | O(N * k * C)                                              |
| A* search                             | O((V + E) log V)                                          |
| RRT planning                          | Depends on iterations and nearest-neighbor implementation |

For image maps, the most expensive part is usually roadmap construction because collision checking is performed for many candidate edges.

---

## 13. Benchmarking

The benchmark script evaluates PRM + A* across fixed map scenarios and records:

```text
number of nodes
number of roadmap edges
A* explored nodes
path found or not
path length
roadmap build time
A* search time
total planning time
```

The comparison script evaluates PRM + A* against RRT and saves results as CSV files.

```text
outputs/benchmark_results.csv
outputs/comparison_results.csv
```

These result files are used to keep the project measurable and reproducible.

---

## 14. Limitations

The current implementation assumes:

```text
1. Static 2D binary maps.
2. Point-to-point planning in image coordinates.
3. Straight-line local connections.
4. No robot dynamics or velocity constraints.
5. No moving obstacles.
```

The PRM path quality depends on the number of samples, the neighbor count, and the obstacle layout. Narrow passages may require more samples or stronger start-goal connectivity.

---

## 15. Future Work

Possible extensions include:

```text
1. PRM* for asymptotically optimal roadmap planning.
2. RRT* for rewiring-based tree optimization.
3. Multi-query mode with cached roadmaps.
4. Weighted terrain costs instead of binary obstacles.
5. Dynamic obstacle handling with D* Lite or kinodynamic planners.
6. 3D occupancy-grid planning for drone navigation.
7. Interactive point selection in the Streamlit interface.
```
