"""Graph-destruction controls used by ablations A4 (random) and A5
(degree-preserving shuffle), plus the dose-response sweep. All functions
preserve the real graph's total (undirected) edge count, so any
performance difference is attributable to topology, not degree/capacity."""
from __future__ import annotations

import numpy as np


def _undirected_pairs(edge_index: np.ndarray) -> np.ndarray:
    src, dst = edge_index[0], edge_index[1]
    mask = src < dst
    return np.stack([src[mask], dst[mask]], axis=1)


def _symmetrize(pairs: np.ndarray) -> np.ndarray:
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return np.stack([src, dst], axis=0).astype(np.int64)


def random_graph(edge_index: np.ndarray, n_nodes: int, seed: int) -> np.ndarray:
    """A4: same node count, same (undirected) edge count, edges placed
    uniformly at random within the slide -- no physical-distance
    information used at all."""
    rng = np.random.RandomState(seed)
    n_edges = _undirected_pairs(edge_index).shape[0]
    max_possible = n_nodes * (n_nodes - 1) // 2
    n_edges = min(n_edges, max_possible)
    chosen = set()
    while len(chosen) < n_edges:
        i, j = rng.randint(0, n_nodes, size=2)
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        chosen.add(key)
    pairs = np.array(sorted(chosen), dtype=np.int64)
    return _symmetrize(pairs)


def degree_preserving_rewire(edge_index: np.ndarray, seed: int, n_swap_multiplier: int = 10) -> np.ndarray:
    """A5, the critical negative control: double-edge-swap rewiring.
    Exactly preserves every node's degree while destroying the real
    physical adjacency structure."""
    rng = np.random.RandomState(seed)
    pairs = _undirected_pairs(edge_index)
    edge_set = {tuple(p) for p in pairs}
    edges = [tuple(p) for p in pairs]
    n_attempts = n_swap_multiplier * max(1, len(edges))

    # double-edge swaps: (a,b),(c,d) -> (a,d),(c,b), rejecting swaps that
    # would create a duplicate or self-loop edge
    for _ in range(n_attempts):
        if len(edges) < 2:
            break
        i1, i2 = rng.randint(0, len(edges), size=2)
        if i1 == i2:
            continue
        a, b = edges[i1]
        c, d = edges[i2]
        if len({a, b, c, d}) < 4:
            continue
        new1, new2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if new1 in edge_set or new2 in edge_set or new1 == new2:
            continue
        edge_set.discard((a, b))
        edge_set.discard((c, d))
        edge_set.add(new1)
        edge_set.add(new2)
        edges[i1] = new1
        edges[i2] = new2

    pairs = np.array(sorted(edge_set), dtype=np.int64)
    return _symmetrize(pairs)


def partial_edge_replacement(edge_index: np.ndarray, n_nodes: int, fraction: float, seed: int,
                              distance_um: np.ndarray | None = None, placeholder_distance: float = 0.0) -> dict:
    """Dose-response sweep: replace `fraction` of the real physical
    (undirected) edges with random edges, keeping total edge count fixed.
    fraction in {0, 0.25, 0.5, 0.75, 1.0}."""
    rng = np.random.RandomState(seed)
    real_pairs = _undirected_pairs(edge_index)
    n_edges = len(real_pairs)
    n_replace = int(round(fraction * n_edges))

    # remember real distances so kept edges can carry them into the output
    real_dist_by_pair = {}
    if distance_um is not None:
        src, dst = edge_index[0], edge_index[1]
        mask = src < dst
        for i, j, d in zip(src[mask], dst[mask], distance_um[mask]):
            real_dist_by_pair[(int(i), int(j))] = float(d)

    if n_replace == 0:
        return {"edge_index": edge_index.copy(),
                "distance_um": distance_um.copy() if distance_um is not None else None}

    # drop a random subset of real edges, keep the rest
    keep_idx = rng.choice(n_edges, size=n_edges - n_replace, replace=False)
    kept = {tuple(p) for p in real_pairs[keep_idx]}

    # top back up to the original edge count with random replacement edges
    max_possible = n_nodes * (n_nodes - 1) // 2
    target_total = min(n_edges, len(kept) + max_possible - len(kept))
    while len(kept) < target_total:
        i, j = rng.randint(0, n_nodes, size=2)
        if i == j:
            continue
        pair = (min(i, j), max(i, j))
        kept.add(pair)
    pairs = np.array(sorted(kept), dtype=np.int64)
    out_edge_index = _symmetrize(pairs)

    # look up distance for kept real edges; placeholder for replacement edges
    out_distance = None
    if distance_um is not None:
        out_distance = np.array([
            real_dist_by_pair.get((min(int(i), int(j)), max(int(i), int(j))), placeholder_distance)
            for i, j in zip(out_edge_index[0], out_edge_index[1])
        ], dtype=np.float32)
    return {"edge_index": out_edge_index, "distance_um": out_distance}
