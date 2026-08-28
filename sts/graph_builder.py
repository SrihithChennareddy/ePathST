"""Physical multiscale graph construction (Final_Project.txt Section 18 /
Step 15). One graph per tissue section (never edges across WSIs). Nodes are
ST spots with the GenBio-distilled ConvNeXt embedding as their only
feature. Edges are physical-distance buckets (local/meso/macro), each
capped at its own max-neighbor count, with a 19-D edge feature: 16
RBF-encoded normalized distance + 3-D one-hot scale code. Raw absolute
(x, y) node coordinates are never used as a feature.

Adapted from the identical, already-validated implementation in
STS_PoC/PoC_5/crcspatial/graph_builder.py -- the spec's radii/caps/RBF
count are numerically identical (150/300/600 um, 8/12/16, 16 RBFs)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

SCALE_NAMES = ("local", "meso", "macro")


def build_multiscale_edges(coords_um: np.ndarray, local_radius_um: float, meso_radius_um: float,
                            macro_radius_um: float, max_local: int, max_meso: int, max_macro: int) -> dict:
    """coords_um: (N, 2) physical micrometer coordinates. Returns dict with
    'edge_index' (2, E) int64 [src, dst], 'distance_um' (E,), 'scale' (E,)
    string array in SCALE_NAMES. Edges are directed both ways so message
    passing is symmetric."""
    n = coords_um.shape[0]
    tree = cKDTree(coords_um)
    pairs = tree.query_pairs(r=macro_radius_um, output_type="ndarray")

    src_list, dst_list, dist_list, scale_list = [], [], [], []
    if len(pairs) > 0:
        d = np.linalg.norm(coords_um[pairs[:, 0]] - coords_um[pairs[:, 1]], axis=1)
        for scale_name, lo, hi, cap in (
            ("local", 0.0, local_radius_um, max_local),
            ("meso", local_radius_um, meso_radius_um, max_meso),
            ("macro", meso_radius_um, macro_radius_um, max_macro),
        ):
            mask = (d > lo) & (d <= hi)
            bucket_pairs, bucket_d = pairs[mask], d[mask]
            if len(bucket_pairs) == 0:
                continue
            kept = _cap_degree(bucket_pairs, bucket_d, n, cap)
            for (i, j), dij in kept:
                src_list += [i, j]
                dst_list += [j, i]
                dist_list += [dij, dij]
                scale_list += [scale_name, scale_name]

    edge_index = np.array([src_list, dst_list], dtype=np.int64) if src_list else np.zeros((2, 0), dtype=np.int64)
    return {
        "edge_index": edge_index,
        "distance_um": np.array(dist_list, dtype=np.float32),
        "scale": np.array(scale_list, dtype=object),
    }


def _cap_degree(pairs: np.ndarray, dists: np.ndarray, n_nodes: int, cap: int) -> list[tuple[tuple[int, int], float]]:
    order = np.argsort(dists)
    degree = np.zeros(n_nodes, dtype=np.int32)
    kept, seen = [], set()
    for idx in order:
        i, j = int(pairs[idx, 0]), int(pairs[idx, 1])
        if degree[i] >= cap or degree[j] >= cap:
            continue
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        kept.append(((i, j), float(dists[idx])))
        degree[i] += 1
        degree[j] += 1
    return kept


def rbf_edge_features(distance_um: np.ndarray, scale: np.ndarray, macro_radius_um: float,
                       n_rbf: int = 16, gamma: float = 10.0) -> np.ndarray:
    """d_bar = d/macro_radius, 16 RBFs phi_m(d)=exp(-gamma*(d-mu_m)^2) with
    mu_m evenly spaced in [0,1], plus a 3-D one-hot scale code -> 19-D edge
    feature (Section 18)."""
    d_norm = np.clip(distance_um / macro_radius_um, 0.0, 1.0)
    mu = np.linspace(0.0, 1.0, n_rbf)
    rbf = np.exp(-gamma * (d_norm[:, None] - mu[None, :]) ** 2).astype(np.float32)

    onehot = np.zeros((len(distance_um), len(SCALE_NAMES)), dtype=np.float32)
    for s_idx, s_name in enumerate(SCALE_NAMES):
        onehot[scale == s_name, s_idx] = 1.0
    return np.concatenate([rbf, onehot], axis=1)


def build_local_only_edges(coords_um: np.ndarray, local_radius_um: float, max_local: int) -> dict:
    """A6/local-only ablation: identical to build_multiscale_edges but with
    meso/macro radii collapsed to local_radius_um (so only local edges
    exist) -- used for the local-vs-multiscale PNA comparison (Section 22.A)."""
    return build_multiscale_edges(coords_um, local_radius_um, local_radius_um, local_radius_um,
                                   max_local, max_local, max_local)
