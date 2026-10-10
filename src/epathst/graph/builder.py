"""Builds the physical multiscale graph of one tissue section from its ST spots.

Nodes carry encoder embeddings only; edges are local/meso/macro with a degree cap per scale.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

SCALE_NAMES = ("local", "meso", "macro")


def build_multiscale_edges(
    coords_um: np.ndarray,
    local_radius_um: float,
    meso_radius_um: float,
    macro_radius_um: float,
    max_local: int,
    max_meso: int,
    max_macro: int,
) -> dict:
    """Bucket spot pairs by distance into degree-capped local/meso/macro edges.

    `coords_um` is (N, 2) in um. Returns edge_index (2, E), distance_um (E,) and scale (E,); every
    edge is listed in both directions so message passing is symmetric.
    """
    n = coords_um.shape[0]
    # coords_um only sets edge distance/bucket; it is never a node feature.
    tree = cKDTree(coords_um)
    pairs = tree.query_pairs(r=macro_radius_um, output_type="ndarray")

    src_list, dst_list, dist_list, scale_list = [], [], [], []
    if len(pairs) > 0:
        d = np.linalg.norm(coords_um[pairs[:, 0]] - coords_um[pairs[:, 1]], axis=1)

        # buckets are (lo, hi]; the degree cap applies per scale
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

    if src_list:
        edge_index = np.array([src_list, dst_list], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    return {
        "edge_index": edge_index,
        "distance_um": np.array(dist_list, dtype=np.float32),
        "scale": np.array(scale_list, dtype=object),
    }


def _cap_degree(
    pairs: np.ndarray, dists: np.ndarray, n_nodes: int, cap: int
) -> list[tuple[tuple[int, int], float]]:
    """Greedily keep the shortest pairs while both endpoints have fewer than `cap` edges."""
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


def _rbf_basis(
    distance_um: np.ndarray, macro_radius_um: float, n_rbf: int, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Shared RBF core: d_bar = d / macro_radius, phi_m = exp(-gamma * (d_bar - mu_m)^2) with mu_m
    evenly spaced in [0, 1]. Returns (d_bar, rbf).
    """
    d_norm = np.clip(distance_um / macro_radius_um, 0.0, 1.0)
    mu = np.linspace(0.0, 1.0, n_rbf)
    rbf = np.exp(-gamma * (d_norm[:, None] - mu[None, :]) ** 2).astype(np.float32)
    return d_norm.astype(np.float32), rbf


def union_edge_features(
    distance_um: np.ndarray,
    scale: np.ndarray,
    macro_radius_um: float,
    n_rbf: int = 16,
    gamma: float = 10.0,
) -> np.ndarray:
    """Union-graph edge feature, 20-D: distance + 16 RBF + 3-D scale one-hot.

    The explicit distance is kept because the RBFs alone under-determine it near bucket boundaries.
    Used by A7 and the random/shuffled/local-only ablations.
    """
    d_norm, rbf = _rbf_basis(distance_um, macro_radius_um, n_rbf, gamma)
    onehot = np.zeros((len(distance_um), len(SCALE_NAMES)), dtype=np.float32)
    for s_idx, s_name in enumerate(SCALE_NAMES):
        onehot[scale == s_name, s_idx] = 1.0
    return np.concatenate([d_norm[:, None], rbf, onehot], axis=1)


def per_scale_edge_features(
    distance_um: np.ndarray, macro_radius_um: float, n_rbf: int = 16, gamma: float = 10.0
) -> np.ndarray:
    """Scale-aware edge feature, 17-D: distance + 16 RBF. No scale one-hot, since the branch
    identity already encodes scale.
    """
    d_norm, rbf = _rbf_basis(distance_um, macro_radius_um, n_rbf, gamma)
    return np.concatenate([d_norm[:, None], rbf], axis=1)


# Back-compat alias. The old 19-D version omitted the distance scalar; new code should call
# union_edge_features.
def rbf_edge_features(
    distance_um: np.ndarray,
    scale: np.ndarray,
    macro_radius_um: float,
    n_rbf: int = 16,
    gamma: float = 10.0,
) -> np.ndarray:
    return union_edge_features(distance_um, scale, macro_radius_um, n_rbf, gamma)


def build_local_only_edges(coords_um: np.ndarray, local_radius_um: float, max_local: int) -> dict:
    """A6 local-only ablation: meso and macro radii are collapsed to local_radius_um."""
    return build_multiscale_edges(
        coords_um,
        local_radius_um,
        local_radius_um,
        local_radius_um,
        max_local,
        max_local,
        max_local,
    )


def split_edges_by_scale(edges: dict) -> dict[str, dict]:
    """Split build_multiscale_edges output into per-scale dicts over the same node indices."""
    out = {}
    for s in SCALE_NAMES:
        mask = edges["scale"] == s
        out[s] = {
            "edge_index": edges["edge_index"][:, mask],
            "distance_um": edges["distance_um"][mask],
            "scale": edges["scale"][mask],
        }
    return out
