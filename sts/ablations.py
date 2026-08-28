"""Step 17 (Section 21): the required ablation grid A0-A9. Each ablation is
expressed as a (feature_source, graph_source) pair so scripts/14 can loop
over them with exactly the same patient folds, training procedure, and
evaluation code -- only the inputs differ.

    A0  mean-expression baseline           (no model)
    A1  ImageNet ConvNeXt patch-only       features=imagenet,  graph=none
    A2  GenBio-distilled ConvNeXt patch    features=distilled, graph=none   <- ŷ_patch,i
    A3  parameter-matched MLP              features=distilled, graph=none, wider MLP (no neighbors)
    A4  random graph PNA                   features=distilled, graph=random
    A5  degree-preserving shuffled graph   features=distilled, graph=shuffled   <- CRITICAL negative control
    A6  real LOCAL PNA                     features=distilled, graph=local
    A7  real MULTISCALE PNA                features=distilled, graph=multiscale
    A8  GenBio teacher + same head         features=genbio_teacher, graph=none
    A9  GenBio teacher + same PNA graph    features=genbio_teacher, graph=multiscale
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sts import destruction, graph_builder


@dataclass
class AblationSpec:
    name: str
    feature_source: str    # imagenet | distilled | genbio_teacher
    graph_source: str      # none | random | shuffled | local | multiscale
    description: str


ABLATION_GRID = [
    AblationSpec("A0_mean_expression", "none", "none", "predict training gene means -- sanity floor"),
    AblationSpec("A1_imagenet_patch", "imagenet", "none", "ImageNet ConvNeXt, no distillation, patch-only"),
    AblationSpec("A2_distilled_patch", "distilled", "none", "GenBio-distilled ConvNeXt, patch-only (=y_patch)"),
    AblationSpec("A3_matched_mlp", "distilled", "none", "parameter-matched MLP, same features, no neighbors"),
    AblationSpec("A4_random_graph", "distilled", "random", "same nodes/edge count, uniform random edges"),
    AblationSpec("A5_shuffled_graph", "distilled", "shuffled", "degree-preserving rewire -- critical negative control"),
    AblationSpec("A6_local_pna", "distilled", "local", "real LOCAL-only physical graph"),
    AblationSpec("A7_multiscale_pna", "distilled", "multiscale", "real local+meso+macro physical graph"),
    AblationSpec("A8_genbio_patch", "genbio_teacher", "none", "GenBio teacher embedding, patch-only, fair comparison"),
    AblationSpec("A9_genbio_pna", "genbio_teacher", "multiscale", "GenBio teacher embedding, same PNA graph"),
]


def build_graph_for_ablation(coords_um: np.ndarray, graph_source: str, cfg: dict, seed: int) -> dict:
    g = cfg["graph"]
    if graph_source == "multiscale":
        edges = graph_builder.build_multiscale_edges(coords_um, g["local_radius_um"], g["meso_radius_um"],
                                                       g["macro_radius_um"], g["max_local_neighbors"],
                                                       g["max_meso_neighbors"], g["max_macro_neighbors"])
    elif graph_source == "local":
        edges = graph_builder.build_local_only_edges(coords_um, g["local_radius_um"], g["max_local_neighbors"])
    elif graph_source in ("random", "shuffled"):
        base = graph_builder.build_multiscale_edges(coords_um, g["local_radius_um"], g["meso_radius_um"],
                                                      g["macro_radius_um"], g["max_local_neighbors"],
                                                      g["max_meso_neighbors"], g["max_macro_neighbors"])
        n = coords_um.shape[0]
        if graph_source == "random":
            edge_index = destruction.random_graph(base["edge_index"], n, seed)
        else:
            edge_index = destruction.degree_preserving_rewire(base["edge_index"], seed)
        # random/shuffled edges have no meaningful physical distance -> zero-fill edge features
        edges = {"edge_index": edge_index, "distance_um": np.zeros(edge_index.shape[1], dtype=np.float32),
                 "scale": np.array(["local"] * edge_index.shape[1], dtype=object)}
    elif graph_source == "none":
        edges = {"edge_index": np.zeros((2, 0), dtype=np.int64), "distance_um": np.zeros(0, dtype=np.float32),
                 "scale": np.zeros(0, dtype=object)}
    else:
        raise ValueError(f"unknown graph_source: {graph_source}")
    return edges


def edge_features_for_ablation(edges: dict, graph_source: str, cfg: dict) -> np.ndarray:
    g = cfg["graph"]
    if graph_source in ("multiscale", "local", "random", "shuffled"):
        return graph_builder.rbf_edge_features(edges["distance_um"], edges["scale"], g["macro_radius_um"],
                                                g["n_rbf"], g["rbf_gamma"])
    return np.zeros((0, g["n_rbf"] + 3), dtype=np.float32)


def matched_mlp_hidden_dim(pna_param_count: int, in_dim: int, out_dim: int) -> int:
    """A3: pick an MLP hidden width so its parameter count is approximately
    matched to the real PNA model (Section 21: 'parameter-matched MLP'),
    solving in_dim*h + h + h*out_dim + out_dim ~= pna_param_count for h."""
    # quadratic-free approx since h appears linearly for a single hidden layer
    h = max(8, int(round((pna_param_count - out_dim) / max(1, in_dim + out_dim + 1))))
    return h
