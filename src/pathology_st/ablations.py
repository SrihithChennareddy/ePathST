"""Defines the A0-A9 ablation grid (feature_source/graph_source/backend triples, see ABLATION_GRID) so
the CV driver loops over every ablation with identical folds/training/eval, plus per-ablation graph/edge/MLP-width helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pathology_st import destruction, graph_builder


@dataclass
class AblationSpec:
    name: str
    feature_source: str  # imagenet | distilled
    graph_source: str    # none | random | shuffled | local | multiscale
    backend: str          # none | pna_union | scale_aware | gatv2
    priority: str          # primary | secondary
    description: str


ABLATION_GRID = [
    # --- primary block: run these first ---
    # A1/A2 must load features from separate caches (feature_cache.load_cached_features(expected_encoder_name=...)) and train separate patch heads, or A1 silently reuses A2's features and becomes byte-identical to it (a real bug this project hit once; see feature_cache.py).
    AblationSpec("A1_imagenet_patch", "imagenet", "none", "none", "primary",
                 "ImageNet ConvNeXt, no distillation, patch-only"),
    AblationSpec("A2_distilled_patch", "distilled", "none", "none", "primary",
                 "GenBio-distilled ConvNeXt, patch-only (=y_patch, the anchor)"),
    AblationSpec("A5_shuffled_graph", "distilled", "shuffled", "pna_union", "primary",
                 "degree-preserving rewire -- CRITICAL negative control (Gate 3)"),
    AblationSpec("A6_local_pna", "distilled", "local", "pna_union", "primary",
                 "real LOCAL-only physical graph, union-PNA backend"),
    AblationSpec("A7_multiscale_pna", "distilled", "multiscale", "pna_union", "primary",
                 "real local+meso+macro union graph, one PNA sees all edges"),
    AblationSpec("A8_scale_aware_pna", "distilled", "multiscale", "scale_aware", "primary",
                 "PROPOSED FINAL MODEL: three per-scale PNA branches + learned softmax gate"),
    # --- secondary block ---
    AblationSpec("A0_mean_expression", "none", "none", "none", "secondary",
                 "predict training gene means -- sanity floor"),
    AblationSpec("A3_matched_mlp", "distilled", "none", "none", "secondary",
                 "parameter-matched MLP, same features, no neighbors"),
    AblationSpec("A4_random_graph", "distilled", "random", "pna_union", "secondary",
                 "same nodes/edge count, uniform random edges, union-PNA backend"),
    AblationSpec("A9_gatv2_multiscale", "distilled", "multiscale", "gatv2", "secondary",
                 "architecture control: same union multiscale graph, GATv2 instead of PNA"),
]

PRIMARY_ABLATIONS = [a for a in ABLATION_GRID if a.priority == "primary"]
SECONDARY_ABLATIONS = [a for a in ABLATION_GRID if a.priority == "secondary"]


def build_graph_for_ablation(
    coords_um: np.ndarray, graph_source: str, cfg: dict, seed: int
) -> dict:
    g = cfg["graph"]
    if graph_source == "multiscale":
        edges = graph_builder.build_multiscale_edges(
            coords_um,
            g["local_radius_um"],
            g["meso_radius_um"],
            g["macro_radius_um"],
            g["max_local_neighbors"],
            g["max_meso_neighbors"],
            g["max_macro_neighbors"],
        )
    elif graph_source == "local":
        edges = graph_builder.build_local_only_edges(
            coords_um, g["local_radius_um"], g["max_local_neighbors"]
        )
    elif graph_source in ("random", "shuffled"):
        base = graph_builder.build_multiscale_edges(
            coords_um,
            g["local_radius_um"],
            g["meso_radius_um"],
            g["macro_radius_um"],
            g["max_local_neighbors"],
            g["max_meso_neighbors"],
            g["max_macro_neighbors"],
        )
        n = coords_um.shape[0]
        if graph_source == "random":
            edge_index = destruction.random_graph(base["edge_index"], n, seed)
        else:
            edge_index = destruction.degree_preserving_rewire(base["edge_index"], seed)
        # random/shuffled edges have no meaningful physical distance -> zero-fill edge features
        edges = {
            "edge_index": edge_index,
            "distance_um": np.zeros(edge_index.shape[1], dtype=np.float32),
            "scale": np.array(["local"] * edge_index.shape[1], dtype=object),
        }
    elif graph_source == "none":
        edges = {
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "distance_um": np.zeros(0, dtype=np.float32),
            "scale": np.zeros(0, dtype=object),
        }
    else:
        raise ValueError(f"unknown graph_source: {graph_source}")
    return edges


def edge_features_for_ablation(edges: dict, graph_source: str, cfg: dict) -> np.ndarray:
    """20-D union edge feature (distance + RBF + scale one-hot) for the pna_union/gatv2 backends; A8 (scale-aware) doesn't call this -- it uses scale_aware_edge_data instead."""
    g = cfg["graph"]
    if graph_source in ("multiscale", "local", "random", "shuffled"):
        return graph_builder.union_edge_features(
            edges["distance_um"], edges["scale"], g["macro_radius_um"], g["n_rbf"], g["rbf_gamma"]
        )
    return np.zeros((0, g["n_rbf"] + 4), dtype=np.float32)  # 1 distance + 16 RBF + 3 onehot = 20


SCALE_TO_INT = {"local": 0, "meso": 1, "macro": 2}


def scale_aware_edge_data(edges: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A8 (scale-aware) helper: returns (edge_index, edge_attr, edge_scale) -- union edge_index, 17-D per-scale features (no one-hot, edge_scale carries that), and an int array (0=local,1=meso,2=macro) that ScaleAwareGraphCorrectionModel.forward uses to route each edge to its branch."""
    g = cfg["graph"]
    edge_attr = graph_builder.per_scale_edge_features(
        edges["distance_um"], g["macro_radius_um"], g["n_rbf"], g["rbf_gamma"]
    )
    edge_scale = np.array([SCALE_TO_INT[s] for s in edges["scale"]], dtype=np.int64)
    return edges["edge_index"], edge_attr, edge_scale


def matched_mlp_hidden_dim(pna_param_count: int, in_dim: int, out_dim: int) -> int:
    """A3: picks an MLP hidden width so its parameter count roughly matches the real PNA model, solving in_dim*h + h + h*out_dim + out_dim ~= pna_param_count for h."""
    h = max(8, int(round((pna_param_count - out_dim) / max(1, in_dim + out_dim + 1))))
    return h
