"""Step 17 (Section 21), revised per Final_Project_V2.txt Phase 14: the
required ablation grid A0-A9. Each ablation is expressed as a
(feature_source, graph_source) pair so the CV driver can loop over them
with exactly the same patient folds, training procedure, and evaluation
code -- only the inputs/backend differ.

    A0  mean-expression baseline           (no model)
    A1  ImageNet ConvNeXt patch-only       features=imagenet,  graph=none
    A2  GenBio-distilled ConvNeXt patch    features=distilled, graph=none   <- y_patch,i
    A3  parameter-matched MLP              features=distilled, graph=none, wider MLP (no neighbors)
    A4  random graph residual PNA          features=distilled, graph=random,   backend=pna_union
    A5  degree-shuffled residual PNA       features=distilled, graph=shuffled, backend=pna_union   <- CRITICAL negative control
    A6  real LOCAL residual PNA            features=distilled, graph=local,    backend=pna_union
    A7  real UNION MULTISCALE residual PNA features=distilled, graph=multiscale, backend=pna_union
    A8  SCALE-AWARE multiscale residual PNA (PROPOSED FINAL MODEL)
                                            features=distilled, graph=multiscale, backend=scale_aware
    A9  multiscale residual GATv2 (architecture control, not the main model)
                                            features=distilled, graph=multiscale, backend=gatv2

V2 Phase 14/18 run order: primary models FIRST (A1, A2, A5, A6, A7, A8),
secondary/lower-priority controls after (A3, A4, A9) -- so the
scientifically essential results land before GPU time is spent on
lower-priority controls. `ABLATION_GRID`'s list order already reflects
this; drivers should iterate it in order and may stop after the primary
block under time pressure.

The GenBio-teacher-vs-student "fair comparison" (V1's old A8/A9 slots) is
NO LONGER part of this grid -- V2 Phase 21 replaces it with a dedicated,
methodologically stricter PCA-256-based comparison (train-fold-only PCA,
not raw 4608-D GenBio fed to a mismatched head), implemented separately
in sts/genbio_fair_comparison.py and run only after the student model
matrix above is locked."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sts import destruction, graph_builder


@dataclass
class AblationSpec:
    name: str
    feature_source: str    # imagenet | distilled
    graph_source: str      # none | random | shuffled | local | multiscale
    backend: str            # none | pna_union | scale_aware | gatv2
    priority: str            # primary | secondary
    description: str


ABLATION_GRID = [
    # --- primary block (V2 Phase 14/18: run these first) ---
    # A1_imagenet_patch RESTORED 2026-08-30 (final-experiment diagnostic): the 2026-08-30
    # removal (commit 96005f2) was because scripts/10_run_patient_cv.py's main() only ever
    # loaded the DISTILLED feature cache and reused ONE patch_head for every
    # graph_source="none" ablation, so A1's declared "imagenet" encoder was never actually
    # consulted (silently byte-identical to A2). Fixed properly this time: script 10 now
    # loads BOTH the "imagenet" and "distilled" caches (each asserted via
    # sts.feature_cache.load_cached_features(expected_encoder_name=...) to only ever load
    # its own encoder's features) and trains a SEPARATE patch_head on each, so A1 and A2
    # now differ only in which frozen encoder produced their input features -- everything
    # else (patient split, HVG1000/HVG50/PROGENy14 targets, head architecture, optimizer,
    # epochs/early stopping, evaluation) is identical between them.
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
    # --- secondary block (run after primary block passes) ---
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
    """Always the 20-D UNION feature (V2 Phase 9) for the pna_union/gatv2
    backends -- the scale-aware backend (A8) never calls this; it uses
    `edge_features_per_scale` on each of the three split edge sets
    instead (17-D each, see sts.graph_builder.per_scale_edge_features)."""
    g = cfg["graph"]
    if graph_source in ("multiscale", "local", "random", "shuffled"):
        return graph_builder.union_edge_features(edges["distance_um"], edges["scale"], g["macro_radius_um"],
                                                  g["n_rbf"], g["rbf_gamma"])
    return np.zeros((0, g["n_rbf"] + 4), dtype=np.float32)  # 1 distance + 16 RBF + 3 onehot = 20


SCALE_TO_INT = {"local": 0, "meso": 1, "macro": 2}


def scale_aware_edge_data(edges: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A8 (scale-aware) helper: returns (edge_index, edge_attr, edge_scale)
    -- the UNION edge_index (same ordering as `edges`), 17-D
    per_scale_edge_features (no one-hot -- edge_scale carries that
    information instead), and an int array tagging each edge's scale
    (0=local, 1=meso, 2=macro). Feeds directly into
    sts.graph_training.make_pyg_data's optional `edge_scale` argument and
    sts.graph_model.ScaleAwareGraphCorrectionModel.forward, which masks by
    `edge_scale` to route each edge to its branch -- no parallel 3-edge-set
    data pipeline needed."""
    g = cfg["graph"]
    edge_attr = graph_builder.per_scale_edge_features(edges["distance_um"], g["macro_radius_um"], g["n_rbf"], g["rbf_gamma"])
    edge_scale = np.array([SCALE_TO_INT[s] for s in edges["scale"]], dtype=np.int64)
    return edges["edge_index"], edge_attr, edge_scale


def matched_mlp_hidden_dim(pna_param_count: int, in_dim: int, out_dim: int) -> int:
    """A3: pick an MLP hidden width so its parameter count is approximately
    matched to the real PNA model (Section 21: 'parameter-matched MLP'),
    solving in_dim*h + h + h*out_dim + out_dim ~= pna_param_count for h."""
    h = max(8, int(round((pna_param_count - out_dim) / max(1, in_dim + out_dim + 1))))
    return h
