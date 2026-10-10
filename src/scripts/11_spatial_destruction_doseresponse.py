#!/usr/bin/env python
"""Spatial destruction dose-response of the graph correction model.

Stage 11 of the ePathST pipeline: replaces q in {0,25,50,75,100}% of the real edges with random
edges (same node and edge count), retrains the graph correction model for each q on the frozen
patch head, and reports the patient-level R(q) to test dR/dq < 0.

Input is a .npz for one fold and seed, produced alongside 10_run_patient_cv.py. Fold, seed and
patch head are fixed across q, so only the graph changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.graph.destruction import partial_edge_replacement
from epathst.graph.builder import build_multiscale_edges, rbf_edge_features
from epathst.models.graph_model import GraphCorrectionModel, ResidualGraphPredictor
from epathst.graph.training import make_pyg_data, predict_graph_model, train_graph_model
from epathst.evaluation.metrics import evaluate_predictions
from epathst.models.graph_layers import compute_degree_histogram
from epathst.utils import RunEnvironment, load_config, resolve_run_paths, save_json


def build_graphs_at_q(x, coords, patient_of, gene_t, path_t, q, cfg, seed):
    """Build one PyG graph per patient with a fraction q of its edges replaced by random edges.

    Real edges come from the multiscale builder (local/meso/macro radii in um); replacement keeps
    node and edge counts fixed. ``patient_of`` maps each row of ``x`` and ``coords`` to a patient.
    """
    g = cfg["graph"]
    graphs = []
    for p in np.unique(patient_of):
        mask = patient_of == p
        c = coords[mask]
        base = build_multiscale_edges(
            c,
            g["local_radius_um"],
            g["meso_radius_um"],
            g["macro_radius_um"],
            g["max_local_neighbors"],
            g["max_meso_neighbors"],
            g["max_macro_neighbors"],
        )
        replaced = partial_edge_replacement(
            base["edge_index"], mask.sum(), q, seed, base["distance_um"]
        )
        # Scale labels are not carried through partial_edge_replacement; every edge is "local".
        scale = np.array(["local"] * replaced["edge_index"].shape[1], dtype=object)
        edge_attr = rbf_edge_features(
            replaced["distance_um"], scale, g["macro_radius_um"], g["n_rbf"], g["rbf_gamma"]
        )
        graphs.append(
            make_pyg_data(x[mask], replaced["edge_index"], edge_attr, gene_t[mask], path_t[mask], p)
        )
    return graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--fold-artifacts",
        required=True,
        help="path to a .npz produced alongside script 10 containing "
        "x/coords/patient_of/targets for one fold+seed",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    # Fold artifacts for one fold and seed; the patch head is frozen and shared across all q.
    data = np.load(args.fold_artifacts, allow_pickle=True)
    x_train, coords_train, pat_train = data["x_train"], data["coords_train"], data["pat_train"]
    x_test, coords_test, pat_test = data["x_test"], data["coords_test"], data["pat_test"]
    y_gene_train, y_path_train = data["y_gene_train"], data["y_path_train"]
    y_gene_test, y_path_test = data["y_gene_test"], data["y_path_test"]
    patch_head_state = data["patch_head_state"].item()
    seed = int(data["seed"])

    from epathst.models.patch_head import build_patch_head

    patch_head = build_patch_head(
        cfg, x_train.shape[1], y_gene_train.shape[1], y_path_train.shape[1]
    ).to(args.device)
    patch_head.load_state_dict(patch_head_state)

    q_values = cfg["destruction_dose_response"]["q_values"]
    r_per_q = {}
    with RunEnvironment(
        "11_spatial_destruction_doseresponse", paths["logs"] / "11_dose_response_env.json"
    ):
        for q in q_values:
            graphs_train = build_graphs_at_q(
                x_train, coords_train, pat_train, y_gene_train, y_path_train, q, cfg, seed
            )
            graphs_test = build_graphs_at_q(
                x_test, coords_test, pat_test, y_gene_test, y_path_test, q, cfg, seed
            )

            ckpt = paths["student_checkpoints"] / f"dose_q{int(q*100)}_seed{seed}.pt"
            train_graph_model(patch_head, graphs_train, graphs_train, cfg, ckpt, seed, args.device)

            deg = compute_degree_histogram(
                [g.edge_index for g in graphs_train], [g.num_nodes for g in graphs_train]
            )
            edge_dim = graphs_train[0].edge_attr.shape[1]
            gc = GraphCorrectionModel(
                x_train.shape[1], cfg, deg, edge_dim, y_gene_train.shape[1], y_path_train.shape[1]
            )
            gc.load_state_dict(
                torch.load(ckpt, map_location=args.device)["graph_correction_state_dict"]
            )
            model = ResidualGraphPredictor(patch_head, gc).to(args.device)
            preds = predict_graph_model(model, graphs_test, args.device)

            met = evaluate_predictions(
                y_gene_test,
                preds["gene_pred"],
                preds["patient_id"],
                np.array(["pooled"] * len(preds["patient_id"])),
                [str(i) for i in range(y_gene_test.shape[1])],
                cfg,
            )
            # R(q): patient-level Fisher-z mean R of the gene predictions on the test patients.
            r_per_q[q] = met["overall"]["fisher_z_mean_patient_R"]
            print(f"[11_dose_response] q={q}: patient-level Fisher-z mean R = {r_per_q[q]:.4f}")

    save_json(r_per_q, paths["results"] / "spatial_destruction_dose_response.json")


if __name__ == "__main__":
    main()
