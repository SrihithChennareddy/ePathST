#!/usr/bin/env python
"""Spatial destruction dose-response: starting from the real multiscale graph,
progressively replace q in {0,25,50,75,100}% of physical edges with random
edges (same node/edge count), retrain/evaluate the graph correction model
at each q, and report R(q) -- testing dR/dq < 0."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathology_st.destruction import partial_edge_replacement
from pathology_st.graph_builder import build_multiscale_edges, rbf_edge_features
from pathology_st.graph_model import GraphCorrectionModel, ResidualGraphPredictor
from pathology_st.graph_training import make_pyg_data, predict_graph_model, train_graph_model
from pathology_st.metrics import evaluate_predictions
from pathology_st.pna_block import compute_degree_histogram
from pathology_st.stats_tests import patient_bootstrap_ci
from pathology_st.utils import RunEnvironment, load_config, resolve_run_paths, save_json


def build_graphs_at_q(x, coords, patient_of, gene_t, path_t, q, cfg, seed):
    g = cfg["graph"]
    graphs = []
    for p in np.unique(patient_of):
        mask = patient_of == p
        c = coords[mask]
        base = build_multiscale_edges(c, g["local_radius_um"], g["meso_radius_um"], g["macro_radius_um"],
                                       g["max_local_neighbors"], g["max_meso_neighbors"], g["max_macro_neighbors"])
        replaced = partial_edge_replacement(base["edge_index"], mask.sum(), q, seed, base["distance_um"])
        scale = np.array(["local"] * replaced["edge_index"].shape[1], dtype=object)
        edge_attr = rbf_edge_features(replaced["distance_um"], scale, g["macro_radius_um"], g["n_rbf"], g["rbf_gamma"])
        graphs.append(make_pyg_data(x[mask], replaced["edge_index"], edge_attr, gene_t[mask], path_t[mask], p))
    return graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fold-artifacts", required=True,
                     help="path to a .npz produced alongside script 10 containing x/coords/patient_of/targets for one fold+seed")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    # load fold artifacts + frozen patch head
    data = np.load(args.fold_artifacts, allow_pickle=True)
    x_train, coords_train, pat_train = data["x_train"], data["coords_train"], data["pat_train"]
    x_test, coords_test, pat_test = data["x_test"], data["coords_test"], data["pat_test"]
    y_gene_train, y_path_train = data["y_gene_train"], data["y_path_train"]
    y_gene_test, y_path_test = data["y_gene_test"], data["y_path_test"]
    patch_head_state = data["patch_head_state"].item()
    seed = int(data["seed"])

    from pathology_st.patch_head import build_patch_head
    patch_head = build_patch_head(cfg, x_train.shape[1], y_gene_train.shape[1], y_path_train.shape[1]).to(args.device)
    patch_head.load_state_dict(patch_head_state)

    q_values = cfg["destruction_dose_response"]["q_values"]
    r_per_q = {}
    with RunEnvironment("11_spatial_destruction_doseresponse", paths["logs"] / "11_dose_response_env.json"):
        for q in q_values:
            # build graphs with q% of edges randomly replaced
            graphs_train = build_graphs_at_q(x_train, coords_train, pat_train, y_gene_train, y_path_train, q, cfg, seed)
            graphs_test = build_graphs_at_q(x_test, coords_test, pat_test, y_gene_test, y_path_test, q, cfg, seed)

            # train graph correction model
            ckpt = paths["student_checkpoints"] / f"dose_q{int(q*100)}_seed{seed}.pt"
            train_graph_model(patch_head, graphs_train, graphs_train, cfg, ckpt, seed, args.device)

            # run inference
            deg = compute_degree_histogram([g.edge_index for g in graphs_train], [g.num_nodes for g in graphs_train])
            edge_dim = graphs_train[0].edge_attr.shape[1]
            gc = GraphCorrectionModel(x_train.shape[1], cfg, deg, edge_dim, y_gene_train.shape[1], y_path_train.shape[1])
            gc.load_state_dict(torch.load(ckpt, map_location=args.device)["graph_correction_state_dict"])
            model = ResidualGraphPredictor(patch_head, gc).to(args.device)
            preds = predict_graph_model(model, graphs_test, args.device)

            # compute metrics
            met = evaluate_predictions(y_gene_test, preds["gene_pred"], preds["patient_id"],
                                        np.array(["pooled"] * len(preds["patient_id"])),
                                        [str(i) for i in range(y_gene_test.shape[1])], cfg)
            r_per_q[q] = met["overall"]["fisher_z_mean_patient_R"]
            print(f"[11_dose_response] q={q}: patient-level Fisher-z mean R = {r_per_q[q]:.4f}")

    save_json(r_per_q, paths["results"] / "spatial_destruction_dose_response.json")


if __name__ == "__main__":
    main()
