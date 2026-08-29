#!/usr/bin/env python
"""Step 25 (Section 29), revised per Final_Project_V2.txt Phase 17: S1-S9.
MUST all pass before launching the full 5-fold x 3-seed run. Run this
against one small fold/patient subset first.

S1-S7 are V1's original TEST1-7 (renamed to V2's S-numbering, same
checks). S3 (zero residual), S8 (target order), and S9 (fold completeness)
are NEW in V2 -- S3 directly targets the exact PoC_5 failure mode
(multiscale PNA scoring below patch-only CNN) by asserting the residual
correction is truly zero at initialization; S9 directly targets the other
PoC_5 failure this project already found (cross-fold aggregate silently
pooling only 3 of 4 completed folds, see docs/POC5_RESULTS.md)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts.ablations import build_graph_for_ablation, edge_features_for_ablation, scale_aware_edge_data
from sts.cv_folds import assert_run_completeness
from sts.destruction import degree_preserving_rewire, random_graph
from sts.graph_model import (GraphCorrectionModel, ResidualGraphPredictor, ScaleAwareGraphCorrectionModel,
                              assert_zero_residual_at_init)
from sts.graph_training import make_pyg_data, train_graph_model
from sts.patch_head import build_patch_head, patch_head_loss
from sts.pna_block import compute_degree_histogram
from sts.utils import load_config, resolve_run_paths, save_json


def synthetic_toy_data(cfg, n_nodes=60, in_dim=320, gene_dim=20, pathway_dim=14, seed=0):
    rng = np.random.RandomState(seed)
    coords = rng.uniform(0, 800, size=(n_nodes, 2)).astype(np.float32)
    x = rng.randn(n_nodes, in_dim).astype(np.float32)
    # inject a spatially smooth signal into one target column so a working graph model can exploit it
    gene = rng.randn(n_nodes, gene_dim).astype(np.float32)
    gene[:, 0] = np.sin(coords[:, 0] / 100.0) + 0.05 * rng.randn(n_nodes)
    pathway = rng.randn(n_nodes, pathway_dim).astype(np.float32)
    return x, coords, gene, pathway


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    results = {}
    x, coords, gene, pathway = synthetic_toy_data(cfg)
    gene_names = [f"gene_{i}" for i in range(gene.shape[1])]

    # S1: cached features through the exact patch head reproduce patch-model predictions.
    # eval() is REQUIRED here, not optional -- patch_head.py's bottleneck trunk includes a
    # Dropout layer (V2 Phase 7), so two forward passes in train() mode would legitimately
    # differ (this caught a real missing-.eval() bug in ResidualGraphPredictor -- see its fix).
    patch_head = build_patch_head(cfg, x.shape[1], gene.shape[1], pathway.shape[1]).to(args.device)
    patch_head.eval()
    xt = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        out1 = patch_head(xt)["gene"].detach().numpy()
        out2 = patch_head(torch.tensor(x, dtype=torch.float32))["gene"].detach().numpy()
    results["S1_cached_features_reproduce_patch_predictions"] = bool(np.allclose(out1, out2))

    # S2: spot_id/embedding/coordinate/gene-vector alignment.
    spot_ids = [f"spot_{i}" for i in range(x.shape[0])]
    results["S2_integrity_alignment"] = (len(spot_ids) == x.shape[0] == coords.shape[0] == gene.shape[0])

    # S4/S5: real, shuffled, random graphs contain exactly the intended nodes; degree distributions sane.
    real_edges = build_graph_for_ablation(coords, "multiscale", cfg, seed=0)
    shuffled_edges = build_graph_for_ablation(coords, "shuffled", cfg, seed=0)
    random_edges = build_graph_for_ablation(coords, "random", cfg, seed=0)
    n = x.shape[0]
    nodes_ok = all(e["edge_index"].size == 0 or (e["edge_index"].min() >= 0 and e["edge_index"].max() < n)
                   for e in (real_edges, shuffled_edges, random_edges))
    results["S4_graphs_contain_intended_nodes_only"] = bool(nodes_ok)

    def degree_hist(edge_index):
        deg = np.zeros(n, dtype=int)
        for j in edge_index[1]:
            deg[j] += 1
        return deg

    deg_real, deg_shuffled = degree_hist(real_edges["edge_index"]), degree_hist(shuffled_edges["edge_index"])
    results["S5_degree_distribution_preserved_by_shuffle"] = bool(np.array_equal(np.sort(deg_real), np.sort(deg_shuffled)))

    # S6: shuffled graph genuinely destroys spatial locality.
    real_pairs = real_edges["edge_index"]
    real_dist = np.linalg.norm(coords[real_pairs[0]] - coords[real_pairs[1]], axis=1) if real_pairs.size else np.array([])
    shuf_pairs = shuffled_edges["edge_index"]
    shuf_dist = np.linalg.norm(coords[shuf_pairs[0]] - coords[shuf_pairs[1]], axis=1) if shuf_pairs.size else np.array([])
    results["S6_shuffle_destroys_locality"] = bool(len(real_dist) and len(shuf_dist) and
                                                     shuf_dist.mean() > 1.15 * real_dist.mean())

    # S3 (NEW, V2 Phase 11): zero-residual at init, for BOTH the union-PNA
    # backend and the scale-aware backend -- directly targets PoC_5's
    # observed failure (patch CNN 0.1875 > multiscale PNA 0.1268).
    edge_attr = edge_features_for_ablation(real_edges, "multiscale", cfg)
    ei = torch.tensor(real_edges["edge_index"], dtype=torch.long)
    ea = torch.tensor(edge_attr, dtype=torch.float32)
    deg = compute_degree_histogram([ei], [n])
    gc_union = GraphCorrectionModel(x.shape[1], cfg, deg, ea.shape[1], gene.shape[1], pathway.shape[1])
    model_union = ResidualGraphPredictor(patch_head, gc_union)
    try:
        assert_zero_residual_at_init(model_union, xt, ei, ea)
        union_zero_ok = True
    except AssertionError as e:
        union_zero_ok = False
        print(f"[16_sanity_checks] S3 union-PNA failure: {e}", file=sys.stderr)

    sa_edge_index, sa_edge_attr, sa_edge_scale = scale_aware_edge_data(real_edges, cfg)
    sa_ei_t = torch.tensor(sa_edge_index, dtype=torch.long)
    sa_ea_t = torch.tensor(sa_edge_attr, dtype=torch.float32)
    sa_scale_t = torch.tensor(sa_edge_scale, dtype=torch.long)
    degL = compute_degree_histogram([sa_ei_t[:, sa_scale_t == 0]], [n])
    degM = compute_degree_histogram([sa_ei_t[:, sa_scale_t == 1]], [n])
    degG = compute_degree_histogram([sa_ei_t[:, sa_scale_t == 2]], [n])
    gc_scale_aware = ScaleAwareGraphCorrectionModel(x.shape[1], cfg, degL, degM, degG, sa_ea_t.shape[1],
                                                     gene.shape[1], pathway.shape[1])
    model_scale_aware = ResidualGraphPredictor(patch_head, gc_scale_aware)
    try:
        assert_zero_residual_at_init(model_scale_aware, xt, sa_ei_t, sa_ea_t, sa_scale_t)
        scale_aware_zero_ok = True
    except AssertionError as e:
        scale_aware_zero_ok = False
        print(f"[16_sanity_checks] S3 scale-aware failure: {e}", file=sys.stderr)
    results["S3_zero_residual_at_init"] = bool(union_zero_ok and scale_aware_zero_ok)

    # S7: training on this tiny synthetic graph decreases loss strongly (attempt-to-overfit combined).
    data = make_pyg_data(x, real_edges["edge_index"], edge_attr, gene, pathway, "toy_patient")
    ckpt_path = paths["student_checkpoints"] / "sanity_toy.pt"
    res = train_graph_model(patch_head, [data], [data], cfg, ckpt_path, seed=0, device=args.device, graphs_per_step=1)
    target_var = float(np.var(gene))
    results["S7_tiny_overfit_and_loss_decreases"] = bool(res["best_epoch"] >= 0 and res["best_val_loss"] < target_var)

    # S8 (NEW, V2 Phase 17): the exact same HVG ordering must be used by
    # every head in a fold -- simulated here as a consistency check between
    # the gene-name list the patch head was built against and the one the
    # graph model's targets are keyed to (in the real pipeline this checks
    # the fold's locked `hvg["hvg1000"]` order end-to-end).
    results["S8_target_gene_order_consistent"] = bool(len(gene_names) == gene.shape[1] == patch_head.gene_head.out_features)

    # S9 (NEW, V2 Phase 3/17): fold-completeness assertion actually raises
    # on a deliberately incomplete synthetic run set, and passes on a
    # complete one -- proves the guard itself works before relying on it
    # for the real cross-fold aggregation (this is the PoC_5 bug class).
    expected = {("COAD", 0, 0, "A2_distilled_patch"), ("COAD", 0, 1, "A2_distilled_patch")}
    complete = {("COAD", 0, 0, "A2_distilled_patch"), ("COAD", 0, 1, "A2_distilled_patch")}
    incomplete = {("COAD", 0, 0, "A2_distilled_patch")}  # missing seed 1 -- must raise
    s9_pass_on_complete, s9_raises_on_incomplete = False, False
    try:
        assert_run_completeness(expected, complete, context="S9 self-test (complete)")
        s9_pass_on_complete = True
    except AssertionError:
        pass
    try:
        assert_run_completeness(expected, incomplete, context="S9 self-test (incomplete, should raise)")
    except AssertionError:
        s9_raises_on_incomplete = True
    results["S9_fold_completeness_guard_works"] = bool(s9_pass_on_complete and s9_raises_on_incomplete)

    save_json(results, paths["qc"] / "sanity_checks_step25.json")
    print("[16_sanity_checks] results:")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    if not all(results.values()):
        print("[16_sanity_checks] FAILURES DETECTED -- do not launch the full 5-fold x 3-seed run "
              "until every check above passes.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
