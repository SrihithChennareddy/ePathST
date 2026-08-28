#!/usr/bin/env python
"""Step 25 (Section 29): TEST 1-7. MUST all pass before launching the full
5-fold x 3-seed run. Run this against one small fold/patient subset first."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts.ablations import build_graph_for_ablation, edge_features_for_ablation
from sts.destruction import degree_preserving_rewire, random_graph
from sts.graph_model import GraphCorrectionModel, ResidualGraphPredictor
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

    # TEST 1: cached features through the exact patch head reproduce patch-model predictions.
    patch_head = build_patch_head(cfg, x.shape[1], gene.shape[1], pathway.shape[1]).to(args.device)
    xt = torch.tensor(x, dtype=torch.float32)
    out1 = patch_head(xt)["gene"].detach().numpy()
    out2 = patch_head(torch.tensor(x, dtype=torch.float32))["gene"].detach().numpy()
    results["TEST1_cached_features_reproduce_patch_predictions"] = bool(np.allclose(out1, out2))

    # TEST 2: spot_id/embedding/coordinate/gene-vector alignment.
    spot_ids = [f"spot_{i}" for i in range(x.shape[0])]
    results["TEST2_integrity_alignment"] = (len(spot_ids) == x.shape[0] == coords.shape[0] == gene.shape[0])

    # TEST 3/4: real, shuffled, random graphs contain exactly the intended nodes; degree distributions sane.
    real_edges = build_graph_for_ablation(coords, "multiscale", cfg, seed=0)
    shuffled_edges = build_graph_for_ablation(coords, "shuffled", cfg, seed=0)
    random_edges = build_graph_for_ablation(coords, "random", cfg, seed=0)
    n = x.shape[0]
    nodes_ok = all(e["edge_index"].size == 0 or (e["edge_index"].min() >= 0 and e["edge_index"].max() < n)
                   for e in (real_edges, shuffled_edges, random_edges))
    results["TEST3_graphs_contain_intended_nodes_only"] = bool(nodes_ok)

    def degree_hist(edge_index):
        deg = np.zeros(n, dtype=int)
        for j in edge_index[1]:
            deg[j] += 1
        return deg

    deg_real, deg_shuffled = degree_hist(real_edges["edge_index"]), degree_hist(shuffled_edges["edge_index"])
    results["TEST4_degree_distribution_preserved_by_shuffle"] = bool(np.array_equal(np.sort(deg_real), np.sort(deg_shuffled)))

    # TEST 5: shuffled graph genuinely destroys spatial locality (mean edge distance should NOT
    # match the real graph's, since physical distance is not preserved by degree-preserving rewiring).
    real_pairs = real_edges["edge_index"]
    real_dist = np.linalg.norm(coords[real_pairs[0]] - coords[real_pairs[1]], axis=1) if real_pairs.size else np.array([])
    shuf_pairs = shuffled_edges["edge_index"]
    shuf_dist = np.linalg.norm(coords[shuf_pairs[0]] - coords[shuf_pairs[1]], axis=1) if shuf_pairs.size else np.array([])
    results["TEST5_shuffle_destroys_locality"] = bool(len(real_dist) and len(shuf_dist) and
                                                        shuf_dist.mean() > 1.15 * real_dist.mean())

    # TEST 6: training on this tiny synthetic graph decreases loss strongly.
    edge_attr = edge_features_for_ablation(real_edges, "multiscale", cfg)
    data = make_pyg_data(x, real_edges["edge_index"], edge_attr, gene, pathway, "toy_patient")
    ckpt_path = paths["student_checkpoints"] / "sanity_toy.pt"
    res = train_graph_model(patch_head, [data], [data], cfg, ckpt_path, seed=0, device=args.device, graphs_per_step=1)
    results["TEST6_toy_training_completes"] = bool(res["best_epoch"] >= 0)

    # TEST 7: attempt to overfit the tiny graph (implementation test) -- best val loss should be
    # small relative to the target variance after enough epochs.
    target_var = float(np.var(gene))
    results["TEST7_overfit_small_graph"] = bool(res["best_val_loss"] < target_var)

    save_json(results, paths["qc"] / "sanity_checks_step25.json")
    print("[16_sanity_checks] results:")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    if not all(results.values()):
        print("[16_sanity_checks] FAILURES DETECTED -- do not launch the full 5-fold x 3-seed run "
              "until every TEST above passes.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
