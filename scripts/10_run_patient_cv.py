#!/usr/bin/env python
"""Steps 11, 14, 15, 17, 19 (Sections 14, 17-19, 21, 23), revised per
Final_Project_V2.txt -- the main downstream experiment loop.

For every cancer's patient-level CV folds x every seed:
  1. select HVG1000/HVG50 from TRAIN patients only, compute PROGENy14 on
     the full normalized expression, standardize using train-fold stats
     (Step 11 / Section 14, leakage-free);
  2. train the patch-only molecular bottleneck head (Step 14) -- this
     defines y_patch,i, gene-only loss by default (V2 Phase 7);
  3. build the physical multiscale graph per section (Step 15);
  4. for every ablation in sts.ablations.ABLATION_GRID (PRIMARY block
     first -- A1/A2/A5/A6/A7/A8 -- then SECONDARY -- A0/A3/A4/A9 -- per V2
     Phase 14/18's run-order), train/evaluate with the SAME folds,
     features, targets, and procedure (Step 17), saving predictions for
     every test spot (Section 21). A8 (scale-aware PNA) and A9 (GATv2)
     are new backends (sts.graph_model/graph_training); A5 (shuffled) is
     the critical negative control.
  5. before returning, assert every expected (cancer, fold, seed,
     ablation) run actually completed (V2 Phase 3 -- the exact PoC_5
     aggregation-completeness bug this project already hit once, see
     docs/POC5_RESULTS.md).

The GenBio-teacher PCA256 fair comparison (V2 Phase 21) and the
destruction dose-response (Step 18) are separate scripts that reuse the
exact fold/target artifacts written here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts.ablations import (ABLATION_GRID, build_graph_for_ablation, edge_features_for_ablation,
                            matched_mlp_hidden_dim, scale_aware_edge_data)
from sts.cv_folds import assert_run_completeness, build_cv_grid, carve_validation_from_train, expected_run_keys
from sts.expression import TrainOnlyStandardizer, normalize_log1p_slide, select_hvg_for_fold
from sts.feature_cache import load_cached_features
from sts.graph_model import ResidualGraphPredictor
from sts.graph_training import _build_correction_model, make_pyg_data, predict_graph_model, train_graph_model
from sts.metrics import evaluate_predictions
from sts.pathways import check_pathway_gene_coverage, compute_progeny_scores, fetch_and_cache_progeny_net
from sts.patch_head import MolecularBottleneckHead, build_patch_head, patch_head_loss
from sts.pna_block import compute_degree_histogram
from sts.utils import RunEnvironment, load_config, resolve_run_paths, save_json, seed_everything


def load_fold_expression(hest_samples, spot_manifest, hest_root, patients, cfg):
    """Loads + per-slide normalizes+log1p's every sample belonging to
    `patients`, returns (expr_df[spot x common_genes], progeny_df, spot_meta).

    Indexed by `spot_uid` (`{sample_id}::{spot_id}`), NOT bare `spot_id` --
    HEST spot ids are Visium array-grid coordinates (e.g. "010x016"), not
    global barcodes, and collide heavily across different samples/slides
    (confirmed live, 2026-08-29: two real breast samples shared 144/176
    spot_ids). Since a fold's train/val/test patient sets always span
    multiple samples, `pd.concat`-ing frames indexed by bare spot_id would
    silently produce a duplicate-keyed index and every downstream `.loc[]`
    lookup (coords_for/patient_for/load_feats below) would then be
    ambiguous across patients -- see scripts/08_prepare_hest_metadata.py's
    `spot_uid` column, the source of truth for this join key."""
    frames, progeny_frames, genes_by_sample = [], [], {}
    net = fetch_and_cache_progeny_net(Path(cfg["paths"]["work_dir"]) / "progeny_net.csv",
                                       cfg["molecular_targets"]["pathway_top_genes"])
    for sample_id in hest_samples[hest_samples["patient_id"].isin(patients)]["sample_id"]:
        row = hest_samples[hest_samples["sample_id"] == sample_id].iloc[0]
        adata = ad.read_h5ad(row["expression_path"])
        normalize_log1p_slide(adata, cfg["molecular_targets"]["normalize_total_target_sum"])
        uid_index = [f"{sample_id}::{sid}" for sid in adata.obs_names.astype(str)]
        expr = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
                             index=uid_index, columns=adata.var_names.astype(str))
        genes_by_sample[sample_id] = set(expr.columns)
        frames.append(expr)
        progeny_frames.append(compute_progeny_scores(expr, net))
    expr_all = pd.concat(frames)
    progeny_all = pd.concat(progeny_frames)
    assert expr_all.index.is_unique, "spot_uid collision within one fold's pooled expression -- should be impossible"
    return expr_all, progeny_all, genes_by_sample, net


def build_graph_data(feat_dict, coords, spot_ids, gene_target, pathway_target, patient_id, graph_source, backend,
                      cfg, seed):
    edges = build_graph_for_ablation(coords, graph_source, cfg, seed)
    if backend == "scale_aware":
        edge_index, edge_attr, edge_scale = scale_aware_edge_data(edges, cfg)
        return make_pyg_data(feat_dict, edge_index, edge_attr, gene_target, pathway_target, patient_id,
                              edge_scale=edge_scale)
    edge_attr = edge_features_for_ablation(edges, graph_source, cfg)
    return make_pyg_data(feat_dict, edges["edge_index"], edge_attr, gene_target, pathway_target, patient_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--feature-seed", type=int, default=0, help="which locked-encoder seed's cached features to use")
    ap.add_argument("--max-folds-per-cancer", type=int, default=None,
                     help="run only the first N of build_cv_grid's folds per cancer (default: all n_outer_folds). "
                          "With the default n_outer_folds=5 (an 80/20 GroupKFold split per fold), "
                          "--max-folds-per-cancer 1 runs a single real 80/20 patient-level split instead of "
                          "the full V2 Phase 3 5-fold design -- for time-budgeted runs; NOT the primary design.")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                     help="override cfg.cross_validation.seeds (default: all configured seeds)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    hest_root = Path(cfg["paths"]["hest_root"])
    if args.seeds is not None:
        cfg["cross_validation"]["seeds"] = args.seeds

    hest_samples = pd.read_csv(paths["manifests"] / "hest_samples.csv")
    spot_manifest = pd.read_csv(paths["manifests"] / "hest_spot_manifest.csv")
    feat_dir = paths["student_embeddings"] / "distilled"

    patients_by_cancer = {c: hest_samples[hest_samples["cancer"] == c]["patient_id"].unique().tolist()
                           for c in cfg["cancers"]["hest"]}
    cv_grid = build_cv_grid(patients_by_cancer, cfg)
    if args.max_folds_per_cancer is not None:
        for cancer, info in cv_grid["per_cancer"].items():
            info["folds"] = info["folds"][:args.max_folds_per_cancer]
        cv_grid["pooled"]["folds"] = cv_grid["pooled"]["folds"][:args.max_folds_per_cancer]
        print(f"[10_run_patient_cv] --max-folds-per-cancer {args.max_folds_per_cancer}: trimmed cv_grid "
              f"(this run's expected_run_keys below must be computed from the TRIMMED grid, not the full V2 design)")
    save_json(cv_grid, paths["manifests"] / "hest_cv_folds.json")

    mt = cfg["molecular_targets"]
    all_predictions = []
    all_metrics = []
    completed_keys = set()  # V2 Phase 3 completeness gate -- populated as ablations finish, checked at the end

    for cancer, cv in cv_grid["per_cancer"].items():
        for fold in cv["folds"]:
            for seed in cfg["cross_validation"]["seeds"]:
                stage = f"10_cv_{cancer}_fold{fold['fold']}_seed{seed}"
                with RunEnvironment(stage, paths["logs"] / f"{stage}_env.json"):
                    seed_everything(seed)
                    train_p, val_p = carve_validation_from_train(fold["train_patients"], 0.15, seed)
                    test_p = fold["test_patients"]

                    train_expr, train_progeny, genes_by_sample, net = load_fold_expression(
                        hest_samples, spot_manifest, hest_root, train_p, cfg)
                    common_genes = sorted(set.intersection(*genes_by_sample.values())) if genes_by_sample else []
                    hvg = select_hvg_for_fold(train_expr, common_genes, mt["hvg1000"], mt["hvg50"])

                    gene_std = TrainOnlyStandardizer().fit(train_expr[hvg["hvg1000"]].values, ids=train_expr.index)
                    path_std = TrainOnlyStandardizer().fit(train_progeny.values, ids=train_progeny.index)

                    def build_targets(patients):
                        expr, progeny, _, _ = load_fold_expression(hest_samples, spot_manifest, hest_root, patients, cfg)
                        gene_std.assert_no_leakage(expr.index) if patients != train_p else None
                        gene_t = gene_std.transform(expr[hvg["hvg1000"]].reindex(columns=hvg["hvg1000"]).values)
                        path_t = path_std.transform(progeny.reindex(columns=net["source"].unique()).values)
                        return gene_t, path_t, expr.index.astype(str)

                    y_gene_train, y_path_train, ids_train = build_targets(train_p)
                    y_gene_val, y_path_val, ids_val = build_targets(val_p)
                    y_gene_test, y_path_test, ids_test = build_targets(test_p)

                    spot_by_uid = spot_manifest.set_index("spot_uid")
                    feat_cache_by_sample = {}

                    def load_feats(spot_uids):
                        out = []
                        for uid in spot_uids:
                            sample_id, bare_spot_id = spot_by_uid.loc[uid, ["sample_id", "spot_id"]]
                            if sample_id not in feat_cache_by_sample:
                                feat_cache_by_sample[sample_id] = load_cached_features(feat_dir / f"{sample_id}.h5")
                            cache = feat_cache_by_sample[sample_id]
                            pos = list(cache["spot_id"]).index(bare_spot_id)
                            out.append(cache["embedding"][pos])
                        return np.stack(out)

                    x_train, x_val, x_test = load_feats(ids_train), load_feats(ids_val), load_feats(ids_test)

                    patch_head = build_patch_head(cfg, x_train.shape[1], mt["hvg1000"], 14).to(args.device)
                    opt = torch.optim.AdamW(patch_head.parameters(), lr=cfg["patch_head"]["lr"],
                                             weight_decay=cfg["patch_head"]["weight_decay"])
                    best_val, patience_ctr, best_state = float("inf"), 0, None
                    for epoch in range(cfg["patch_head"]["max_epochs"]):
                        patch_head.train()
                        xb = torch.tensor(x_train, dtype=torch.float32, device=args.device)
                        gb = torch.tensor(y_gene_train, dtype=torch.float32, device=args.device)
                        pb = torch.tensor(y_path_train, dtype=torch.float32, device=args.device)
                        opt.zero_grad()
                        out = patch_head(xb)
                        loss = patch_head_loss(out, gb, pb, cfg["patch_head"]["pathway_loss_weight"])
                        loss["loss"].backward()
                        opt.step()

                        patch_head.eval()
                        with torch.no_grad():
                            xv = torch.tensor(x_val, dtype=torch.float32, device=args.device)
                            gv = torch.tensor(y_gene_val, dtype=torch.float32, device=args.device)
                            pv = torch.tensor(y_path_val, dtype=torch.float32, device=args.device)
                            val_loss = patch_head_loss(patch_head(xv), gv, pv, cfg["patch_head"]["pathway_loss_weight"])["loss"].item()
                        if val_loss < best_val:
                            best_val, patience_ctr, best_state = val_loss, 0, {k: v.clone() for k, v in patch_head.state_dict().items()}
                        else:
                            patience_ctr += 1
                        if patience_ctr >= cfg["patch_head"]["early_stop_patience"]:
                            break
                    patch_head.load_state_dict(best_state)

                    ckpt_dir = paths["student_checkpoints"] / "patch_head" / cancer / f"fold{fold['fold']}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(patch_head.state_dict(), ckpt_dir / f"seed{seed}.pt")

                    # V2 Phase 14/18: ABLATION_GRID is already ordered primary-block-first
                    # (A1/A2/A5/A6/A7/A8), secondary-block second (A0/A3/A4/A9).
                    for spec in ABLATION_GRID:
                        preds, r_metrics = run_ablation(spec, x_train, x_val, x_test, ids_train, ids_val, ids_test,
                                                         y_gene_train, y_path_train, y_gene_val, y_path_val,
                                                         y_gene_test, y_path_test, patch_head, spot_manifest, cfg,
                                                         seed, args.device)
                        for pid, g_pred, p_pred, g_true, p_true in zip(ids_test, preds["gene"], preds["pathway"],
                                                                        y_gene_test, y_path_test):
                            all_predictions.append({"cancer": cancer, "fold": fold["fold"], "seed": seed,
                                                     "ablation": spec.name, "spot_uid": pid,
                                                     "patient_id": spot_by_uid.loc[pid, "patient_id"]})
                        all_metrics.append({"cancer": cancer, "fold": fold["fold"], "seed": seed,
                                             "ablation": spec.name, **r_metrics})
                        completed_keys.add((cancer, fold["fold"], seed, spec.name))

    pd.DataFrame(all_metrics).to_csv(paths["results"] / "cv_ablation_metrics.csv", index=False)
    print(f"[10_run_patient_cv] wrote {paths['results'] / 'cv_ablation_metrics.csv'}")

    # V2 Phase 3 / S9: hard completeness gate before this run is trusted for
    # any downstream aggregation -- see docs/POC5_RESULTS.md for the exact
    # failure this prevents (a cross-fold aggregate silently pooling only
    # 3 of 4 completed folds).
    expected_keys = expected_run_keys(cv_grid, [a.name for a in ABLATION_GRID], unit="per_cancer")
    assert_run_completeness(expected_keys, completed_keys, context="10_run_patient_cv final gate")
    print("[10_run_patient_cv] completeness gate PASSED -- every expected (cancer, fold, seed, ablation) run present")


def run_ablation(spec, x_train, x_val, x_test, ids_train, ids_val, ids_test, y_gene_train, y_path_train,
                  y_gene_val, y_path_val, y_gene_test, y_path_test, patch_head, spot_manifest, cfg, seed, device):
    """Trains/evaluates one ablation from sts.ablations.ABLATION_GRID and
    returns (test predictions, headline metrics). Dispatches on
    (spec.graph_source, spec.backend):
      A0 (graph_source='none', feature_source='none'): training-mean baseline, no model at all.
      A1/A2 (graph_source='none'): frozen patch_head forward pass -- y_patch,i itself.
      A3 (graph_source='none', but NOT reusing patch_head): parameter-matched MLP, no neighbors.
      A4/A5/A6/A7 (backend='pna_union'), A9 (backend='gatv2'): residual graph correction, union edge features.
      A8 (backend='scale_aware'): residual graph correction, per-scale edge features + edge_scale tags."""
    # Joined on `spot_uid` (`{sample_id}::{spot_id}`), never bare `spot_id` -- HEST spot ids are
    # Visium array-grid coordinates that collide across different samples, and `ids_train/val/test`
    # here always span multiple patients (see load_fold_expression's docstring / scripts/08's spot_uid).
    spot_by_uid = spot_manifest.set_index("spot_uid")

    def coords_for(spot_uids):
        return spot_by_uid.loc[spot_uids, ["x_um", "y_um"]].values

    def patient_for(spot_uids):
        return spot_by_uid.loc[spot_uids, "patient_id"].values

    if spec.name == "A0_mean_expression":
        gene_mean = y_gene_train.mean(axis=0, keepdims=True)
        path_mean = y_path_train.mean(axis=0, keepdims=True)
        gene_pred = np.repeat(gene_mean, len(ids_test), axis=0)
        path_pred = np.repeat(path_mean, len(ids_test), axis=0)
    elif spec.name == "A3_matched_mlp":
        # V2 Phase 14: parameter-matched MLP, same distilled features, no neighbors -- tests
        # whether extra capacity ALONE (without any graph structure) explains a would-be gain.
        pna_param_count = sum(p.numel() for p in patch_head.parameters())
        h = matched_mlp_hidden_dim(pna_param_count, x_train.shape[1], y_gene_train.shape[1] + y_path_train.shape[1])
        mlp = MolecularBottleneckHead(x_train.shape[1], [x_train.shape[1], h, h], y_gene_train.shape[1],
                                       y_path_train.shape[1]).to(device)
        opt = torch.optim.AdamW(mlp.parameters(), lr=cfg["patch_head"]["lr"], weight_decay=cfg["patch_head"]["weight_decay"])
        best_val, patience_ctr, best_state = float("inf"), 0, None
        for epoch in range(cfg["patch_head"]["max_epochs"]):
            mlp.train()
            xb = torch.tensor(x_train, dtype=torch.float32, device=device)
            gb = torch.tensor(y_gene_train, dtype=torch.float32, device=device)
            pb = torch.tensor(y_path_train, dtype=torch.float32, device=device)
            opt.zero_grad()
            loss = patch_head_loss(mlp(xb), gb, pb, cfg["patch_head"]["pathway_loss_weight"])
            loss["loss"].backward()
            opt.step()
            mlp.eval()
            with torch.no_grad():
                xv = torch.tensor(x_val, dtype=torch.float32, device=device)
                gv = torch.tensor(y_gene_val, dtype=torch.float32, device=device)
                pv = torch.tensor(y_path_val, dtype=torch.float32, device=device)
                val_loss = patch_head_loss(mlp(xv), gv, pv, cfg["patch_head"]["pathway_loss_weight"])["loss"].item()
            if val_loss < best_val:
                best_val, patience_ctr, best_state = val_loss, 0, {k: v.clone() for k, v in mlp.state_dict().items()}
            else:
                patience_ctr += 1
            if patience_ctr >= cfg["patch_head"]["early_stop_patience"]:
                break
        mlp.load_state_dict(best_state)
        mlp.eval()
        with torch.no_grad():
            out = mlp(torch.tensor(x_test, dtype=torch.float32, device=device))
        gene_pred, path_pred = out["gene"].cpu().numpy(), out["pathway"].cpu().numpy()
    elif spec.graph_source == "none":
        # A1/A2: frozen patch head forward pass -- this IS y_patch,i for A2 (the anchor).
        patch_head.eval()
        with torch.no_grad():
            out = patch_head(torch.tensor(x_test, dtype=torch.float32, device=device))
        gene_pred, path_pred = out["gene"].cpu().numpy(), out["pathway"].cpu().numpy()
    else:
        backend = spec.backend
        graphs_train = [build_graph_data(x_train[patient_for(ids_train) == p], coords_for(ids_train[patient_for(ids_train) == p]),
                                          ids_train[patient_for(ids_train) == p], y_gene_train[patient_for(ids_train) == p],
                                          y_path_train[patient_for(ids_train) == p], p, spec.graph_source, backend, cfg, seed)
                         for p in np.unique(patient_for(ids_train))]
        graphs_val = [build_graph_data(x_val[patient_for(ids_val) == p], coords_for(ids_val[patient_for(ids_val) == p]),
                                        ids_val[patient_for(ids_val) == p], y_gene_val[patient_for(ids_val) == p],
                                        y_path_val[patient_for(ids_val) == p], p, spec.graph_source, backend, cfg, seed)
                       for p in np.unique(patient_for(ids_val))]
        graphs_test = [build_graph_data(x_test[patient_for(ids_test) == p], coords_for(ids_test[patient_for(ids_test) == p]),
                                         ids_test[patient_for(ids_test) == p], y_gene_test[patient_for(ids_test) == p],
                                         y_path_test[patient_for(ids_test) == p], p, spec.graph_source, backend, cfg, seed)
                        for p in np.unique(patient_for(ids_test))]

        ckpt_path = Path(cfg["paths"]["work_dir"]) / "student_checkpoints" / f"graph_{spec.name}_seed{seed}.pt"
        train_graph_model(patch_head, graphs_train, graphs_val, cfg, ckpt_path, seed, device, backend=backend)
        edge_dim = graphs_train[0].edge_attr.shape[1]
        graph_corr = _build_correction_model(backend, x_train.shape[1], cfg, graphs_train, edge_dim,
                                              y_gene_train.shape[1], y_path_train.shape[1])
        state = torch.load(ckpt_path, map_location=device)
        graph_corr.load_state_dict(state["graph_correction_state_dict"])
        model = ResidualGraphPredictor(patch_head, graph_corr).to(device)
        preds = predict_graph_model(model, graphs_test, device, backend=backend)
        gene_pred, path_pred = preds["gene_pred"], preds["pathway_pred"]
        # NOTE: predict_graph_model concatenates node predictions in the same
        # per-patient grouping order used to build graphs_test above, so
        # `ids_test` must be reordered identically before being used to index
        # spot-level metadata downstream (e.g. via a per-patient stable sort).
        ids_test = np.concatenate([ids_test[patient_for(ids_test) == p] for p in np.unique(patient_for(ids_test))])

    from sts.metrics import evaluate_predictions as _eval
    metrics = _eval(y_gene_test, gene_pred, patient_for(ids_test), np.array(["cancer"] * len(ids_test)),
                     [str(i) for i in range(y_gene_test.shape[1])], cfg)
    return {"gene": gene_pred, "pathway": path_pred}, metrics.get("overall", {})


if __name__ == "__main__":
    main()
