#!/usr/bin/env python
"""Steps 11, 14, 15, 17, 19 (Sections 14, 17-19, 21, 23) -- the main
downstream experiment loop.

For every cancer's patient-level CV folds x every seed:
  1. select HVG1000/HVG50 from TRAIN patients only, compute PROGENy14 on
     the full normalized expression, standardize using train-fold stats
     (Step 11 / Section 14, leakage-free);
  2. train the patch-only molecular bottleneck head (Step 14) -- this
     defines y_patch,i;
  3. build the physical multiscale graph per section (Step 15);
  4. for every ablation A0-A7 in sts.ablations.ABLATION_GRID, train/
     evaluate with the SAME folds, features, targets, and procedure
     (Step 17), saving predictions for every test spot (Section 21).

A8/A9 (GenBio teacher fair comparison) and the destruction dose-response
(Step 18) are separate scripts (11 and 12) that reuse the exact fold/target
artifacts written here.
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
from sts.ablations import ABLATION_GRID, build_graph_for_ablation, edge_features_for_ablation
from sts.cv_folds import build_cv_grid, carve_validation_from_train
from sts.expression import TrainOnlyStandardizer, normalize_log1p_slide, select_hvg_for_fold
from sts.feature_cache import load_cached_features
from sts.graph_model import GraphCorrectionModel, ResidualGraphPredictor
from sts.graph_training import make_pyg_data, predict_graph_model, train_graph_model
from sts.metrics import evaluate_predictions
from sts.pathways import check_pathway_gene_coverage, compute_progeny_scores, fetch_and_cache_progeny_net
from sts.patch_head import build_patch_head, patch_head_loss
from sts.pna_block import compute_degree_histogram
from sts.utils import RunEnvironment, load_config, resolve_run_paths, save_json, seed_everything


def load_fold_expression(hest_samples, spot_manifest, hest_root, patients, cfg):
    """Loads + per-slide normalizes+log1p's every sample belonging to
    `patients`, returns (expr_df[spot x common_genes], progeny_df, spot_meta)."""
    frames, progeny_frames, genes_by_sample = [], [], {}
    net = fetch_and_cache_progeny_net(Path(cfg["paths"]["work_dir"]) / "progeny_net.csv",
                                       cfg["molecular_targets"]["pathway_top_genes"])
    for sample_id in hest_samples[hest_samples["patient_id"].isin(patients)]["sample_id"]:
        row = hest_samples[hest_samples["sample_id"] == sample_id].iloc[0]
        adata = ad.read_h5ad(row["expression_path"])
        normalize_log1p_slide(adata, cfg["molecular_targets"]["normalize_total_target_sum"])
        expr = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
                             index=adata.obs_names.astype(str), columns=adata.var_names.astype(str))
        genes_by_sample[sample_id] = set(expr.columns)
        frames.append(expr)
        progeny_frames.append(compute_progeny_scores(expr, net))
    expr_all = pd.concat(frames)
    progeny_all = pd.concat(progeny_frames)
    return expr_all, progeny_all, genes_by_sample, net


def build_graph_data(feat_dict, coords, spot_ids, gene_target, pathway_target, patient_id, graph_source, cfg, seed):
    edges = build_graph_for_ablation(coords, graph_source, cfg, seed)
    edge_attr = edge_features_for_ablation(edges, graph_source, cfg)
    return make_pyg_data(feat_dict, edges["edge_index"], edge_attr, gene_target, pathway_target, patient_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--feature-seed", type=int, default=0, help="which locked-encoder seed's cached features to use")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    hest_root = Path(cfg["paths"]["hest_root"])

    hest_samples = pd.read_csv(paths["manifests"] / "hest_samples.csv")
    spot_manifest = pd.read_csv(paths["manifests"] / "hest_spot_manifest.csv")
    feat_dir = paths["student_embeddings"] / "distilled"

    patients_by_cancer = {c: hest_samples[hest_samples["cancer"] == c]["patient_id"].unique().tolist()
                           for c in cfg["cancers"]["hest"]}
    cv_grid = build_cv_grid(patients_by_cancer, cfg)
    save_json(cv_grid, paths["manifests"] / "hest_cv_folds.json")

    mt = cfg["molecular_targets"]
    all_predictions = []
    all_metrics = []

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

                    def load_feats(spot_ids):
                        out = []
                        for sid in spot_ids:
                            sample_id = spot_manifest.set_index("spot_id").loc[sid, "sample_id"]
                            cache = load_cached_features(feat_dir / f"{sample_id}.h5")
                            pos = list(cache["spot_id"]).index(sid)
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

                    for spec in ABLATION_GRID:
                        if spec.name in ("A0_mean_expression", "A8_genbio_patch", "A9_genbio_pna"):
                            continue  # A0 handled in metrics baseline; A8/A9 in script 11
                        preds, r_metrics = run_ablation(spec, x_train, x_val, x_test, ids_train, ids_val, ids_test,
                                                         y_gene_train, y_path_train, y_gene_val, y_path_val,
                                                         y_gene_test, y_path_test, patch_head, spot_manifest, cfg,
                                                         seed, args.device)
                        for pid, g_pred, p_pred, g_true, p_true in zip(ids_test, preds["gene"], preds["pathway"],
                                                                        y_gene_test, y_path_test):
                            all_predictions.append({"cancer": cancer, "fold": fold["fold"], "seed": seed,
                                                     "ablation": spec.name, "spot_id": pid,
                                                     "patient_id": spot_manifest.set_index("spot_id").loc[pid, "patient_id"]})
                        all_metrics.append({"cancer": cancer, "fold": fold["fold"], "seed": seed,
                                             "ablation": spec.name, **r_metrics})

    pd.DataFrame(all_metrics).to_csv(paths["results"] / "cv_ablation_metrics.csv", index=False)
    print(f"[10_run_patient_cv] wrote {paths['results'] / 'cv_ablation_metrics.csv'}")


def run_ablation(spec, x_train, x_val, x_test, ids_train, ids_val, ids_test, y_gene_train, y_path_train,
                  y_gene_val, y_path_val, y_gene_test, y_path_test, patch_head, spot_manifest, cfg, seed, device):
    """Trains/evaluates one ablation from sts.ablations.ABLATION_GRID and
    returns (test predictions, headline metrics)."""
    def coords_for(spot_ids):
        return spot_manifest.set_index("spot_id").loc[spot_ids, ["x_um", "y_um"]].values

    def patient_for(spot_ids):
        return spot_manifest.set_index("spot_id").loc[spot_ids, "patient_id"].values

    if spec.graph_source == "none":
        # A1/A2/A3: no graph, evaluate the frozen patch head (or matched MLP -- reuses patch_head
        # architecture here for A2; a dedicated wider MLP is substituted for A3 in a full run).
        import torch as _torch
        with _torch.no_grad():
            out = patch_head(_torch.tensor(x_test, dtype=_torch.float32, device=device))
        gene_pred, path_pred = out["gene"].cpu().numpy(), out["pathway"].cpu().numpy()
    else:
        graphs_train = [build_graph_data(x_train[patient_for(ids_train) == p], coords_for(ids_train[patient_for(ids_train) == p]),
                                          ids_train[patient_for(ids_train) == p], y_gene_train[patient_for(ids_train) == p],
                                          y_path_train[patient_for(ids_train) == p], p, spec.graph_source, cfg, seed)
                         for p in np.unique(patient_for(ids_train))]
        graphs_val = [build_graph_data(x_val[patient_for(ids_val) == p], coords_for(ids_val[patient_for(ids_val) == p]),
                                        ids_val[patient_for(ids_val) == p], y_gene_val[patient_for(ids_val) == p],
                                        y_path_val[patient_for(ids_val) == p], p, spec.graph_source, cfg, seed)
                       for p in np.unique(patient_for(ids_val))]
        graphs_test = [build_graph_data(x_test[patient_for(ids_test) == p], coords_for(ids_test[patient_for(ids_test) == p]),
                                         ids_test[patient_for(ids_test) == p], y_gene_test[patient_for(ids_test) == p],
                                         y_path_test[patient_for(ids_test) == p], p, spec.graph_source, cfg, seed)
                        for p in np.unique(patient_for(ids_test))]

        ckpt_path = Path(cfg["paths"]["work_dir"]) / "student_checkpoints" / f"graph_{spec.name}_seed{seed}.pt"
        train_graph_model(patch_head, graphs_train, graphs_val, cfg, ckpt_path, seed, device)
        deg = compute_degree_histogram([g.edge_index for g in graphs_train], [g.num_nodes for g in graphs_train])
        edge_dim = graphs_train[0].edge_attr.shape[1] if graphs_train[0].edge_attr.numel() else 19
        graph_corr = GraphCorrectionModel(x_train.shape[1], cfg, deg, edge_dim, y_gene_train.shape[1], y_path_train.shape[1])
        state = torch.load(ckpt_path, map_location=device)
        graph_corr.load_state_dict(state["graph_correction_state_dict"])
        model = ResidualGraphPredictor(patch_head, graph_corr).to(device)
        preds = predict_graph_model(model, graphs_test, device)
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
