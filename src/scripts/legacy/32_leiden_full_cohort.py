#!/usr/bin/env python
"""Leiden/ARI check of ePathST-E (A2) and ePathST (A8) predictions on the full HEST-1k cohort.
Keeps the locked 5-fold patient CV: every sample is scored once, by the fold that held it out.

Uses cached student-encoder embeddings and no WSI pixels; graphs come from spot coordinates only
(see epathst/graph/builder.py). The A2 patch head is loaded from its checkpoint. The A8 graph model
was not persisted per fold, so it is retrained from the cached embeddings.

Leiden/ARI recipe matches the one-off TENX149 check: real-scale expression -> scale -> PCA (<=30
components) -> kNN(15) -> Leiden (igraph, 2 iterations). The resolution is searched on the
ground truth to give 5-9 clusters and reused for predictions; ARI is adjusted_rand_score.

Earlier variant kept for provenance; official-panel Leiden is now
`src/scripts/25_leiden_official_panel.py official`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from epathst.ablations import ABLATION_GRID
from epathst.data.expression import TrainOnlyStandardizer, select_hvg_for_fold
from epathst.data.feature_cache import load_cached_features
from epathst.data.splits import carve_validation_from_train, exclude_targeted_panel_samples
from epathst.evaluation.ablation_runner import load_fold_expression, run_ablation
from epathst.models.patch_head import build_patch_head
from epathst.utils import RunEnvironment, load_config, resolve_run_paths, save_json, seed_everything

MODELS = {"A2_distilled_patch": "ePathST-E", "A8_scale_aware_pna": "ePathST"}
ABLATION_BY_NAME = {a.name: a for a in ABLATION_GRID}


def leiden_cluster(expr_matrix: np.ndarray, resolution: float, seed: int = 0) -> np.ndarray:
    adata = ad.AnnData(X=expr_matrix.astype(np.float32))
    sc.pp.scale(adata, max_value=10)
    adata.X = np.nan_to_num(adata.X)
    n_comps = min(30, expr_matrix.shape[1] - 1, expr_matrix.shape[0] - 1)
    sc.pp.pca(adata, n_comps=max(2, n_comps), random_state=seed)
    sc.pp.neighbors(adata, n_neighbors=15, random_state=seed)
    sc.tl.leiden(adata, resolution=resolution, random_state=seed, flavor="igraph", n_iterations=2)
    return adata.obs["leiden"].astype(int).to_numpy()


def gt_clustering(gt_matrix: np.ndarray, min_clusters: int = 5, max_clusters: int = 9):
    labels, res, k = None, None, None
    for res in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]:
        labels = leiden_cluster(gt_matrix, res)
        k = len(set(labels))
        if min_clusters <= k <= max_clusters:
            return labels, res, k
    return labels, res, k


def reorder_like_graph_output(ids_test, y_gene_test, y_path_test, spot_by_uid):
    """Reorder targets as `run_ablation` does for graph ablations (grouped by patient, in order
    of first appearance). Must match exactly or ground-truth and prediction rows desync."""
    patient_of = spot_by_uid.loc[ids_test, "patient_id"].values
    order = np.concatenate([np.where(patient_of == p)[0] for p in np.unique(patient_of)])
    return ids_test[order], y_gene_test[order], y_path_test[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cancers", nargs="+", default=None, help="debug: restrict to these cancers")
    ap.add_argument(
        "--max-folds-per-cancer", type=int, default=None, help="debug: first N folds only"
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    device = args.device if torch.cuda.is_available() else "cpu"

    out_dir = Path(cfg["paths"]["persist_dir"]) / "leiden_clustering"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    hest_samples = pd.read_csv(paths["manifests"] / "hest_samples.csv")
    hest_samples = exclude_targeted_panel_samples(hest_samples)
    spot_manifest = pd.read_csv(paths["manifests"] / "hest_spot_manifest.csv")
    spot_by_uid = spot_manifest.set_index("spot_uid")
    cv_grid = load_json_cv_grid(paths["manifests"] / "hest_cv_folds.json")
    if args.cancers:
        cv_grid["per_cancer"] = {
            k: v for k, v in cv_grid["per_cancer"].items() if k in args.cancers
        }
    if args.max_folds_per_cancer:
        for info in cv_grid["per_cancer"].values():
            info["folds"] = info["folds"][: args.max_folds_per_cancer]

    feat_dirs = {src: paths["student_embeddings"] / src for src in ("distilled", "imagenet")}
    mt = cfg["molecular_targets"]

    per_sample_rows = []
    with RunEnvironment("32_leiden_full_cohort", out_dir / "logs" / "run_env.json"):
        for cancer, cv in cv_grid["per_cancer"].items():
            for fold in cv["folds"]:
                fold_idx = fold["fold"]
                seed = args.seed
                seed_everything(seed)
                print(
                    f"[32_leiden_full_cohort] {cancer} fold{fold_idx}: building targets + "
                    "features..."
                )

                train_p, val_p = carve_validation_from_train(fold["train_patients"], 0.15, seed)
                test_p = fold["test_patients"]

                train_expr, train_progeny, genes_by_sample, net = load_fold_expression(
                    hest_samples, spot_manifest, Path(cfg["paths"]["hest_root"]), train_p, cfg
                )
                common_genes = (
                    sorted(set.intersection(*genes_by_sample.values())) if genes_by_sample else []
                )
                hvg = select_hvg_for_fold(train_expr, common_genes, mt["hvg1000"], mt["hvg50"])

                gene_std = TrainOnlyStandardizer().fit(
                    train_expr[hvg["hvg1000"]].values, ids=train_expr.index
                )
                path_std = TrainOnlyStandardizer().fit(
                    train_progeny.values, ids=train_progeny.index
                )

                def build_targets(patients):
                    expr, progeny, _, _ = load_fold_expression(
                        hest_samples, spot_manifest, Path(cfg["paths"]["hest_root"]), patients, cfg
                    )
                    gene_t = gene_std.transform(
                        expr[hvg["hvg1000"]].reindex(columns=hvg["hvg1000"]).values
                    )
                    path_t = path_std.transform(
                        progeny.reindex(columns=net["source"].unique()).values
                    )
                    return gene_t, path_t, expr.index.astype(str)

                y_gene_train, y_path_train, ids_train = build_targets(train_p)
                y_gene_val, y_path_val, ids_val = build_targets(val_p)
                y_gene_test, y_path_test, ids_test = build_targets(test_p)
                ids_train, ids_val, ids_test = (
                    np.array(ids_train),
                    np.array(ids_val),
                    np.array(ids_test),
                )

                def load_feats(spot_uids, source):
                    cache_by_sample, out = {}, []
                    for uid in spot_uids:
                        sample_id, bare_spot_id = spot_by_uid.loc[uid, ["sample_id", "spot_id"]]
                        if sample_id not in cache_by_sample:
                            cache_by_sample[sample_id] = load_cached_features(
                                feat_dirs[source] / f"{sample_id}.h5", expected_encoder_name=source
                            )
                        cache = cache_by_sample[sample_id]
                        pos = list(cache["spot_id"]).index(bare_spot_id)
                        out.append(cache["embedding"][pos])
                    return np.stack(out)

                x_train = load_feats(ids_train, "distilled")
                x_val = load_feats(ids_val, "distilled")
                x_test = load_feats(ids_test, "distilled")

                # A2 patch head is reused from its saved checkpoint; only the A8 GNN is retrained.
                ckpt_path = (
                    paths["student_checkpoints"]
                    / "patch_head"
                    / cancer
                    / f"fold{fold_idx}"
                    / f"seed{seed}.pt"
                )
                patch_head = build_patch_head(
                    cfg, x_train.shape[1], y_gene_train.shape[1], y_path_train.shape[1]
                ).to(device)
                patch_head.load_state_dict(torch.load(ckpt_path, map_location=device))

                preds_by_model = {}
                for model_name in MODELS:
                    spec = ABLATION_BY_NAME[model_name]
                    print(
                        f"[32_leiden_full_cohort] {cancer} fold{fold_idx}: running {model_name} "
                        f"({'frozen patch head' if spec.graph_source == 'none' else 'training GNN'})..."
                    )
                    preds, _metrics, eval_ids_test = run_ablation(
                        spec,
                        x_train,
                        x_val,
                        x_test,
                        ids_train,
                        ids_val,
                        ids_test,
                        y_gene_train,
                        y_path_train,
                        y_gene_val,
                        y_path_val,
                        y_gene_test,
                        y_path_test,
                        patch_head,
                        spot_manifest,
                        cfg,
                        seed,
                        device,
                        n_hvg50=len(hvg["hvg50"]),
                        pathway_names=list(net["source"].unique()),
                    )
                    preds_by_model[model_name] = {
                        "gene_pred": preds["gene"],
                        "eval_ids_test": eval_ids_test,
                    }

                # Ground truth follows the graph-ablation row order (A8 reorders internally).
                gt_ids, gt_gene, _gt_path = reorder_like_graph_output(
                    ids_test, y_gene_test, y_path_test, spot_by_uid
                )
                gt_real = (
                    gt_gene * (gene_std.sigma_ + gene_std.eps) + gene_std.mu_
                )  # back to real expression scale

                for sample_id in hest_samples[hest_samples["patient_id"].isin(test_p)]["sample_id"]:
                    sample_uid_mask = np.array([u.startswith(f"{sample_id}::") for u in gt_ids])
                    if (
                        sample_uid_mask.sum() < 20
                    ):  # too few spots for stable Leiden; skipped, not a failure
                        continue
                    sample_uids = gt_ids[sample_uid_mask]
                    gt_matrix = gt_real[sample_uid_mask]
                    gt_labels, gt_res, gt_k = gt_clustering(gt_matrix)
                    if gt_labels is None or not (5 <= gt_k <= 9):
                        print(
                            f"[32_leiden_full_cohort] {sample_id}: GT clustering never hit 5-9 "
                            "clusters "
                            f"(got {gt_k} at res={gt_res}) -- keeping anyway, flagged in output"
                        )

                    row = {
                        "cancer": cancer,
                        "fold": fold_idx,
                        "sample_id": sample_id,
                        "n_spots": int(sample_uid_mask.sum()),
                        "gt_resolution": gt_res,
                        "gt_n_clusters": gt_k,
                    }
                    for model_name, disp in MODELS.items():
                        eval_ids_n = preds_by_model[model_name]["eval_ids_test"]
                        uid_to_row = {u: i for i, u in enumerate(eval_ids_n)}
                        pred_rows = [uid_to_row[u] for u in sample_uids]
                        pred_real = (
                            preds_by_model[model_name]["gene_pred"][pred_rows]
                            * (gene_std.sigma_ + gene_std.eps)
                            + gene_std.mu_
                        )
                        pred_labels = leiden_cluster(pred_real, gt_res)
                        ari = float(adjusted_rand_score(gt_labels, pred_labels))
                        row[f"ari_{disp}"] = ari
                    per_sample_rows.append(row)
                    ari_str = " ".join(f"{d}={row['ari_' + d]:.4f}" for d in MODELS.values())
                    print(f"[32_leiden_full_cohort] {sample_id} ({cancer}): {ari_str}")

                # written after every fold so a partial run keeps its results
                pd.DataFrame(per_sample_rows).to_csv(out_dir / "per_sample_ari.csv", index=False)

    per_sample = pd.DataFrame(per_sample_rows)
    per_sample.to_csv(out_dir / "per_sample_ari.csv", index=False)

    summary_rows = []
    for cancer, grp in per_sample.groupby("cancer"):
        row = {"cancer": cancer, "n_samples": len(grp)}
        for disp in MODELS.values():
            row[f"ari_{disp}_mean"] = float(grp[f"ari_{disp}"].mean())
            row[f"ari_{disp}_std"] = float(grp[f"ari_{disp}"].std())
        summary_rows.append(row)
    overall = {"cancer": "ALL", "n_samples": len(per_sample)}
    for disp in MODELS.values():
        overall[f"ari_{disp}_mean"] = float(per_sample[f"ari_{disp}"].mean())
        overall[f"ari_{disp}_std"] = float(per_sample[f"ari_{disp}"].std())
    summary_rows.append(overall)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "ari_summary_by_cancer.csv", index=False)
    save_json(summary.to_dict("records"), out_dir / "ari_summary_by_cancer.json")
    print(summary.to_string(index=False))
    print(
        f"[32_leiden_full_cohort] wrote {out_dir / 'per_sample_ari.csv'} and "
        f"{out_dir / 'ari_summary_by_cancer.csv'}"
    )


def load_json_cv_grid(path: Path) -> dict:
    import json

    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    main()
