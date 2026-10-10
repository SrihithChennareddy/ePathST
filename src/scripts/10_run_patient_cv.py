#!/usr/bin/env python
"""Patient-level cross-validation of every downstream ablation (A0-A9) on HEST.

Stage 10 of the ePathST pipeline: for each (cancer, outer fold, seed) builds HVG and pathway
targets from the training patients only, trains the patch head and runs every spec in
ABLATION_GRID. A final gate asserts that every expected (cancer, fold, seed, ablation) run
completed before results are aggregated.

Leakage controls: folds are grouped by patient, validation patients are carved out of the training
fold, and HVG selection and target standardizers are fit on training patients only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.ablations import ABLATION_GRID
from epathst.data.expression import TrainOnlyStandardizer, select_hvg_for_fold
from epathst.data.feature_cache import load_cached_features
from epathst.data.splits import (
    assert_run_completeness,
    build_cv_grid,
    carve_validation_from_train,
    exclude_targeted_panel_samples,
    expected_run_keys,
)
from epathst.evaluation.ablation_runner import load_fold_expression, run_ablation, train_patch_head
from epathst.utils import RunEnvironment, load_config, resolve_run_paths, save_json, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--feature-seed",
        type=int,
        default=0,
        help="which locked-encoder seed's cached features to use",
    )
    ap.add_argument(
        "--max-folds-per-cancer",
        type=int,
        default=None,
        help="run only the first N of build_cv_grid's folds per cancer (default: all "
        "n_outer_folds). "
        "With the default n_outer_folds=5 (an 80/20 GroupKFold split per fold), "
        "--max-folds-per-cancer 1 runs a single 80/20 patient-level split instead of the full "
        "5-fold design -- for time-budgeted runs; not the primary design.",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="override cfg.cross_validation.seeds (default: all configured seeds)",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    hest_root = Path(cfg["paths"]["hest_root"])
    if args.seeds is not None:
        cfg["cross_validation"]["seeds"] = args.seeds

    hest_samples = pd.read_csv(paths["manifests"] / "hest_samples.csv")
    hest_samples = exclude_targeted_panel_samples(hest_samples)
    spot_manifest = pd.read_csv(paths["manifests"] / "hest_spot_manifest.csv")
    # Both encoder caches are used; load_feats() passes expected_encoder_name to the loader so
    # A1 (imagenet) can never silently receive distilled features.
    FEATURE_SOURCES = ("distilled", "imagenet")
    feat_dirs = {src: paths["student_embeddings"] / src for src in FEATURE_SOURCES}

    patients_by_cancer = {
        c: hest_samples[hest_samples["cancer"] == c]["patient_id"].unique().tolist()
        for c in cfg["cancers"]["hest"]
    }
    cv_grid = build_cv_grid(patients_by_cancer, cfg)
    if args.max_folds_per_cancer is not None:
        for cancer, info in cv_grid["per_cancer"].items():
            info["folds"] = info["folds"][: args.max_folds_per_cancer]
        cv_grid["pooled"]["folds"] = cv_grid["pooled"]["folds"][: args.max_folds_per_cancer]
        print(
            f"[10_run_patient_cv] --max-folds-per-cancer {args.max_folds_per_cancer}: trimmed "
            "cv_grid "
            f"(expected_run_keys below must be computed from this trimmed grid, not the full "
            "design)"
        )
    save_json(cv_grid, paths["manifests"] / "hest_cv_folds.json")

    mt = cfg["molecular_targets"]
    all_predictions = []
    all_metrics = []
    completed_keys = set()  # filled as runs finish; checked by the completeness gate at the end

    for cancer, cv in cv_grid["per_cancer"].items():
        for fold in cv["folds"]:
            for seed in cfg["cross_validation"]["seeds"]:
                stage = f"10_cv_{cancer}_fold{fold['fold']}_seed{seed}"
                with RunEnvironment(stage, paths["logs"] / f"{stage}_env.json"):
                    seed_everything(seed)
                    # Validation patients are taken from the training fold, never from test.
                    train_p, val_p = carve_validation_from_train(fold["train_patients"], 0.15, seed)
                    test_p = fold["test_patients"]

                    # Leakage guard: HVGs and standardizers use training patients only.
                    train_expr, train_progeny, genes_by_sample, net = load_fold_expression(
                        hest_samples, spot_manifest, hest_root, train_p, cfg
                    )
                    common_genes = (
                        sorted(set.intersection(*genes_by_sample.values()))
                        if genes_by_sample
                        else []
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
                            hest_samples, spot_manifest, hest_root, patients, cfg
                        )
                        # Val/test targets reuse the train-fitted standardizer; assert none of
                        # their spots were in its fit.
                        if patients != train_p:
                            gene_std.assert_no_leakage(expr.index)
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

                    spot_by_uid = spot_manifest.set_index("spot_uid")
                    feat_cache_by_source = {src: {} for src in FEATURE_SOURCES}

                    def load_feats(spot_uids, source):
                        cache_by_sample = feat_cache_by_source[source]
                        out = []
                        for uid in spot_uids:
                            sample_id, bare_spot_id = spot_by_uid.loc[uid, ["sample_id", "spot_id"]]
                            if sample_id not in cache_by_sample:
                                cache_by_sample[sample_id] = load_cached_features(
                                    feat_dirs[source] / f"{sample_id}.h5",
                                    expected_encoder_name=source,
                                )
                            cache = cache_by_sample[sample_id]
                            pos = list(cache["spot_id"]).index(bare_spot_id)
                            out.append(cache["embedding"][pos])
                        return np.stack(out)

                    # A1/A2 each get their own features and their own patch head, trained
                    # identically otherwise (see train_patch_head).
                    x_by_source = {
                        src: {
                            "train": load_feats(ids_train, src),
                            "val": load_feats(ids_val, src),
                            "test": load_feats(ids_test, src),
                        }
                        for src in FEATURE_SOURCES
                    }
                    patch_heads = {
                        src: train_patch_head(
                            x_by_source[src]["train"],
                            x_by_source[src]["val"],
                            y_gene_train,
                            y_path_train,
                            y_gene_val,
                            y_path_val,
                            cfg,
                            args.device,
                        )
                        for src in FEATURE_SOURCES
                    }

                    ckpt_dir = (
                        paths["student_checkpoints"] / "patch_head" / cancer / f"fold{fold['fold']}"
                    )
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    for src, ph in patch_heads.items():
                        suffix = (
                            "" if src == "distilled" else f"_{src}"
                        )  # keep the historical "distilled" filename
                        torch.save(ph.state_dict(), ckpt_dir / f"seed{seed}{suffix}.pt")

                    # ABLATION_GRID is ordered primary block first (A1/A2/A5/A6/A7/A8), then
                    # secondary (A0/A3/A4/A9).
                    for spec in ABLATION_GRID:
                        # Specs without their own feature source use the distilled features.
                        src = (
                            spec.feature_source
                            if spec.feature_source in x_by_source
                            else "distilled"
                        )
                        spec_x_train = x_by_source[src]["train"]
                        spec_x_val = x_by_source[src]["val"]
                        spec_x_test = x_by_source[src]["test"]
                        spec_patch_head = patch_heads[src]
                        preds, r_metrics, eval_ids_test = run_ablation(
                            spec,
                            spec_x_train,
                            spec_x_val,
                            spec_x_test,
                            ids_train,
                            ids_val,
                            ids_test,
                            y_gene_train,
                            y_path_train,
                            y_gene_val,
                            y_path_val,
                            y_gene_test,
                            y_path_test,
                            spec_patch_head,
                            spot_manifest,
                            cfg,
                            seed,
                            args.device,
                            n_hvg50=len(hvg["hvg50"]),
                            pathway_names=list(net["source"].unique()),
                        )
                        # eval_ids_test is ids_test in the order preds were scored in: identity
                        # for A0-A3, grouped by patient for graph ablations. Zipping the original
                        # ids_test here would reintroduce the spot/prediction misalignment bug.
                        for pid, g_pred, p_pred in zip(
                            eval_ids_test, preds["gene"], preds["pathway"]
                        ):
                            all_predictions.append(
                                {
                                    "cancer": cancer,
                                    "fold": fold["fold"],
                                    "seed": seed,
                                    "ablation": spec.name,
                                    "spot_uid": pid,
                                    "patient_id": spot_by_uid.loc[pid, "patient_id"],
                                }
                            )
                        all_metrics.append(
                            {
                                "cancer": cancer,
                                "fold": fold["fold"],
                                "seed": seed,
                                "ablation": spec.name,
                                **r_metrics,
                            }
                        )
                        completed_keys.add((cancer, fold["fold"], seed, spec.name))

    pd.DataFrame(all_metrics).to_csv(paths["results"] / "cv_ablation_metrics.csv", index=False)
    print(f"[10_run_patient_cv] wrote {paths['results'] / 'cv_ablation_metrics.csv'}")

    # Hard completeness gate before aggregation: prevents silently pooling only 3 of 4 completed
    # folds (see docs/reports/POC5_RESULTS.md).
    expected_keys = expected_run_keys(cv_grid, [a.name for a in ABLATION_GRID], unit="per_cancer")
    assert_run_completeness(expected_keys, completed_keys, context="10_run_patient_cv final gate")
    print(
        "[10_run_patient_cv] completeness gate PASSED -- every expected (cancer, fold, seed, "
        "ablation) run present"
    )


if __name__ == "__main__":
    main()
