#!/usr/bin/env python
"""Runs the A0-A9 ablation grid (`epathst.ablations.ABLATION_GRID`) under the official HEST-1k
Benchmark protocol (splits, genes, log1p-only targets from MahmoodLab/hest-bench on Hugging Face).
Compares patch-only and GNN-corrected ablations on the official 50-gene per-cancer panel and
k-fold splits, unlike src/scripts/10_run_patient_cv.py's HVG1000/HVG50/PROGENy14 protocol with
project GroupKFold splits.

`src/scripts/19_official_hest_benchmark.py` reproduces the official protocol only for two raw
encoders with a PCA(256)+Ridge head and never touches the A0-A9 grid. This script reuses the same
trained-model code (`epathst.evaluation.ablation_runner`) so A0-A9 run as in the custom pipeline
with official data, targets and splits swapped in.

Deliberate adaptations of the official protocol:
  - target: the official <cancer>/var_50genes.json panel, `sc.pp.log1p` only (no normalize_total),
    as in script 19's `official_target`. With a single 50-gene panel there is no HVG1000/HVG50
    distinction: `n_hvg50` is len(genes), so the "hvg50" slice from `run_ablation` aliases the
    full-panel "hvg1000" metrics and both columns are identical by design.
  - pathway (PROGENy14) target: not evaluated. Some official-split samples carry a targeted panel
    (a few hundred genes), so PROGENy scoring is unreliable. A single all-NaN pathway column is
    passed so the shared patch_head/graph_loss code runs unmodified: both mask by `~isnan(target)`,
    so it adds zero loss regardless of `pathway_loss_weight`, and `evaluate_predictions` reports
    NaN for it (as for A0's undefined Pearson r). pathway_* output columns are dropped.
  - patient unit for graph building and evaluation: one official `sample_id`. The official splits
    are already held out at sample/patient level, so `patient_id` is set to `sample_id`.
  - spot coordinates (um): official patches ship only pixel coordinates (`coords` in
    `patches/<sample>.h5`), converted with the 10x Visium convention
    `mpp = 55.0 / spot_diameter_fullres` (`adata.uns['spatial'][<tech>]['scalefactors']`). This
    matched array-grid spacing on a real IDC sample (mpp ~= 0.2126 um/px). The value is asserted
    to lie in [0.05, 2.0] um/px so an unvalidated technology fails instead of silently building
    graphs at the wrong physical scale.
  - validation split: 85/15 sample-level split of each fold's train samples
    (`carve_validation_from_train`, sample_id treated as patient_id), used only for early
    stopping; the official split itself prevents train/test leakage. Some folds have a single
    train sample (COAD). When fewer than 2 train samples exist, or the carve would empty train,
    the full train pool serves as both train and val (early stopping then monitors train loss)
    and the log says "val==train fallback".

Output: `results/official_hest_ablation_grid/official_ablation_metrics.csv` (one row per cancer x
fold x ablation, same metric columns as `cv_ablation_metrics.csv`) and
`official_ablation_summary_macro.csv` (mean +/- std per ablation, pooled across cancers/folds).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.ablations import ABLATION_GRID
from epathst.data.splits import carve_validation_from_train
from epathst.data.expression import TrainOnlyStandardizer
from epathst.data.feature_cache import extract_student_features
from epathst.evaluation.ablation_runner import run_ablation, train_patch_head
from epathst.evaluation.official_hest import (
    SampleCache,
    build_manifest_rows,
    gather,
    official_pooled_pearson,
)
from epathst.models.student import ConvNeXtV2AttoStudent, load_locked_encoder
from epathst.utils import load_config, resolve_run_paths, save_json, seed_everything

FEATURE_SOURCES = ("distilled", "imagenet")


def run_one_cancer(
    bench_root: Path,
    cancer: str,
    embed_fns: dict,
    cfg: dict,
    seed: int,
    device: str,
    max_folds: int | None,
) -> list[dict]:
    cancer_root = bench_root / cancer
    genes = json.loads((cancer_root / "var_50genes.json").read_text())["genes"]
    split_dir = cancer_root / "splits"
    n_splits = len(list(split_dir.glob("train_*.csv")))
    assert n_splits > 0, f"no official splits found under {split_dir}"
    if max_folds is not None:
        n_splits = min(n_splits, max_folds)

    cache = SampleCache(cancer_root, genes, embed_fns, device)
    metrics_rows = []

    for fold in range(n_splits):
        train_df = pd.read_csv(split_dir / f"train_{fold}.csv")
        test_df = pd.read_csv(split_dir / f"test_{fold}.csv")
        all_df = pd.concat([train_df, test_df], ignore_index=True)

        # Folds can have a single train sample (COAD); carving a val sample would empty train.
        # Fall back to train == val (early stopping monitors train loss) when fewer than 2 train
        # samples exist or the carve leaves none.
        train_samples_all = train_df["sample_id"].tolist()
        if len(train_samples_all) >= 2:
            train_ids_s, val_ids_s = carve_validation_from_train(train_samples_all, 0.15, seed)
            if not train_ids_s:
                train_ids_s, val_ids_s = train_samples_all, train_samples_all
        else:
            train_ids_s, val_ids_s = train_samples_all, train_samples_all
        test_ids_s = test_df["sample_id"].tolist()
        fallback_note = (
            " (val==train fallback, too few samples for a real held-out split)"
            if val_ids_s == train_ids_s
            else ""
        )
        print(
            f"[22_official_hest_ablation_grid] {cancer} fold{fold}: "
            f"{len(train_ids_s)} train / {len(val_ids_s)} val / {len(test_ids_s)} "
            f"test samples{fallback_note}"
        )

        # dedupe in order: in the val==train fallback the same sample appears twice, which would
        # duplicate its spot_uids and break every `.loc[spot_uids]` lookup downstream.
        manifest_sample_ids = list(dict.fromkeys(train_ids_s + val_ids_s + test_ids_s))
        manifest = pd.DataFrame(build_manifest_rows(manifest_sample_ids, cache, all_df))

        ids_train, x_by_src_train, y_train_raw = gather(train_ids_s, cache, all_df)
        ids_val, x_by_src_val, y_val_raw = gather(val_ids_s, cache, all_df)
        ids_test, x_by_src_test, y_test_raw = gather(test_ids_s, cache, all_df)

        std = TrainOnlyStandardizer().fit(y_train_raw, ids=ids_train)
        if val_ids_s != train_ids_s:  # val==train fallback is expected overlap, not leakage
            std.assert_no_leakage(ids_val)
        std.assert_no_leakage(ids_test)
        y_gene_train = std.transform(y_train_raw)
        y_gene_val = std.transform(y_val_raw)
        y_gene_test = std.transform(y_test_raw)

        # dummy all-NaN pathway target (see module docstring): fully masked, zero loss
        y_path_train = np.full((len(ids_train), 1), np.nan, dtype=np.float32)
        y_path_val = np.full((len(ids_val), 1), np.nan, dtype=np.float32)
        y_path_test = np.full((len(ids_test), 1), np.nan, dtype=np.float32)

        x_by_source = {
            src: {
                "train": x_by_src_train[src],
                "val": x_by_src_val[src],
                "test": x_by_src_test[src],
            }
            for src in embed_fns
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
                device,
            )
            for src in embed_fns
        }

        # ids_test -> row index, to reorder y_gene_test into eval_ids_test, run_ablation's
        # prediction order (identity for A0-A3, per-patient-regrouped for graph ablations A4-A9)
        row_of = {uid: i for i, uid in enumerate(ids_test)}

        for spec in ABLATION_GRID:
            src = spec.feature_source if spec.feature_source in x_by_source else "distilled"
            sx_train = x_by_source[src]["train"]
            sx_val = x_by_source[src]["val"]
            sx_test = x_by_source[src]["test"]
            spec_patch_head = patch_heads[src]
            preds, r_metrics, eval_ids_test = run_ablation(
                spec,
                sx_train,
                sx_val,
                sx_test,
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
                manifest,
                cfg,
                seed,
                device,
                n_hvg50=len(genes),
                pathway_names=["_unused_dummy_pathway"],
            )
            r_metrics = {k: v for k, v in r_metrics.items() if not k.startswith("pathway_")}

            # Official metric (docs/reports/HEST_OFFICIAL_ABLATION_RESULTS.md): per-gene pearsonr
            # on pooled test-fold spots, plain mean over genes, as in script 19. Reported next to
            # the Fisher-z metric so A0-A9 compare with the official foundation-model baselines.
            order = [row_of[u] for u in eval_ids_test]
            y_true_ord = y_gene_test[order]
            y_pred = preds["gene"]
            official_pearson_mean = official_pooled_pearson(y_true_ord, y_pred)
            r_metrics["official_pooled_pearson_mean"] = official_pearson_mean

            row = {
                "cancer": cancer,
                "fold": fold,
                "seed": seed,
                "ablation": spec.name,
                "n_train_spots": len(ids_train),
                "n_val_spots": len(ids_val),
                "n_test_spots": len(ids_test),
                "n_train_samples": len(train_ids_s),
                "n_test_samples": len(test_ids_s),
                **r_metrics,
            }
            metrics_rows.append(row)
            r_val = r_metrics.get("fisher_z_mean_patient_R")
            print(
                (
                    f"[22_official_hest_ablation_grid] {cancer} fold{fold} {spec.name}: "
                    f"fisher_z_R={r_val:.4f} official_pooled_R={official_pearson_mean:.4f}"
                )
                if r_val is not None
                else "n/a"
            )
    return metrics_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument(
        "--bench-root", default="/lustre/nvwulf/scratch/schennareddy/HEST-1k/hest_bench"
    )
    ap.add_argument("--cancers", nargs="+", default=["IDC", "COAD", "CCRCC"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--seed",
        type=int,
        default=1,
        help="official BenchmarkConfig default (matches script 19)",
    )
    ap.add_argument(
        "--feature-seed",
        type=int,
        default=0,
        help="which locked distilled-encoder checkpoint to use",
    )
    ap.add_argument(
        "--max-folds-per-cancer",
        type=int,
        default=None,
        help="run only the first N official folds per cancer (time-budgeted / smoke-test runs)",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    bench_root = Path(args.bench_root)
    seed_everything(args.seed)

    checkpoint = torch.load(
        paths["student_checkpoints"] / f"locked_encoder_seed{args.feature_seed}.pt",
        map_location=args.device,
    )
    distilled = load_locked_encoder(checkpoint["encoder_state_dict"], args.device)
    imagenet = ConvNeXtV2AttoStudent(pretrained="imagenet").to(args.device).eval()

    def embed_distilled(imgs):
        return extract_student_features(distilled, imgs, args.device)

    def embed_imagenet(imgs):
        return extract_student_features(imagenet, imgs, args.device)

    embed_fns = {"distilled": embed_distilled, "imagenet": embed_imagenet}

    all_rows = []
    for cancer in args.cancers:
        all_rows.extend(
            run_one_cancer(
                bench_root,
                cancer,
                embed_fns,
                cfg,
                args.seed,
                args.device,
                args.max_folds_per_cancer,
            )
        )

    out_dir = paths["results"] / "official_hest_ablation_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(out_dir / "official_ablation_metrics.csv", index=False)

    macro = (
        metrics_df.groupby("ablation")
        .agg(
            fisher_z_mean_patient_R_mean=("fisher_z_mean_patient_R", "mean"),
            fisher_z_mean_patient_R_std=("fisher_z_mean_patient_R", "std"),
            official_pooled_pearson_mean_mean=("official_pooled_pearson_mean", "mean"),
            official_pooled_pearson_mean_std=("official_pooled_pearson_mean", "std"),
            n_cells=("fisher_z_mean_patient_R", "count"),
        )
        .reset_index()
    )
    priority = {a.name: a.priority for a in ABLATION_GRID}
    macro["priority"] = macro["ablation"].map(priority)
    macro = macro.sort_values("ablation")
    macro.to_csv(out_dir / "official_ablation_summary_macro.csv", index=False)

    save_json({"config": args.__dict__}, out_dir / "run_args.json")
    print(macro.to_string(index=False))
    print(f"[22_official_hest_ablation_grid] wrote {out_dir}")


if __name__ == "__main__":
    main()
