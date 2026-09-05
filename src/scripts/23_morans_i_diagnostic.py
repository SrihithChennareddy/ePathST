#!/usr/bin/env python
"""Tests whether the A8 graph correction's per-gene gain over A2 (patch-only)
tracks each gene's real spatial autocorrelation (ground-truth Moran's I) --
the direct test of "does the GNN exploit genuine spatial structure", as
opposed to Gate 3's indirect test (A8 vs a shuffled-graph control). Replicates
the same diagnostic POC_5 ran (`pathology_st.analysis.spatial_benefit_vs_autocorrelation`,
which found Corr(delta, MoranI) = 0.1148, p = 0.630 -- no relationship) on
the current official HEST-1k protocol/data, per-cancer and pooled."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathology_st.ablations import ABLATION_GRID
from pathology_st.analysis import ground_truth_morans_i_per_gene, spatial_benefit_vs_autocorrelation
from pathology_st.cv_folds import carve_validation_from_train
from pathology_st.expression import TrainOnlyStandardizer
from pathology_st.feature_cache import extract_student_features
from pathology_st.metrics import per_patient_per_column_pearson
from pathology_st.student_model import ConvNeXtV2AttoStudent, load_locked_encoder
from pathology_st.utils import load_config, resolve_run_paths, save_json, seed_everything

# reuse script 22's already-built official-protocol plumbing (SampleCache, official_target,
# gather, build_manifest_rows, run_ablation, train_patch_head) instead of reimplementing it --
# same importlib precedent script 22 itself uses to reuse script 10.
_S22_PATH = Path(__file__).resolve().parent / "22_official_hest_ablation_grid.py"
_spec = importlib.util.spec_from_file_location("_scripts22_official_hest_ablation_grid", _S22_PATH)
_s22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s22)

SPEC_BY_NAME = {a.name: a for a in ABLATION_GRID}
A2_SPEC = SPEC_BY_NAME["A2_distilled_patch"]
A8_SPEC = SPEC_BY_NAME["A8_scale_aware_pna"]


def gather_coords(sample_ids: list[str], cache, split_df: pd.DataFrame) -> np.ndarray:
    """Same iteration order as `_s22.gather` (over sample_ids, using each
    sample's own cached spot order), so it can be reordered by the identical
    `order = [row_of[u] for u in eval_ids_test]` index script 22 uses for y/preds."""
    out = []
    for sid in sample_ids:
        row = _s22._row_for(sid, split_df)
        entry = cache.get(sid, row["patches_path"], row["expr_path"])
        out.append(entry["coords_um"])
    return np.concatenate(out, axis=0)


def run_one_cancer(bench_root: Path, cancer: str, embed_fns: dict, cfg: dict, seed: int, device: str,
                    max_folds: int | None) -> pd.DataFrame:
    import json
    cancer_root = bench_root / cancer
    genes = json.loads((cancer_root / "var_50genes.json").read_text())["genes"]
    split_dir = cancer_root / "splits"
    n_splits = len(list(split_dir.glob("train_*.csv")))
    assert n_splits > 0, f"no official splits found under {split_dir}"
    if max_folds is not None:
        n_splits = min(n_splits, max_folds)

    cache = _s22.SampleCache(cancer_root, genes, embed_fns, device)
    per_gene_rows = []  # one row per (fold, gene): r_a2, r_a8, morans_i

    for fold in range(n_splits):
        train_df = pd.read_csv(split_dir / f"train_{fold}.csv")
        test_df = pd.read_csv(split_dir / f"test_{fold}.csv")
        all_df = pd.concat([train_df, test_df], ignore_index=True)

        train_samples_all = train_df["sample_id"].tolist()
        if len(train_samples_all) >= 2:
            train_ids_s, val_ids_s = carve_validation_from_train(train_samples_all, 0.15, seed)
            if not train_ids_s:
                train_ids_s, val_ids_s = train_samples_all, train_samples_all
        else:
            train_ids_s, val_ids_s = train_samples_all, train_samples_all
        test_ids_s = test_df["sample_id"].tolist()

        manifest_sample_ids = list(dict.fromkeys(train_ids_s + val_ids_s + test_ids_s))
        manifest = pd.DataFrame(_s22.build_manifest_rows(manifest_sample_ids, cache, all_df))

        ids_train, x_by_src_train, y_train_raw = _s22.gather(train_ids_s, cache, all_df)
        ids_val, x_by_src_val, y_val_raw = _s22.gather(val_ids_s, cache, all_df)
        ids_test, x_by_src_test, y_test_raw = _s22.gather(test_ids_s, cache, all_df)
        coords_test = gather_coords(test_ids_s, cache, all_df)  # same spot order as ids_test/y_test_raw

        std = TrainOnlyStandardizer().fit(y_train_raw, ids=ids_train)
        if val_ids_s != train_ids_s:
            std.assert_no_leakage(ids_val)
        std.assert_no_leakage(ids_test)
        y_gene_train = std.transform(y_train_raw)
        y_gene_val = std.transform(y_val_raw)
        y_gene_test = std.transform(y_test_raw)

        y_path_train = np.full((len(ids_train), 1), np.nan, dtype=np.float32)
        y_path_val = np.full((len(ids_val), 1), np.nan, dtype=np.float32)
        y_path_test = np.full((len(ids_test), 1), np.nan, dtype=np.float32)

        x_by_source = {src: {"train": x_by_src_train[src], "val": x_by_src_val[src], "test": x_by_src_test[src]}
                       for src in embed_fns}
        patch_heads = {src: _s22.train_patch_head(x_by_source[src]["train"], x_by_source[src]["val"],
                                                   y_gene_train, y_path_train, y_gene_val, y_path_val, cfg, device)
                       for src in embed_fns}

        row_of = {uid: i for i, uid in enumerate(ids_test)}
        patient_ids_test_raw = np.array([uid.split("::")[0] for uid in ids_test])

        # ground-truth Moran's I for this fold's test spots -- computed once per fold (not
        # per-ablation, it depends only on y_gene_test/coords_test), same convention as
        # spatial_benefit_vs_autocorrelation's expected per-gene column format.
        morans_df = ground_truth_morans_i_per_gene(y_gene_test, coords_test, patient_ids_test_raw, genes, k=6)
        morans_by_gene = morans_df.set_index("column")["morans_i"]

        r_by_gene = {}
        for spec in (A2_SPEC, A8_SPEC):
            src = spec.feature_source if spec.feature_source in x_by_source else "distilled"
            sx_train, sx_val, sx_test = x_by_source[src]["train"], x_by_source[src]["val"], x_by_source[src]["test"]
            spec_patch_head = patch_heads[src]
            preds, _, eval_ids_test = _s22.run_ablation(
                spec, sx_train, sx_val, sx_test, ids_train, ids_val, ids_test, y_gene_train, y_path_train,
                y_gene_val, y_path_val, y_gene_test, y_path_test, spec_patch_head, manifest, cfg, seed, device,
                n_hvg50=len(genes), pathway_names=["_unused_dummy_pathway"])
            order = [row_of[u] for u in eval_ids_test]
            y_true_ord = y_gene_test[order]
            patient_ids_ord = patient_ids_test_raw[order]
            per_patient_gene = per_patient_per_column_pearson(y_true_ord, preds["gene"], patient_ids_ord,
                                                                genes, min_regions=1)
            r_by_gene[spec.name] = per_patient_gene.dropna(subset=["r"]).groupby("column")["r"].mean()
            print(f"[23_morans_i_diagnostic] {cancer} fold{fold} {spec.name}: computed {len(r_by_gene[spec.name])} "
                  f"per-gene R values")

        for gene in genes:
            per_gene_rows.append({
                "cancer": cancer, "fold": fold, "gene": gene,
                "r_a2": r_by_gene["A2_distilled_patch"].get(gene, np.nan),
                "r_a8": r_by_gene["A8_scale_aware_pna"].get(gene, np.nan),
                "morans_i": morans_by_gene.get(gene, np.nan),
            })

    return pd.DataFrame(per_gene_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--bench-root", default="/lustre/nvwulf/scratch/schennareddy/HEST-1k/hest_bench")
    ap.add_argument("--cancers", nargs="+", default=["IDC", "COAD", "CCRCC"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1, help="official BenchmarkConfig default (matches scripts 19/22)")
    ap.add_argument("--feature-seed", type=int, default=0, help="which locked distilled-encoder checkpoint to use")
    ap.add_argument("--max-folds-per-cancer", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    bench_root = Path(args.bench_root)
    seed_everything(args.seed)

    # load frozen encoders
    distilled = load_locked_encoder(
        torch.load(paths["student_checkpoints"] / f"locked_encoder_seed{args.feature_seed}.pt",
                   map_location=args.device)["encoder_state_dict"], args.device)
    imagenet = ConvNeXtV2AttoStudent(pretrained="imagenet").to(args.device).eval()

    def embed_distilled(imgs):
        return extract_student_features(distilled, imgs, args.device)

    def embed_imagenet(imgs):
        return extract_student_features(imagenet, imgs, args.device)

    embed_fns = {"distilled": embed_distilled, "imagenet": embed_imagenet}

    per_cancer_dfs = []
    for cancer in args.cancers:
        per_cancer_dfs.append(run_one_cancer(bench_root, cancer, embed_fns, cfg, args.seed, args.device,
                                              args.max_folds_per_cancer))
    full_df = pd.concat(per_cancer_dfs, ignore_index=True)

    # average per-gene R and Moran's I across folds within a cancer (panel is fixed per cancer)
    per_gene_avg = (full_df.groupby(["cancer", "gene"])
                    .agg(r_a2=("r_a2", "mean"), r_a8=("r_a8", "mean"), morans_i=("morans_i", "mean"))
                    .reset_index())
    per_gene_avg["delta"] = per_gene_avg["r_a8"] - per_gene_avg["r_a2"]

    out_dir = paths["results"] / "morans_i_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_dir / "per_fold_per_gene.csv", index=False)
    per_gene_avg.to_csv(out_dir / "per_gene_averaged.csv", index=False)

    # spatial_benefit_vs_autocorrelation expects a "column" key (matching
    # ground_truth_morans_i_per_gene's own output schema) to merge delta/morans_i on --
    # rename "gene" -> "column" explicitly rather than relying on pandas' implicit
    # common-column merge fallback.
    def _corr(df: pd.DataFrame) -> dict:
        renamed = df.rename(columns={"gene": "column"}).set_index("column")
        return spatial_benefit_vs_autocorrelation(renamed[["delta"]], renamed[["morans_i"]].reset_index())

    results = {}
    for cancer in args.cancers:
        results[cancer] = _corr(per_gene_avg[per_gene_avg["cancer"] == cancer])
        print(f"[23_morans_i_diagnostic] {cancer}: n_genes={results[cancer]['n_genes']} "
              f"pearson_r={results[cancer]['pearson_r']:.4f} p={results[cancer]['p_value']:.4f}"
              if results[cancer]["n_genes"] >= 5 else f"[23_morans_i_diagnostic] {cancer}: too few valid genes")

    # pooled: prefix gene names by cancer so identical gene symbols across cancers don't collide
    pooled = per_gene_avg.copy()
    pooled["gene"] = pooled["cancer"] + ":" + pooled["gene"]
    results["pooled"] = _corr(pooled)
    print(f"[23_morans_i_diagnostic] pooled: n_genes={results['pooled']['n_genes']} "
          f"pearson_r={results['pooled']['pearson_r']:.4f} p={results['pooled']['p_value']:.4f}")

    save_json(results, out_dir / "spatial_benefit_vs_autocorrelation.json")
    print(f"[23_morans_i_diagnostic] wrote {out_dir}")


if __name__ == "__main__":
    main()
