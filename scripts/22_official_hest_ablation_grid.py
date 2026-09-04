#!/usr/bin/env python
"""Runs the full A0-A9 ablation grid (`sts.ablations.ABLATION_GRID`) under
the OFFICIAL HEST-1k Benchmark protocol (official splits/genes/log1p-only
targets from https://huggingface.co/datasets/MahmoodLab/hest-bench) --
answers "how do the patch-only / GNN-corrected ablations compare on the
official 50-gene per-cancer panel and official k-fold splits", as opposed
to scripts/10_run_patient_cv.py's custom HVG1000/HVG50/PROGENy14 protocol
with this project's own patient-level GroupKFold splits.

`scripts/19_official_hest_benchmark.py` already reproduces the official
protocol, but only for two raw encoders via a from-scratch PCA(256)+Ridge
head -- it never touches the A0-A9 grid (trained patch_head + PNA/GATv2
graph correction). This script instead reuses that SAME trained-model code
(via importlib, not reimplemented) so A0-A9 run end-to-end exactly as in
the custom pipeline, just swapping in official data/targets/splits.

Adaptations required, all deliberate and documented here rather than
buried in the diff:
  - target: the official <cancer>/var_50genes.json panel, `sc.pp.log1p`
    only (no normalize_total) -- exactly script 19's `official_target`.
    There is no HVG1000/HVG50 distinction here (only one 50-gene panel);
    `n_hvg50` is passed as len(genes) (=all genes) so the "hvg50" slice
    scripts/10's run_ablation always computes is a no-op alias of the
    full-panel ("hvg1000"-labeled) metrics -- both columns will be
    identical in the output, which is expected, not a bug.
  - pathway (PROGENy14) target: NOT evaluated -- some official-split
    samples carry a targeted gene panel (a few hundred genes) rather than
    whole-transcriptome, so decoupler's PROGENy scoring is not reliably
    computable here. A single all-NaN pathway column is passed through so
    the shared patch_head/graph_loss code (which expects some pathway_dim)
    runs unmodified: `patch_head_loss`/`graph_loss` mask by
    `~isnan(target)`, so an all-NaN column always contributes exactly
    zero loss regardless of `pathway_loss_weight`, and
    `evaluate_predictions` always reports NaN/n-a for it (same convention
    already used for A0_mean_expression's undefined Pearson r). The
    pathway_* output columns are dropped before saving since they carry no
    information.
  - "patient" unit for graph-building/evaluation: one official `sample_id`
    -- the official splits are already sample/patient-level held-out, and
    there is no multi-sample-per-patient concept in this data, so
    `patient_id` is set equal to `sample_id` throughout.
  - physical spot coordinates (um): official patches only ship pixel
    coordinates (`patches/<sample>.h5`'s `coords` field); converted via
    the standard 10x Visium convention `mpp = 55.0 / spot_diameter_fullres`
    (`adata.uns['spatial'][<tech>]['scalefactors']`) -- cross-checked
    during development against the array-grid pixel spacing for a real
    IDC sample (both gave mpp ~= 0.2126 um/px, consistent). Asserted to
    fall in [0.05, 2.0] um/px so a sample on a technology this wasn't
    validated against fails loudly instead of silently building graphs at
    the wrong physical scale.
  - validation split: 85/15 sample-level split of each official fold's
    train samples (`sts.cv_folds.carve_validation_from_train`, treating
    sample_id like a patient_id) -- only for early stopping; the official
    split itself is what prevents train/test leakage. Some official folds
    have as few as 1 train sample (COAD); when there aren't at least 2
    train samples to split (or the normal carve would still zero out
    train), the full train pool is used as BOTH train and val -- early
    stopping then just monitors train loss, a real limitation flagged in
    the run's own log output ("val==train fallback"), not silently
    absorbed.

Output: `results/official_hest_ablation_grid/official_ablation_metrics.csv`
(one row per cancer x fold x ablation, same metric columns
`cv_ablation_metrics.csv` has) and `official_ablation_summary_macro.csv`
(mean +/- std per ablation, pooled across cancers/folds)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts.ablations import ABLATION_GRID
from sts.cv_folds import carve_validation_from_train
from sts.expression import TrainOnlyStandardizer
from sts.feature_cache import extract_student_features
from sts.student_model import ConvNeXtV2AttoStudent, load_locked_encoder
from sts.utils import load_config, resolve_run_paths, save_json, seed_everything

# `10_run_patient_cv.py` starts with a digit -- not a valid module name for a
# normal `import` -- loaded via importlib instead, same precedent as
# scripts/run_genbio_hest_baseline.py's `load_fold_expression` reuse.
_S10_PATH = Path(__file__).resolve().parent / "10_run_patient_cv.py"
_spec = importlib.util.spec_from_file_location("_scripts10_run_patient_cv", _S10_PATH)
_s10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s10)
run_ablation = _s10.run_ablation
train_patch_head = _s10.train_patch_head

FEATURE_SOURCES = ("distilled", "imagenet")


def _decode_barcodes(raw: np.ndarray) -> list[str]:
    out = []
    for b in raw:
        v = b[0] if isinstance(b, np.ndarray) else b
        out.append(v.decode() if isinstance(v, bytes) else str(v))
    return out


def official_target(expr_path: Path, genes: list[str], barcodes: list[str]) -> np.ndarray:
    """Exactly script 19's `official_target`: subset to `genes`/`barcodes`,
    then log1p only (no total-count normalization)."""
    adata = ad.read_h5ad(expr_path)
    adata = adata[barcodes, genes].copy()
    adata.X = adata.X.astype(np.float64)
    sc.pp.log1p(adata)
    x = adata.X
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def get_mpp(adata) -> float:
    """Standard 10x Visium convention: a spot is 55um in diameter, so
    um-per-pixel = 55 / spot_diameter_fullres. See module docstring for the
    cross-check against array-grid spacing."""
    tech = next(iter(adata.uns["spatial"].keys()))
    sf = adata.uns["spatial"][tech]["scalefactors"]
    mpp = 55.0 / sf["spot_diameter_fullres"]
    assert 0.05 <= mpp <= 2.0, (
        f"implausible mpp={mpp:.4f} um/px (spot_diameter_fullres={sf['spot_diameter_fullres']}, tech={tech}) "
        "-- unexpected ST technology, not validated by this script's mpp convention")
    return mpp


class SampleCache:
    """Loads + embeds (every feature source) + targets one official sample
    exactly once, no matter how many of the cancer's folds reuse it."""

    def __init__(self, cancer_root: Path, genes: list[str], embed_fns: dict, device: str):
        self.cancer_root = cancer_root
        self.genes = genes
        self.embed_fns = embed_fns
        self.device = device
        self._cache: dict[str, dict] = {}

    def get(self, sample_id: str, patches_path: str, expr_path: str) -> dict:
        if sample_id in self._cache:
            return self._cache[sample_id]
        with h5py.File(self.cancer_root / patches_path, "r") as f:
            imgs = f["img"][:]
            barcodes = _decode_barcodes(f["barcode"][:])
            coords_px = f["coords"][:].astype(np.float64)
        adata = ad.read_h5ad(self.cancer_root / expr_path)
        mpp = get_mpp(adata)
        coords_um = coords_px * mpp
        y = official_target(self.cancer_root / expr_path, self.genes, barcodes)
        feats = {src: fn(imgs) for src, fn in self.embed_fns.items()}
        entry = {"barcodes": barcodes, "coords_um": coords_um, "y": y, "feats": feats}
        self._cache[sample_id] = entry
        print(f"[22_official_hest_ablation_grid] embedded+targeted {sample_id}: "
              f"{len(barcodes)} spots, mpp={mpp:.4f} um/px")
        return entry


def _row_for(sample_id: str, split_df: pd.DataFrame) -> pd.Series:
    return split_df[split_df["sample_id"] == sample_id].iloc[0]


def build_manifest_rows(sample_ids: list[str], cache: SampleCache, split_df: pd.DataFrame) -> list[dict]:
    rows = []
    for sid in sample_ids:
        row = _row_for(sid, split_df)
        entry = cache.get(sid, row["patches_path"], row["expr_path"])
        for i, bc in enumerate(entry["barcodes"]):
            rows.append({"spot_uid": f"{sid}::{bc}", "x_um": entry["coords_um"][i, 0],
                         "y_um": entry["coords_um"][i, 1], "patient_id": sid})
    return rows


def gather(sample_ids: list[str], cache: SampleCache, split_df: pd.DataFrame):
    ids, feats_by_src, ys = [], {src: [] for src in cache.embed_fns}, []
    for sid in sample_ids:
        row = _row_for(sid, split_df)
        entry = cache.get(sid, row["patches_path"], row["expr_path"])
        ids.extend([f"{sid}::{bc}" for bc in entry["barcodes"]])
        for src in cache.embed_fns:
            feats_by_src[src].append(entry["feats"][src])
        ys.append(entry["y"])
    ids = np.array(ids)
    feats_by_src = {src: np.concatenate(v, axis=0) for src, v in feats_by_src.items()}
    y = np.concatenate(ys, axis=0).astype(np.float32)
    return ids, feats_by_src, y


def run_one_cancer(bench_root: Path, cancer: str, embed_fns: dict, cfg: dict, seed: int, device: str,
                    max_folds: int | None) -> list[dict]:
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

        # Official folds can have as few as 1 train sample (COAD) -- carving a
        # held-out validation sample the normal way would zero out train
        # entirely. Fall back to using the full train pool as both train and
        # val (early stopping then just monitors train loss) whenever there
        # aren't at least 2 train samples to split, or whenever the normal
        # carve would still leave 0 train samples.
        train_samples_all = train_df["sample_id"].tolist()
        if len(train_samples_all) >= 2:
            train_ids_s, val_ids_s = carve_validation_from_train(train_samples_all, 0.15, seed)
            if not train_ids_s:
                train_ids_s, val_ids_s = train_samples_all, train_samples_all
        else:
            train_ids_s, val_ids_s = train_samples_all, train_samples_all
        test_ids_s = test_df["sample_id"].tolist()
        fallback_note = " (val==train fallback, too few samples for a real held-out split)" \
            if val_ids_s == train_ids_s else ""
        print(f"[22_official_hest_ablation_grid] {cancer} fold{fold}: "
              f"{len(train_ids_s)} train / {len(val_ids_s)} val / {len(test_ids_s)} test samples{fallback_note}")

        # dict.fromkeys dedupes while preserving order -- train_ids_s == val_ids_s in the
        # val==train fallback case above, and building manifest rows for the same sample
        # twice would duplicate every one of its spot_uids in the index (breaking every
        # downstream `.loc[spot_uids]` lookup in run_ablation/build_graph_data).
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

        # Dummy all-NaN pathway target -- see module docstring. Always fully
        # masked (path_mask all-False), contributes exactly 0 to every loss.
        y_path_train = np.full((len(ids_train), 1), np.nan, dtype=np.float32)
        y_path_val = np.full((len(ids_val), 1), np.nan, dtype=np.float32)
        y_path_test = np.full((len(ids_test), 1), np.nan, dtype=np.float32)

        x_by_source = {src: {"train": x_by_src_train[src], "val": x_by_src_val[src], "test": x_by_src_test[src]}
                       for src in embed_fns}
        patch_heads = {src: train_patch_head(x_by_source[src]["train"], x_by_source[src]["val"],
                                              y_gene_train, y_path_train, y_gene_val, y_path_val, cfg, device)
                       for src in embed_fns}

        for spec in ABLATION_GRID:
            src = spec.feature_source if spec.feature_source in x_by_source else "distilled"
            sx_train, sx_val, sx_test = x_by_source[src]["train"], x_by_source[src]["val"], x_by_source[src]["test"]
            spec_patch_head = patch_heads[src]
            preds, r_metrics, eval_ids_test = run_ablation(
                spec, sx_train, sx_val, sx_test, ids_train, ids_val, ids_test, y_gene_train, y_path_train,
                y_gene_val, y_path_val, y_gene_test, y_path_test, spec_patch_head, manifest, cfg, seed, device,
                n_hvg50=len(genes), pathway_names=["_unused_dummy_pathway"])
            r_metrics = {k: v for k, v in r_metrics.items() if not k.startswith("pathway_")}
            row = {"cancer": cancer, "fold": fold, "seed": seed, "ablation": spec.name,
                   "n_train_spots": len(ids_train), "n_val_spots": len(ids_val), "n_test_spots": len(ids_test),
                   "n_train_samples": len(train_ids_s), "n_test_samples": len(test_ids_s), **r_metrics}
            metrics_rows.append(row)
            r_val = r_metrics.get("fisher_z_mean_patient_R")
            print(f"[22_official_hest_ablation_grid] {cancer} fold{fold} {spec.name}: "
                  f"official50_R={r_val:.4f}" if r_val is not None else "n/a")
    return metrics_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--bench-root", default="/lustre/nvwulf/scratch/schennareddy/HEST-1k/hest_bench")
    ap.add_argument("--cancers", nargs="+", default=["IDC", "COAD", "CCRCC"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1, help="official BenchmarkConfig default (matches script 19)")
    ap.add_argument("--feature-seed", type=int, default=0, help="which locked distilled-encoder checkpoint to use")
    ap.add_argument("--max-folds-per-cancer", type=int, default=None,
                     help="run only the first N official folds per cancer (time-budgeted / smoke-test runs)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    bench_root = Path(args.bench_root)
    seed_everything(args.seed)

    distilled = load_locked_encoder(
        torch.load(paths["student_checkpoints"] / f"locked_encoder_seed{args.feature_seed}.pt",
                   map_location=args.device)["encoder_state_dict"], args.device)
    imagenet = ConvNeXtV2AttoStudent(pretrained="imagenet").to(args.device).eval()

    def embed_distilled(imgs):
        return extract_student_features(distilled, imgs, args.device)

    def embed_imagenet(imgs):
        return extract_student_features(imagenet, imgs, args.device)

    embed_fns = {"distilled": embed_distilled, "imagenet": embed_imagenet}

    all_rows = []
    for cancer in args.cancers:
        all_rows.extend(run_one_cancer(bench_root, cancer, embed_fns, cfg, args.seed, args.device,
                                        args.max_folds_per_cancer))

    out_dir = paths["results"] / "official_hest_ablation_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(out_dir / "official_ablation_metrics.csv", index=False)

    macro = (metrics_df.groupby("ablation")
             .agg(fisher_z_mean_patient_R_mean=("fisher_z_mean_patient_R", "mean"),
                  fisher_z_mean_patient_R_std=("fisher_z_mean_patient_R", "std"),
                  n_cells=("fisher_z_mean_patient_R", "count"))
             .reset_index())
    priority = {a.name: a.priority for a in ABLATION_GRID}
    macro["priority"] = macro["ablation"].map(priority)
    macro = macro.sort_values("ablation")
    macro.to_csv(out_dir / "official_ablation_summary_macro.csv", index=False)

    save_json({"config": args.__dict__}, out_dir / "run_args.json")
    print(macro.to_string(index=False))
    print(f"[22_official_hest_ablation_grid] wrote {out_dir}")


if __name__ == "__main__":
    main()
