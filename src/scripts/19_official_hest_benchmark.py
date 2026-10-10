#!/usr/bin/env python
"""Scores the frozen H-Optimus-0 teacher and distilled student under the official HEST-Benchmark
protocol: official splits and genes, log1p-only targets, PCA256 + Ridge head.

Gives numbers comparable to published results, unlike the custom three-cancer experiment in
src/scripts/10_run_patient_cv.py. Data: https://huggingface.co/datasets/MahmoodLab/hest-bench;
the head and target handling follow mahmoodlab/HEST's benchmark/trainer/st_dataset code."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.data.feature_cache import extract_student_features
from epathst.models.student import load_locked_encoder
from epathst.models.teacher import HOptimusTeacher
from epathst.utils import load_config, resolve_run_paths, save_json


def _decode_barcodes(raw: np.ndarray) -> list[str]:
    out = []
    for b in raw:
        v = b[0] if isinstance(b, np.ndarray) else b
        out.append(v.decode() if isinstance(v, bytes) else str(v))
    return out


def official_target(expr_path: Path, genes: list[str], barcodes: list[str]) -> np.ndarray:
    """Matches `load_adata(..., normalize=True)` in st_dataset.py: subset to `genes` and
    `barcodes`, then log1p only (no total-count normalization)."""
    adata = ad.read_h5ad(expr_path)
    adata = adata[barcodes, genes].copy()
    adata.X = adata.X.astype(np.float64)
    sc.pp.log1p(adata)
    x = adata.X
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def embed_official_patches(patches_h5: Path, embed_fn) -> tuple[list[str], np.ndarray]:
    with h5py.File(patches_h5, "r") as f:
        imgs = f["img"][:]
        barcodes = _decode_barcodes(f["barcode"][:])
    return barcodes, embed_fn(imgs)


def run_one_cancer_one_encoder(
    bench_root: Path,
    cancer: str,
    encoder_name: str,
    embed_fn,
    seed: int = 1,
    latent_dim: int = 256,
) -> dict:
    cancer_root = bench_root / cancer
    # var_50genes.json, not mean_50genes.json: the official eval code never reads the latter.
    genes = json.loads((cancer_root / "var_50genes.json").read_text())["genes"]
    split_dir = cancer_root / "splits"
    n_splits = len(list(split_dir.glob("train_*.csv")))
    assert n_splits > 0, f"no official splits found under {split_dir}"

    embed_cache: dict[str, tuple[list[str], np.ndarray]] = {}  # sample_id -> (barcodes, embeddings)

    def load_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        feats, targets = [], []
        for _, row in df.iterrows():
            sample_id = row["sample_id"]
            if sample_id not in embed_cache:
                embed_cache[sample_id] = embed_official_patches(
                    cancer_root / row["patches_path"], embed_fn
                )
            barcodes, emb = embed_cache[sample_id]
            y = official_target(cancer_root / row["expr_path"], genes, barcodes)
            feats.append(emb)
            targets.append(y)
        return np.concatenate(feats, axis=0), np.concatenate(targets, axis=0)

    fold_results = []
    for i in range(n_splits):
        train_df = pd.read_csv(split_dir / f"train_{i}.csv")
        test_df = pd.read_csv(split_dir / f"test_{i}.csv")
        X_train, y_train = load_split(train_df)
        X_test, y_test = load_split(test_df)

        # Scaler and PCA are fit on the train split only, so no test information leaks in.
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=latent_dim, random_state=seed)),
            ]
        )
        X_train_p = pipe.fit_transform(X_train)
        X_test_p = pipe.transform(X_test)

        # Alpha is set from the dimensions, 100 / (latent_dim * n_genes), and is not tuned.
        alpha = 100.0 / (X_train_p.shape[1] * y_train.shape[1])
        reg = Ridge(
            solver="lsqr", alpha=alpha, random_state=seed, fit_intercept=False, max_iter=1000
        )
        reg.fit(X_train_p, y_train)
        preds = reg.predict(X_test_p)

        # Genes with undefined r (constant column) are stored as None and left out of the mean.
        gene_corrs = []
        for g, gene in enumerate(genes):
            r, _ = pearsonr(y_test[:, g], preds[:, g])
            gene_corrs.append({"gene": gene, "pearson_corr": float(r) if np.isfinite(r) else None})
        valid = [g["pearson_corr"] for g in gene_corrs if g["pearson_corr"] is not None]
        fold_results.append(
            {
                "fold": i,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "alpha": alpha,
                "pearson_mean": float(np.mean(valid)) if valid else float("nan"),
                "gene_corrs": gene_corrs,
            }
        )
        print(
            f"[19_official_hest_benchmark] {cancer}/{encoder_name} fold {i}: "
            f"n_train={len(y_train)} n_test={len(y_test)} "
            f"pearson_mean={fold_results[-1]['pearson_mean']:.4f}"
        )

    per_fold_means = [f["pearson_mean"] for f in fold_results]
    return {
        "cancer": cancer,
        "encoder": encoder_name,
        "n_folds": n_splits,
        "genes": genes,
        "seed": seed,
        "latent_dim": latent_dim,
        "pearson_mean": float(np.mean(per_fold_means)),
        "pearson_std": float(np.std(per_fold_means)),
        "folds": fold_results,
    }


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
        help="official BenchmarkConfig default (NOT this project's usual seed 0)",
    )
    ap.add_argument(
        "--feature-seed",
        type=int,
        default=0,
        help="which locked distilled-encoder checkpoint to use",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    bench_root = Path(args.bench_root)

    checkpoint = torch.load(
        paths["student_checkpoints"] / f"locked_encoder_seed{args.feature_seed}.pt",
        map_location=args.device,
    )
    distilled = load_locked_encoder(checkpoint["encoder_state_dict"], args.device)
    teacher = HOptimusTeacher(cfg, device=args.device)

    def embed_distilled(imgs):
        return extract_student_features(distilled, imgs, args.device)

    def embed_hoptimus(imgs):
        bs = cfg["teacher"]["batch_size"]
        batches = [teacher.embed_batch(imgs[s : s + bs]) for s in range(0, len(imgs), bs)]
        return np.concatenate(batches, axis=0)

    encoders = {"distilled_convnextv2_atto": embed_distilled, "h_optimus_0": embed_hoptimus}

    all_results = []
    for cancer in args.cancers:
        for encoder_name, embed_fn in encoders.items():
            res = run_one_cancer_one_encoder(
                bench_root, cancer, encoder_name, embed_fn, seed=args.seed
            )
            all_results.append(res)
            print(
                f"[19_official_hest_benchmark] {cancer}/{encoder_name}: "
                f"pearson_mean={res['pearson_mean']:.4f} +/- {res['pearson_std']:.4f} "
                f"over {res['n_folds']} folds"
            )

    out_dir = paths["results"] / "official_hest_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(all_results, out_dir / "official_hest_benchmark_full.json")
    summary = pd.DataFrame(
        [
            {
                "cancer": r["cancer"],
                "encoder": r["encoder"],
                "n_folds": r["n_folds"],
                "pearson_mean": r["pearson_mean"],
                "pearson_std": r["pearson_std"],
            }
            for r in all_results
        ]
    )
    summary.to_csv(out_dir / "official_hest_benchmark_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"[19_official_hest_benchmark] wrote {out_dir}")


if __name__ == "__main__":
    main()
