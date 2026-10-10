#!/usr/bin/env python
"""Zero-shot Leiden/ARI baseline: frozen H-Optimus-0 patch embeddings clustered directly, with no
prediction head, across the full HEST-1k cohort.
Needs no training or train/test split, so nothing can leak; reads cached HEST patches
(hest_patches/*.h5), no WSI access.

Ground-truth clusters come from each sample's own top-variance genes. The Leiden/ARI recipe is the
same as 32_leiden_full_cohort.py (scale -> PCA<=30 -> kNN15 -> Leiden igraph, resolution searched
to give 5-9 ground-truth clusters).

Earlier variant kept for provenance; official-panel Leiden is now
`src/scripts/25_leiden_official_panel.py official`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from epathst.data.expression import ensure_gene_symbol_index, normalize_log1p_slide
from epathst.data.splits import exclude_targeted_panel_samples
from epathst.models.teacher import HOptimusTeacher
from epathst.utils import RunEnvironment, load_config, resolve_run_paths, save_json


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--hvg",
        type=int,
        default=1000,
        help="per-sample top-variance genes for the GT clustering target",
    )
    ap.add_argument("--max-samples", type=int, default=None, help="debug: stop after N samples")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    device = args.device if torch.cuda.is_available() else "cpu"

    out_dir = Path(cfg["paths"]["persist_dir"]) / "leiden_clustering"
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    hest_samples = pd.read_csv(paths["manifests"] / "hest_samples.csv")
    hest_samples = exclude_targeted_panel_samples(hest_samples)
    patch_dir = Path(cfg["paths"]["work_dir"]) / "hest_patches"

    teacher = HOptimusTeacher(cfg, device=device)
    batch_size = cfg["teacher"]["batch_size"]

    rows = []
    with RunEnvironment("33_leiden_hoptimus_baseline", out_dir / "logs" / "hoptimus_run_env.json"):
        for i, (_, r) in enumerate(hest_samples.iterrows()):
            if args.max_samples and i >= args.max_samples:
                break
            sample_id, cancer = r["sample_id"], r["cancer"]
            patch_path = patch_dir / f"{sample_id}.h5"
            if not patch_path.exists():
                print(f"[33_leiden_hoptimus_baseline] {sample_id}: no cached patches, skipping")
                continue

            with h5py.File(patch_path, "r") as f:
                patch_ids = [p.decode() for p in f["patch_id"][:]]
                patches = f["patches"][:]

            adata = ad.read_h5ad(r["expression_path"])
            adata = ensure_gene_symbol_index(adata)
            normalize_log1p_slide(adata, cfg["molecular_targets"]["normalize_total_target_sum"])
            sc.pp.highly_variable_genes(
                adata, n_top_genes=min(args.hvg, adata.n_vars), flavor="seurat"
            )
            hvg_genes = adata.var_names[adata.var["highly_variable"]].tolist()

            common_ids = sorted(set(patch_ids) & set(adata.obs_names.astype(str)))
            if len(common_ids) < 20:
                print(
                    f"[33_leiden_hoptimus_baseline] {sample_id}: only {len(common_ids)} spots have "
                    f"both a patch and expression, skipping"
                )
                continue

            id_to_patch_row = {pid: i for i, pid in enumerate(patch_ids)}
            patch_rows = [id_to_patch_row[i] for i in common_ids]
            gt_matrix = adata[common_ids, hvg_genes].X
            gt_matrix = (
                gt_matrix.toarray() if hasattr(gt_matrix, "toarray") else np.asarray(gt_matrix)
            )

            # patch_rows must stay in the row order of common_ids / gt_matrix
            sel_patches = patches[patch_rows]
            embeds = [
                teacher.embed_batch(sel_patches[s : s + batch_size])
                for s in range(0, len(sel_patches), batch_size)
            ]
            embed_matrix = np.concatenate(embeds, axis=0)

            gt_labels, gt_res, gt_k = gt_clustering(gt_matrix)
            pred_labels = leiden_cluster(embed_matrix, gt_res)
            ari = float(adjusted_rand_score(gt_labels, pred_labels))
            rows.append(
                {
                    "cancer": cancer,
                    "sample_id": sample_id,
                    "n_spots": len(common_ids),
                    "gt_resolution": gt_res,
                    "gt_n_clusters": gt_k,
                    "ari_HOptimus0": ari,
                }
            )
            print(
                f"[33_leiden_hoptimus_baseline] {sample_id} ({cancer}): ARI={ari:.4f} "
                f"(n_spots={len(common_ids)}, gt_k={gt_k})"
            )
            pd.DataFrame(rows).to_csv(out_dir / "per_sample_ari_hoptimus.csv", index=False)

    per_sample = pd.DataFrame(rows)
    per_sample.to_csv(out_dir / "per_sample_ari_hoptimus.csv", index=False)
    summary = (
        per_sample.groupby("cancer")["ari_HOptimus0"].agg(["mean", "std", "count"]).reset_index()
    )
    overall = pd.DataFrame(
        [
            {
                "cancer": "ALL",
                "mean": per_sample["ari_HOptimus0"].mean(),
                "std": per_sample["ari_HOptimus0"].std(),
                "count": len(per_sample),
            }
        ]
    )
    summary = pd.concat([summary, overall], ignore_index=True)
    summary.to_csv(out_dir / "ari_summary_hoptimus.csv", index=False)
    save_json(summary.to_dict("records"), out_dir / "ari_summary_hoptimus.json")
    print(summary.to_string(index=False))
    print(f"[33_leiden_hoptimus_baseline] wrote {out_dir / 'ari_summary_hoptimus.csv'}")


if __name__ == "__main__":
    main()
