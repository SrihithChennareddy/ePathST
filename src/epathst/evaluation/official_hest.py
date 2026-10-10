"""Official HEST-1k Benchmark protocol: log1p-only 50-gene targets, pooled-spot Pearson, per-sample
feature cache. Defines the target and metric every official-protocol comparison in the paper is
scored with.

Spot coordinates are converted from pixels to um with the 10x Visium convention (55 um spot
diameter); `get_mpp` asserts a plausible scale so unvalidated technologies fail loudly.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr


def decode_barcodes(raw: np.ndarray) -> list[str]:
    out = []
    for b in raw:
        v = b[0] if isinstance(b, np.ndarray) else b
        out.append(v.decode() if isinstance(v, bytes) else str(v))
    return out


def official_target(expr_path: Path, genes: list[str], barcodes: list[str]) -> np.ndarray:
    """Script 19's `official_target`: subset to `genes`/`barcodes`, then log1p only (no
    total-count normalization)."""
    adata = ad.read_h5ad(expr_path)
    adata = adata[barcodes, genes].copy()
    adata.X = adata.X.astype(np.float64)
    sc.pp.log1p(adata)
    x = adata.X
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def official_pooled_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Official HEST-1k Benchmark metric (mahmoodlab/HEST, src/hest/bench/trainer.py
    `train_test_reg`): per-gene pearsonr on pooled test-fold spots (not per-patient), then a
    plain mean across genes. Constant predictions (A0) give NaN per gene and are dropped, as in
    the official `np.isfinite` guard."""
    corrs = []
    for g in range(y_true.shape[1]):
        r, _ = pearsonr(y_true[:, g], y_pred[:, g])
        if np.isfinite(r):
            corrs.append(r)
    return float(np.mean(corrs)) if corrs else float("nan")


def get_mpp(adata) -> float:
    """um per pixel under the 10x Visium convention (55 um spot diameter):
    55 / spot_diameter_fullres. Matched array-grid spacing on a real IDC sample (~0.2126)."""
    tech = next(iter(adata.uns["spatial"].keys()))
    sf = adata.uns["spatial"][tech]["scalefactors"]
    mpp = 55.0 / sf["spot_diameter_fullres"]
    assert 0.05 <= mpp <= 2.0, (
        f"implausible mpp={mpp:.4f} um/px "
        f"(spot_diameter_fullres={sf['spot_diameter_fullres']}, tech={tech}) "
        "-- unexpected ST technology, not validated by this script's mpp convention"
    )
    return mpp


class SampleCache:
    """Loads, embeds (every feature source) and builds targets for each official sample once,
    however many folds reuse it."""

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
            barcodes = decode_barcodes(f["barcode"][:])
            coords_px = f["coords"][:].astype(np.float64)

        adata = ad.read_h5ad(self.cancer_root / expr_path)
        mpp = get_mpp(adata)
        coords_um = coords_px * mpp
        y = official_target(self.cancer_root / expr_path, self.genes, barcodes)
        feats = {src: fn(imgs) for src, fn in self.embed_fns.items()}
        entry = {"barcodes": barcodes, "coords_um": coords_um, "y": y, "feats": feats}
        self._cache[sample_id] = entry
        print(
            f"[official_hest] embedded+targeted {sample_id}: "
            f"{len(barcodes)} spots, mpp={mpp:.4f} um/px"
        )
        return entry


def row_for_sample(sample_id: str, split_df: pd.DataFrame) -> pd.Series:
    return split_df[split_df["sample_id"] == sample_id].iloc[0]


def build_manifest_rows(
    sample_ids: list[str], cache: SampleCache, split_df: pd.DataFrame
) -> list[dict]:
    rows = []
    for sid in sample_ids:
        row = row_for_sample(sid, split_df)
        entry = cache.get(sid, row["patches_path"], row["expr_path"])
        for i, bc in enumerate(entry["barcodes"]):
            rows.append(
                {
                    "spot_uid": f"{sid}::{bc}",
                    "x_um": entry["coords_um"][i, 0],
                    "y_um": entry["coords_um"][i, 1],
                    "patient_id": sid,
                }
            )
    return rows


def gather(sample_ids: list[str], cache: SampleCache, split_df: pd.DataFrame):
    ids, feats_by_src, ys = [], {src: [] for src in cache.embed_fns}, []
    for sid in sample_ids:
        row = row_for_sample(sid, split_df)
        entry = cache.get(sid, row["patches_path"], row["expr_path"])
        ids.extend([f"{sid}::{bc}" for bc in entry["barcodes"]])
        for src in cache.embed_fns:
            feats_by_src[src].append(entry["feats"][src])
        ys.append(entry["y"])
    ids = np.array(ids)
    feats_by_src = {src: np.concatenate(v, axis=0) for src, v in feats_by_src.items()}
    y = np.concatenate(ys, axis=0).astype(np.float32)
    return ids, feats_by_src, y
