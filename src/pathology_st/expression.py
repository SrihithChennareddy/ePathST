"""HEST molecular preprocessing: per-slide normalize+log1p, train-fold-only
standardization, and leakage-free per-fold HVG1000/HVG50 selection (nested:
G_50 subset G_1000). Adapted from STS_PoC/PoC_5/crcspatial/expression.py;
generalized for multiple cancer types / ST platforms."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc


def normalize_log1p_slide(adata, target_sum: float = 1e4) -> None:
    """In-place per-slide normalize_total + log1p (never across the pooled
    cohort -- slides differ in technology/sequencing depth)."""
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)


def ensure_gene_symbol_index(adata):
    """HEST slides don't share one gene-ID convention (e.g. within the
    breast/IDC cohort, `NCBI*` samples index by HGNC symbol while `SPA1xx`
    samples index by Ensembl ID with the symbol stashed in
    `var['SYMBOL']`). Everything downstream (PROGENy's net, HVG selection,
    cross-sample gene intersection) is keyed by gene symbol, so an
    Ensembl-indexed slide left unconverted has ~zero overlap with any of
    that -- or makes decoupler hard-crash ('No sources with more than
    tmin=N targets').

    Detects an Ensembl-style index (`ENSG...`) with a `SYMBOL` column and
    remaps to it, dropping genes with no symbol and keeping the first of
    any duplicate symbols. Always finishes with var_names_make_unique(),
    even for slides already symbol-indexed: a few real COAD slides
    (TENX128/154/155/156) are symbol-indexed but contain a handful of
    naturally duplicate symbols that make decoupler's MLM hard-crash
    ('mat contains repeated feature names') otherwise. The suffixed
    duplicates (GENE-1, ...) simply won't match any symbol-keyed
    net/HVG list -- the same harmless outcome as any other absent gene."""
    if adata.var.index.size and str(adata.var.index[0]).startswith("ENSG") and "SYMBOL" in adata.var.columns:
        symbol = adata.var["SYMBOL"].astype(str)
        keep = symbol.notna() & (symbol != "nan") & ~symbol.duplicated(keep="first")
        adata = adata[:, keep.values].copy()
        adata.var_names = symbol[keep].tolist()
    adata.var_names_make_unique()
    return adata


class TrainOnlyStandardizer:
    """Fits mu/sigma on training spots only; applies identically to val/
    test. Never let val/test statistics leak into the scaling -- tracks
    fit ids so leakage can be asserted via `assert_no_leakage`."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mu_: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None
        self._fit_ids: set = set()

    def fit(self, X: np.ndarray, ids=None) -> "TrainOnlyStandardizer":
        self.mu_ = np.nanmean(X, axis=0)
        self.sigma_ = np.nanstd(X, axis=0)
        if ids is not None:
            self._fit_ids = set(ids)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu_) / (self.sigma_ + self.eps)

    def assert_no_leakage(self, other_ids) -> None:
        overlap = self._fit_ids & set(other_ids)
        assert not overlap, f"standardizer was fit using {len(overlap)} ids that also appear in this eval/test set"


def cross_platform_gene_intersection(genes_by_slide: dict[str, set[str]]) -> list[str]:
    """Common genes across slides/platforms, as a deterministic
    (lexicographically sorted) list -- gene order must never change
    silently between calls."""
    common = set.intersection(*genes_by_slide.values()) if genes_by_slide else set()
    return sorted(common)


def select_hvg_for_fold(train_expr: pd.DataFrame, common_genes: list[str], n_hvg1000: int, n_hvg50: int,
                         min_variance: float = 1e-8) -> dict:
    """Rank `common_genes` by variance across TRAINING spots only, drop
    near-zero-variance genes, take top n_hvg1000 as G_1000, and the top
    n_hvg50 of THAT ranking as G_50 (nested targets: G_50 subset G_1000, one
    1000-output model serves both evaluations). Never called with
    validation/test expression."""
    var = train_expr[common_genes].var(axis=0, skipna=True)
    usable = var[var >= min_variance].sort_values(ascending=False)
    hvg1000 = usable.index[:n_hvg1000].tolist()
    hvg50 = hvg1000[:n_hvg50]
    return {
        "hvg1000": hvg1000,
        "hvg50": hvg50,
        "n_common_genes": len(common_genes),
        "n_usable_after_variance_filter": len(usable),
    }
