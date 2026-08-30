"""HEST molecular preprocessing (Final_Project.txt Section 14 / Step 11):
per-slide normalize+log1p, train-fold-only standardization, and
leakage-free per-fold HVG1000/HVG50 selection (nested: G_50 subset G_1000).
Adapted from STS_PoC/PoC_5/crcspatial/expression.py; generalized for
multiple cancer types / ST platforms."""
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
    """Some HEST slides (confirmed live, 2026-08-30: a real mix within the
    breast/IDC cohort -- e.g. `NCBI*` samples already index by HGNC gene
    SYMBOL, `SPA1xx` samples index by Ensembl gene ID with the symbol
    stashed in `var['SYMBOL']` instead) don't share one gene-ID convention.
    Everything downstream (PROGENy's net, HVG selection, cross-sample gene
    intersection) is keyed by gene SYMBOL, so an Ensembl-indexed slide has
    ~zero overlap with any of that and silently produces nonsense (or, for
    PROGENy, decoupler hard-crashes with 'No sources with more than
    tmin=N targets'). Detects an Ensembl-style index (`ENSG...`) with a
    `SYMBOL` column present and remaps to it; drops genes with no symbol
    and collapses duplicate symbols (keep first -- confirmed a small
    minority, ~3 in 16.7k genes on a real affected slide) so var_names
    stay unique. Slides already indexed by symbol are returned unchanged
    EXCEPT for a final var_names_make_unique() pass -- confirmed live,
    2026-08-30: 4 real COAD slides (TENX156/155/154/128) already index by
    symbol but have a handful (3/18085) of naturally duplicate gene
    symbols, which decoupler's mlm hard-crashes on ('mat contains repeated
    feature names') regardless of whether the Ensembl-remap path above
    ran. var_names_make_unique() suffixes duplicates (GENE, GENE-1, ...)
    rather than dropping them -- the suffixed copies just won't match any
    gene-symbol-keyed net/HVG list, which is the same harmless outcome as
    any other gene absent from those, not a crash."""
    if adata.var.index.size and str(adata.var.index[0]).startswith("ENSG") and "SYMBOL" in adata.var.columns:
        symbol = adata.var["SYMBOL"].astype(str)
        keep = symbol.notna() & (symbol != "nan") & ~symbol.duplicated(keep="first")
        adata = adata[:, keep.values].copy()
        adata.var_names = symbol[keep].tolist()
    adata.var_names_make_unique()
    return adata


class TrainOnlyStandardizer:
    """Fits mu/sigma on training spots only; applies identically to val/
    test (Section 14: 'Never allow test-patient gene-expression statistics
    to determine ... scaling'). Tracks fit ids so leakage can be asserted."""

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
    """Section 14: 'Handle cross-platform gene intersections explicitly.
    Never silently change gene order.' Returns a deterministic
    (lexicographically sorted) common-gene list."""
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
