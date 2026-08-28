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
