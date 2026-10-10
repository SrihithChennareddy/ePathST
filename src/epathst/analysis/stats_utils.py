"""Shared statistics for FINAL_GUIDE.md Steps 4-8 (Appendix B). Every analysis script imports
these instead of re-implementing them. NumPy/pandas arrays in, NumPy/pandas out; never touches
global random state (every draw takes an explicit seed or a pre-built index matrix)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def spearman_boot(x, y, n_boot: int = 2000, seed: int = 0, idx: np.ndarray | None = None):
    """Spearman rho with a percentile bootstrap CI.

    NaN pairs are dropped before anything else. `idx`, if given, is an (n_boot, n) index matrix
    of patient positions (into the NaN-dropped arrays) so several systems can share the same
    resamples; if `idx` doesn't match the dropped length, draws are made fresh and `idx` is
    ignored for this call (caller is responsible for aligning idx to a common patient set first).
    Returns (rho, ci_low, ci_high, p).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan
    rho, _ = stats.spearmanr(x, y)
    if idx is not None and idx.shape[1] == n:
        draws = idx
    else:
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, n, size=(n_boot, n))
    boot_rho = np.empty(draws.shape[0])
    for i, d in enumerate(draws):
        xb, yb = x[d], y[d]
        if np.std(xb) == 0 or np.std(yb) == 0:
            boot_rho[i] = np.nan
        else:
            boot_rho[i] = stats.spearmanr(xb, yb)[0]
    lo, hi = np.nanpercentile(boot_rho, [2.5, 97.5])
    if n >= 100:
        p = stats.spearmanr(x, y)[1]
    else:
        rng = np.random.default_rng(seed + 1)
        p = perm_p_single(x, y, n_perm=10000, seed=seed + 1)
    return float(rho), float(lo), float(hi), float(p)


def perm_p_single(x, y, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided permutation p-value for Spearman rho(x, y) by shuffling y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    obs = abs(stats.spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    n = len(y)
    count = 0
    for _ in range(n_perm):
        yp = y[rng.permutation(n)]
        if abs(stats.spearmanr(x, yp)[0]) >= obs:
            count += 1
    return (1 + count) / (n_perm + 1)


def perm_p(stat_fn, labels, groups, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided permutation p-value: permute `labels` within each level of `groups`, recomputing
    `stat_fn(labels_perm)` each time; p = (1 + #|null| >= |observed|) / (n_perm + 1)."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    obs = stat_fn(labels)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = labels.copy()
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            perm[idx] = labels[rng.permutation(idx)]
        if abs(stat_fn(perm)) >= abs(obs):
            count += 1
    return (1 + count) / (n_perm + 1)


def partial_spearman(x, y, z):
    """Partial Spearman of x and y given z: rank all three, residualize x and y on z (OLS with
    intercept), Pearson r of the residuals, t-test df = n - 3. Returns (r, p, n)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)
    if n < 4:
        return np.nan, np.nan, n
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)

    def resid(a, b):
        A = np.column_stack([np.ones_like(b), b])
        beta, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ beta

    ex, ey = resid(rx, rz), resid(ry, rz)
    r, _ = stats.pearsonr(ex, ey)
    df = n - 3
    if df <= 0 or abs(r) >= 1:
        return float(r), np.nan, n
    t = r * np.sqrt(df / (1 - r**2))
    p = 2 * stats.t.sf(abs(t), df)
    return float(r), float(p), n


def bh(pvals) -> np.ndarray:
    """Benjamini-Hochberg q-values; NaNs are ignored and kept in their positions."""
    pvals = np.asarray(pvals, dtype=float)
    q = np.full(pvals.shape, np.nan)
    mask = np.isfinite(pvals)
    if mask.sum() == 0:
        return q
    _, qvals, _, _ = multipletests(pvals[mask], method="fdr_bh")
    q[mask] = qvals
    return q


def nb_ttest(deltas, n_train: int, n_test: int):
    """Nadeau-Bengio corrected resampled t-test on paired deltas (one per CV split).
    Returns (mean, lo, hi, t, p) with the 95% corrected CI and df = J - 1."""
    deltas = np.asarray(deltas, dtype=float)
    J = len(deltas)
    mean = deltas.mean()
    s2 = deltas.var(ddof=1)
    corrected_var = (1.0 / J + n_test / n_train) * s2
    se = np.sqrt(corrected_var)
    df = J - 1
    t = mean / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), df) if se > 0 else np.nan
    tcrit = stats.t.ppf(0.975, df)
    lo, hi = mean - tcrit * se, mean + tcrit * se
    return float(mean), float(lo), float(hi), float(t), float(p)


def to_percentile(risk) -> np.ndarray:
    """rank / (n + 1), values in (0, 1)."""
    risk = np.asarray(risk, dtype=float)
    n = len(risk)
    ranks = stats.rankdata(risk)
    return ranks / (n + 1)


def rank_biserial(a, b) -> float:
    """Rank-biserial effect size for Mann-Whitney: 2U/(n_a n_b) - 1."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    u, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(2 * u / (len(a) * len(b)) - 1)
