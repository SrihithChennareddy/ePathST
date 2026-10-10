"""Shared statistics for FINAL_GUIDE.md Steps 4-8 (Appendix B). Every analysis script imports
these instead of re-implementing them. NumPy/pandas arrays in, NumPy/pandas out; never touches
global random state (every draw takes an explicit seed or a pre-built index matrix)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv
from sklearn.model_selection import StratifiedKFold
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


def uno_c(y_tr, y_te, risk, tau: float) -> float:
    """Wraps concordance_index_ipcw; raises if tau exceeds training follow-up (step 5.4)."""
    max_follow = y_tr["time"][~y_tr["event"]].max() if (~y_tr["event"]).any() else y_tr["time"].max()
    if tau > y_tr["time"].max():
        raise ValueError(f"tau={tau} exceeds training follow-up {y_tr['time'].max()}")
    return float(concordance_index_ipcw(y_tr, y_te, risk, tau=tau)[0])


def harrell_c(y_te, risk) -> float:
    return float(concordance_index_censored(y_te["event"], y_te["time"], risk)[0])


def fit_coxnet(X_tr, y_tr, n_clin: int, seed: int = 100):
    """Step 5.3 items 3-6: penalty_factor 0 for the first n_clin (clinical) columns, 1 elsewhere;
    alpha path fit once on all of TR, best alpha chosen by 5-fold inner CV (StratifiedKFold on
    events, random_state=seed) maximising mean Harrell's C on the held-out inner fold (ties -> the
    larger alpha); refit on all of TR with that alpha. Returns (fitted_model, best_alpha).
    Falls back to alpha_min_ratio=0.05 if the solver fails to converge."""
    pf = np.concatenate([np.zeros(n_clin), np.ones(X_tr.shape[1] - n_clin)])

    def _fit_path(alpha_min_ratio):
        m = CoxnetSurvivalAnalysis(
            l1_ratio=0.5, n_alphas=100, alpha_min_ratio=alpha_min_ratio, penalty_factor=pf,
            fit_baseline_model=True, max_iter=100000,
        )
        m.fit(X_tr, y_tr)
        return m

    try:
        path_model = _fit_path(0.01)
    except Exception:
        path_model = _fit_path(0.05)
    alphas = path_model.alphas_

    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    mean_c = np.zeros(len(alphas))
    counts = np.zeros(len(alphas))
    for tr_i, va_i in inner.split(X_tr, y_tr["event"]):
        for ai, a in enumerate(alphas):
            try:
                m = CoxnetSurvivalAnalysis(
                    l1_ratio=0.5, alphas=[a], penalty_factor=pf, fit_baseline_model=False,
                    max_iter=100000,
                )
                m.fit(X_tr[tr_i], y_tr[tr_i])
                risk = m.predict(X_tr[va_i])
                mean_c[ai] += harrell_c(y_tr[va_i], risk)
                counts[ai] += 1
            except Exception:
                continue
    counts[counts == 0] = np.nan
    mean_c = mean_c / counts
    valid = np.isfinite(mean_c)
    best_idx = np.where(valid)[0][np.argmax(mean_c[valid])]
    # ties -> the larger alpha (simpler model); alphas_ is already sorted descending by sksurv convention
    best_candidates = np.where(valid & np.isclose(mean_c, mean_c[best_idx]))[0]
    best_idx = best_candidates[np.argmax(alphas[best_candidates])]
    best_alpha = alphas[best_idx]

    final = CoxnetSurvivalAnalysis(
        l1_ratio=0.5, alphas=[best_alpha], penalty_factor=pf, fit_baseline_model=True, max_iter=100000,
    )
    try:
        final.fit(X_tr, y_tr)
    except Exception:
        # Step 5.3.6's sanctioned fallback, applied here too: if even the chosen alpha fails to
        # converge, raise alpha_min_ratio to 0.05 for a fresh path/selection and retry once. Any
        # further failure propagates -- it must not be silently swallowed.
        import logging

        logging.getLogger("epathst.analysis.stats_utils").warning(
            "fit_coxnet: final refit at alpha=%.3g failed to converge, retrying with alpha_min_ratio=0.05", best_alpha
        )
        path_model = _fit_path(0.05)
        alphas = path_model.alphas_
        best_alpha = alphas[np.argmin(np.abs(alphas - best_alpha))]
        final = CoxnetSurvivalAnalysis(
            l1_ratio=0.5, alphas=[best_alpha], penalty_factor=pf, fit_baseline_model=True, max_iter=100000,
        )
        final.fit(X_tr, y_tr)
    return final, float(best_alpha)
