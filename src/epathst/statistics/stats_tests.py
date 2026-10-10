"""Patient-level inference: bootstrap CI, paired sign-flip test and Benjamini-Hochberg FDR.

Inputs are per-patient values, since patients rather than spots are the independent unit.
"""

from __future__ import annotations

import numpy as np


def patient_bootstrap_ci(
    deltas: np.ndarray,
    n_iterations: int = 10000,
    seed: int = 0,
    ci: tuple[float, float] = (2.5, 97.5),
) -> dict:
    """Percentile bootstrap CI of the mean of `deltas` (e.g. per-patient R_real - R_shuffle).

    Also reports Cohen's d (mean / sample SD), NaN for a single patient.
    """
    rng = np.random.RandomState(seed)
    n = len(deltas)
    boot_means = np.empty(n_iterations)
    for i in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(deltas[idx])
    if n > 1:
        effect_size = float(np.mean(deltas) / (np.std(deltas, ddof=1) + 1e-12))
    else:
        effect_size = float("nan")
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_low": float(np.percentile(boot_means, ci[0])),
        "ci_high": float(np.percentile(boot_means, ci[1])),
        "n_patients": n,
        "effect_size_cohens_d": effect_size,
    }


def paired_sign_flip_test(deltas: np.ndarray, n_iterations: int = 10000, seed: int = 0) -> dict:
    """One-sided sign-flip permutation test for H1: mean(delta) > 0.

    p = (1 + #{null >= observed}) / (1 + n_iterations), so p is never exactly 0.
    """
    rng = np.random.RandomState(seed)
    n = len(deltas)
    observed = float(np.mean(deltas))
    null = np.empty(n_iterations)
    for i in range(n_iterations):
        flips = rng.choice([-1.0, 1.0], size=n)
        null[i] = np.mean(deltas * flips)
    p = (1 + np.sum(null >= observed)) / (1 + n_iterations)
    return {"observed_mean": observed, "p_value_one_sided": float(p)}


def bh_fdr(pvalues: np.ndarray, q_threshold: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction. Returns (reject_mask, q_values).

    NaN p-values are skipped: they get q = NaN and are never rejected.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    valid = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if valid.sum() == 0:
        return np.zeros_like(p, dtype=bool), q

    idx = np.where(valid)[0]
    pv = p[idx]

    # BH step-up: raw q = p * m / rank, then enforce monotonicity from the largest rank down
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    raw_q = ranked * m / (np.arange(1, m + 1))
    raw_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    raw_q = np.clip(raw_q, 0, 1)

    q_full = np.empty(m)
    q_full[order] = raw_q
    q[idx] = q_full
    reject = np.zeros_like(p, dtype=bool)
    reject[idx] = q[idx] < q_threshold
    return reject, q
