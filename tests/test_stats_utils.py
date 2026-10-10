"""Unit tests for epathst.analysis.stats_utils (FINAL_GUIDE.md Appendix B)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pingouin as pg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from epathst.analysis import stats_utils as su  # noqa: E402


def test_spearman_boot_perfect_monotone():
    x = np.arange(50, dtype=float)
    y = x**3
    rho, lo, hi, p = su.spearman_boot(x, y, n_boot=500, seed=0)
    assert np.isclose(rho, 1.0)
    assert np.isclose(lo, 1.0) and np.isclose(hi, 1.0)
    assert p < 0.05


def test_spearman_boot_independent_ci_covers_zero_about_95_percent():
    rng = np.random.default_rng(0)
    covers = 0
    n_sims = 200
    for i in range(n_sims):
        x = rng.normal(size=500)
        y = rng.normal(size=500)
        _, lo, hi, _ = su.spearman_boot(x, y, n_boot=200, seed=i)
        covers += lo <= 0 <= hi
    assert covers / n_sims > 0.85  # loose bound; exact 95% needs more sims/bootstraps


def test_partial_spearman_matches_pingouin():
    rng = np.random.default_rng(0)
    n = 300
    z = rng.normal(size=n)
    x = z + rng.normal(scale=0.5, size=n)
    y = z + rng.normal(scale=0.5, size=n)
    r, p, n_out = su.partial_spearman(x, y, z)
    df = {"x": x, "y": y, "z": z}
    import pandas as pd

    res = pg.partial_corr(data=pd.DataFrame(df), x="x", y="y", covar="z", method="spearman")
    assert abs(r - res["r"].iloc[0]) < 1e-2  # rank-residual OLS vs pingouin's own partial rank corr
    assert n_out == n


def test_perm_p_uniform_under_null():
    # groups = stratum (e.g. cancer); labels = categorical assignment permuted within each stratum;
    # outcome is a fixed external variable independent of labels, so the null is genuinely true.
    groups = np.repeat([0, 1, 2], 20)
    outcome = np.random.default_rng(0).normal(size=60)
    ps = []
    for i in range(200):
        labels = np.random.default_rng(1000 + i).integers(0, 2, size=60)

        def stat(lab):
            return outcome[lab == 0].mean() - outcome[lab == 1].mean()

        ps.append(su.perm_p(stat, labels, groups, n_perm=300, seed=i))
    ps = np.array(ps)
    assert 0.25 < (ps < 0.5).mean() < 0.75  # loose uniformity check, not a strict KS test


def test_bh_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    p = np.array([0.001, 0.01, 0.03, 0.04, 0.5, np.nan, 0.2])
    q = su.bh(p)
    mask = np.isfinite(p)
    _, q_expected, _, _ = multipletests(p[mask], method="fdr_bh")
    assert np.allclose(q[mask], q_expected)
    assert np.isnan(q[~mask]).all()


def test_nb_ttest_hand_example():
    rng = np.random.default_rng(0)
    deltas = 0.01 + rng.normal(scale=0.001, size=25)
    mean, lo, hi, t, p = su.nb_ttest(deltas, n_train=200, n_test=50)
    s2 = deltas.var(ddof=1)
    expected_se = np.sqrt((1 / 25 + 50 / 200) * s2)
    assert abs(mean - deltas.mean()) < 1e-12
    assert abs((hi - lo) / 2 - expected_se * 2.064) < 1e-6  # t(0.975, df=24) ~= 2.064


def test_to_percentile_monotone_in_open_interval():
    risk = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    pct = su.to_percentile(risk)
    assert (pct > 0).all() and (pct < 1).all()
    assert np.argsort(risk).tolist() == np.argsort(pct).tolist()


def test_rank_biserial_all_a_greater_than_b():
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([1.0, 2.0, 3.0])
    assert su.rank_biserial(a, b) == 1.0
