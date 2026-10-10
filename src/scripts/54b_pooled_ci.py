#!/usr/bin/env python
"""Step 5.5 follow-up: the "complementary pooled CI" that 54_tcga_survival.py's main nested-CV
pass does not compute. Within each outer split, risks were already converted to percentiles
(rank/(n_TE+1)); here each patient's percentile is averaged over the 5 repeats, Uno's C of that
pooled score is computed on all patients (using the whole cohort as its own IPCW reference), and
1,000 patient bootstrap replicates (shared draws within a cancer x cohort, so comparisons are
paired) give CIs for pooled C, pooled ΔC for every Step 5 comparison, and the U3 retention ratio R
in KIRC. All of this is explicitly "conditional on the fitted models" -- it does not refit
anything; it only re-aggregates already-produced out-of-fold risk percentiles."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

N_BOOT = 1000
log = logging.getLogger("54b_pooled_ci")

PRIMARY = [("C+P", "C", "BRCA", "full")]
SECONDARY = [
    ("C+P", "C", "COAD", "full"), ("C+P", "C", "KIRC", "full"),
    ("C+P", "C+E", "BRCA", "full"), ("C+P", "C+E", "COAD", "full"), ("C+P", "C+E", "KIRC", "full"),
    ("C+E", "C+T", "BRCA", "full"), ("C+E", "C+T", "COAD", "full"), ("C+E", "C+T", "KIRC", "full"),
    ("C+P", "C+T", "BRCA", "full"), ("C+P", "C+T", "COAD", "full"), ("C+P", "C+T", "KIRC", "full"),
    ("C+B", "C", "BRCA", "rna"), ("C+B", "C", "COAD", "rna"), ("C+B", "C", "KIRC", "rna"),
    ("C+B+P", "C+B", "BRCA", "rna"), ("C+B+P", "C+B", "COAD", "rna"), ("C+B+P", "C+B", "KIRC", "rna"),
]
RETENTION = {
    "student": (("C+E", "C", "full"), ("C+HS", "C", "full")),
    "teacher": (("C+T", "C", "full"), ("C+HT", "C", "full")),
}


def load_scores(oof: pd.DataFrame, cancer: str, cohort_tag: str, fset: str) -> pd.Series:
    g = oof[(oof.cancer == cancer) & (oof.cohort == cohort_tag) & (oof.feature_set == fset)]
    return g.groupby("patient")["percentile"].mean()


def load_outcomes(cohort_dir: Path, cancer: str) -> pd.DataFrame:
    co = pd.read_csv(cohort_dir / f"{cancer}_patients.csv")
    return co.set_index("patient")[["event", "time_days"]]


def pooled_c(score: pd.Series, outcomes: pd.DataFrame, tau: float) -> float:
    d = outcomes.reindex(score.index).dropna()
    s = score.reindex(d.index)
    y = Surv.from_arrays(event=d["event"].astype(bool).to_numpy(), time=d["time_days"].to_numpy())
    return float(concordance_index_ipcw(y, y, s.to_numpy(), tau=tau)[0])


def boot_c_series(score: pd.Series, outcomes: pd.DataFrame, tau: float, idx: np.ndarray, patients: np.ndarray) -> np.ndarray:
    d = outcomes.reindex(patients)
    s = score.reindex(patients)
    out = np.full(len(idx), np.nan)
    for i, draw in enumerate(idx):
        dd, ss = d.iloc[draw], s.iloc[draw]
        y = Surv.from_arrays(event=dd["event"].astype(bool).to_numpy(), time=dd["time_days"].to_numpy())
        if not np.any(dd["time_days"].to_numpy() < tau):
            continue
        try:
            out[i] = concordance_index_ipcw(y, y, ss.to_numpy(), tau=tau)[0]
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/tcga_final")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    out_root = Path(args.out)
    surv_dir = out_root / "survival"
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(out_root / "logs" / "54b_pooled_ci.log"), logging.StreamHandler()],
    )
    oof = pd.read_csv(surv_dir / "oof_percentiles.csv")
    cindex = pd.read_csv(surv_dir / "cindex_by_split.csv")
    cohort_dir = out_root / "cohort"

    tau_of = {
        (r.cancer, r.cohort): r.tau_days
        for r in cindex.drop_duplicates(["cancer", "cohort"]).itertuples()
    }
    outcomes_of = {c: load_outcomes(cohort_dir, c) for c in ("BRCA", "COAD", "KIRC")}

    # shared bootstrap draws per (cancer, cohort): same patients, same resample indices, reused
    # across every feature set so comparisons (and the retention ratio) are paired.
    rng = np.random.default_rng(0)
    idx_of, patients_of = {}, {}
    for cancer in ("BRCA", "COAD", "KIRC"):
        for cohort_tag in ("full", "rna"):
            pats = oof[(oof.cancer == cancer) & (oof.cohort == cohort_tag)]["patient"].unique()
            if len(pats) == 0:
                continue
            patients_of[(cancer, cohort_tag)] = pats
            idx_of[(cancer, cohort_tag)] = rng.integers(0, len(pats), size=(args.n_boot, len(pats)))

    # pooled C per feature set
    pooled_rows = []
    boot_cache = {}  # (cancer,cohort,fset) -> boot array
    for (cancer, cohort_tag, fset), g in oof.groupby(["cancer", "cohort", "feature_set"]):
        score = load_scores(oof, cancer, cohort_tag, fset)
        tau = tau_of[(cancer, cohort_tag)]
        c_point = pooled_c(score, outcomes_of[cancer], tau)
        patients = patients_of[(cancer, cohort_tag)]
        idx = idx_of[(cancer, cohort_tag)]
        boot = boot_c_series(score, outcomes_of[cancer], tau, idx, patients)
        boot_cache[(cancer, cohort_tag, fset)] = boot
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        pooled_rows.append(
            {"cancer": cancer, "cohort": cohort_tag, "feature_set": fset, "c_pooled": c_point,
             "ci_low": lo, "ci_high": hi, "n_boot_valid": int(np.isfinite(boot).sum()),
             "label": "conditional on the fitted models"}
        )
        log.info("%s/%s %s: pooled C=%.4f [%.4f, %.4f]", cancer, cohort_tag, fset, c_point, lo, hi)
    pooled_c_df = pd.DataFrame(pooled_rows)
    pooled_c_df.to_csv(surv_dir / "pooled_c.csv", index=False)

    # pooled delta C for every Step 5 comparison (primary + secondary), paired bootstrap
    delta_rows = []
    for a, b, cancer, cohort_tag in PRIMARY + SECONDARY:
        key_a, key_b = (cancer, cohort_tag, a), (cancer, cohort_tag, b)
        if key_a not in boot_cache or key_b not in boot_cache:
            continue
        ca = pooled_c_df[(pooled_c_df.cancer == cancer) & (pooled_c_df.cohort == cohort_tag) & (pooled_c_df.feature_set == a)]["c_pooled"].iloc[0]
        cb = pooled_c_df[(pooled_c_df.cancer == cancer) & (pooled_c_df.cohort == cohort_tag) & (pooled_c_df.feature_set == b)]["c_pooled"].iloc[0]
        delta_boot = boot_cache[key_a] - boot_cache[key_b]
        lo, hi = np.nanpercentile(delta_boot, [2.5, 97.5])
        delta_rows.append(
            {"cancer": cancer, "cohort": cohort_tag, "comparison": f"{a} vs {b}",
             "pooled_delta_c": ca - cb, "ci_low": lo, "ci_high": hi,
             "label": "conditional on the fitted models"}
        )
    pd.DataFrame(delta_rows).to_csv(surv_dir / "pooled_delta_c.csv", index=False)

    # U3 retention ratio R in KIRC, same draws for numerator and denominator
    ret_rows = []
    for name, ((num_a, num_b, coh), (den_a, den_b, coh2)) in RETENTION.items():
        cancer = "KIRC"
        key_num_a, key_num_b = (cancer, coh, num_a), (cancer, coh, num_b)
        key_den_a, key_den_b = (cancer, coh2, den_a), (cancer, coh2, den_b)
        if any(k not in boot_cache for k in (key_num_a, key_num_b, key_den_a, key_den_b)):
            continue
        num_boot = boot_cache[key_num_a] - boot_cache[key_num_b]
        den_boot = boot_cache[key_den_a] - boot_cache[key_den_b]
        with np.errstate(divide="ignore", invalid="ignore"):
            r_boot = num_boot / den_boot
        r_boot = r_boot[np.isfinite(r_boot) & (np.abs(den_boot) > 1e-6)]
        num_point = pooled_c_df[(pooled_c_df.cancer == cancer) & (pooled_c_df.feature_set == num_a)]["c_pooled"].iloc[0] - \
            pooled_c_df[(pooled_c_df.cancer == cancer) & (pooled_c_df.feature_set == num_b)]["c_pooled"].iloc[0]
        den_point = pooled_c_df[(pooled_c_df.cancer == cancer) & (pooled_c_df.feature_set == den_a)]["c_pooled"].iloc[0] - \
            pooled_c_df[(pooled_c_df.cancer == cancer) & (pooled_c_df.feature_set == den_b)]["c_pooled"].iloc[0]
        r_point = num_point / den_point if abs(den_point) > 1e-6 else np.nan
        lo, hi = np.nanpercentile(r_boot, [2.5, 97.5]) if len(r_boot) else (np.nan, np.nan)
        ret_rows.append(
            {"cancer": cancer, "embedding": name, "R": r_point, "ci_low": lo, "ci_high": hi,
             "n_boot_valid": len(r_boot), "label": "conditional on the fitted models"}
        )
        log.info("KIRC R_%s = %.3f [%.3f, %.3f] (n_valid=%d/%d)", name, r_point, lo, hi, len(r_boot), args.n_boot)
    pd.DataFrame(ret_rows).to_csv(surv_dir / "retention_ratio_ci.csv", index=False)

    log.info("wrote pooled_c.csv, pooled_delta_c.csv, retention_ratio_ci.csv")


if __name__ == "__main__":
    main()
