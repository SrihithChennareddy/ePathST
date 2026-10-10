#!/usr/bin/env python
"""Step 5.7 sensitivity analyses (pre-registered in configs/analysis_plan.json's
sensitivity_covariates and aggregation.sensitivity fields): BRCA IDC-only, cancer-specific
covariates (BRCA ER+HER2, KIRC grade), median instead of mean aggregation, and spots-per-patient
as a covariate. Same folds, same CoxNet/CoxPH procedure as 54_tcga_survival.py (reused directly,
not reimplemented). Output in results/tcga_final/survival_sensitivity/."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SRIHITH = Path("/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith")
QC_ALL = SRIHITH / "tcga" / "predictions" / "hex100_leiden_official_hvg50" / "qc_all.csv"
PROJECTS = {"TCGA-BRCA": "BRCA", "TCGA-COAD": "COAD", "TCGA-KIRC": "KIRC"}

log = logging.getLogger("54c_sensitivity")


def load_survival_module():
    spec = importlib.util.spec_from_file_location(
        "survmod", Path(__file__).resolve().parent / "54_tcga_survival.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_frame_median(surv_mod, cancer, project, features, cohort):
    """Same as surv_mod.load_frame but E/P/T come from the median-over-spots parquet instead of
    mean (3.1's aggregation sensitivity); HS/HT/B are unchanged (no median variant defined)."""
    co = pd.read_csv(cohort / f"{cancer}_patients.csv")
    df = co[co["survival_eligible"]][["patient", "event", "time_days"] + surv_mod.CLIN + ["has_rna"]].copy()
    for s in ("E", "P", "T"):
        g = pd.read_parquet(features / f"{cancer}_{s}_median.parquet").set_index("patient")
        g.columns = [f"{s}__{c}" for c in g.columns]
        df = df.merge(g, left_on="patient", right_index=True, how="left")
    hs = pd.read_parquet(features / f"{cancer}_HS.parquet").set_index("patient")
    ht = pd.read_parquet(features / f"{cancer}_HT.parquet").set_index("patient")
    df = df.merge(hs, left_on="patient", right_index=True, how="left")
    df = df.merge(ht, left_on="patient", right_index=True, how="left")
    b = pd.read_parquet(features / f"{cancer}_B.parquet").set_index("patient")
    b.columns = [f"B__{c}" for c in b.columns]
    df = df.merge(b, left_on="patient", right_index=True, how="left")
    return df.reset_index(drop=True)


def run_variant(surv_mod, variant: str, cancer: str, project: str, panel: list, features: Path,
                 cohort: Path, seeds: list, pca_k: int, out_root: Path, deviations: list):
    orig_clin = list(surv_mod.CLIN)
    try:
        if variant == "median_aggregation":
            df_full = load_frame_median(surv_mod, cancer, project, features, cohort)
        else:
            df_full = surv_mod.load_frame(cancer, project, features, cohort)

        extra_cols, extra_note = [], ""
        if variant == "idc_only":
            co = pd.read_csv(cohort / f"{cancer}_patients.csv")
            idc_patients = set(co.loc[co["histological_type"] == "Infiltrating Ductal Carcinoma", "patient"])
            df_full = df_full[df_full["patient"].isin(idc_patients)].reset_index(drop=True)
            extra_note = f"IDC-only: n={len(df_full)}"
        elif variant == "cancer_covariates":
            co = pd.read_csv(cohort / f"{cancer}_patients.csv")
            cov_cols = {"BRCA": ["ER", "HER2"], "KIRC": ["grade_kirc"], "COAD": []}[cancer]
            if not cov_cols:
                return None  # no cancer-specific covariates defined for COAD
            co_cov = co.set_index("patient")[cov_cols]
            df_full = df_full.merge(co_cov, left_on="patient", right_index=True, how="left")
            df_full = df_full.dropna(subset=cov_cols).reset_index(drop=True)
            surv_mod.CLIN = orig_clin + cov_cols
            extra_cols = cov_cols
            extra_note = f"+{cov_cols}: n={len(df_full)}"
        elif variant == "spots_covariate":
            qc = pd.read_csv(QC_ALL)
            qc = qc[qc.project == project][["case", "n_spots"]].rename(columns={"case": "patient"})
            df_full = df_full.merge(qc, on="patient", how="left").dropna(subset=["n_spots"]).reset_index(drop=True)
            surv_mod.CLIN = orig_clin + ["n_spots"]
            extra_cols = ["n_spots"]
            extra_note = f"+n_spots: n={len(df_full)}"

        if len(df_full) < 20 or df_full["event"].sum() < 20:
            log.warning("%s/%s: too few patients/events (%d/%d), skipping", variant, cancer, len(df_full), int(df_full["event"].sum()))
            return None

        sets = surv_mod.FULL_SETS
        log.info("%s/%s: %s, events=%d", variant, cancer, extra_note, int(df_full["event"].sum()))
        cindex, alpha, coef, pcavar, oof, tau, folds = surv_mod.run_cohort(
            cancer, df_full, panel, sets, "full", seeds, pca_k, out_root, deviations
        )
        cindex.insert(0, "variant", variant)
        results = [cindex]

        df_rna = df_full[df_full["has_rna"]].reset_index(drop=True)
        if len(df_rna) >= 20 and df_rna["event"].sum() >= 20:
            cindex_r, *_rest = surv_mod.run_cohort(
                cancer, df_rna, panel, surv_mod.RNA_SETS, "rna", seeds, pca_k, out_root, deviations
            )
            cindex_r.insert(0, "variant", variant)
            results.append(cindex_r)
        return pd.concat(results, ignore_index=True)
    finally:
        surv_mod.CLIN = orig_clin


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="configs/analysis_plan.json")
    ap.add_argument("--out", default="results/tcga_final")
    args = ap.parse_args()
    out_root = Path(args.out)
    out = out_root / "survival_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(out_root / "logs" / "54c_sensitivity.log"), logging.StreamHandler()],
    )
    plan = json.loads(Path(args.plan).read_text())
    panels = {c: plan["panels"][cfg["head"]] for c, cfg in plan["cohorts"].items()}
    seeds = plan["survival"]["repeat_seeds"]
    pca_k = plan["bottleneck_U3"]["pca_components"]
    features, cohort = out_root / "features", out_root / "cohort"
    surv_mod = load_survival_module()

    deviations: list = []
    all_results = []
    variants = {
        "idc_only": ["BRCA"],
        "cancer_covariates": ["BRCA", "KIRC"],
        "median_aggregation": ["BRCA", "COAD", "KIRC"],
        "spots_covariate": ["BRCA", "COAD", "KIRC"],
    }
    rev_projects = {v: k for k, v in PROJECTS.items()}
    for variant, cancers in variants.items():
        for cancer in cancers:
            project = rev_projects[cancer]
            res = run_variant(surv_mod, variant, cancer, project, panels[cancer], features, cohort,
                               seeds, pca_k, out_root, deviations)
            if res is not None:
                all_results.append(res)

    cindex_all = pd.concat(all_results, ignore_index=True)
    cindex_all.to_csv(out / "cindex_by_split_sensitivity.csv", index=False)

    delta_rows = []
    comparisons = [
        ("C+P", "C"), ("C+P", "C+E"), ("C+E", "C+T"), ("C+P", "C+T"),
        ("C+B", "C"), ("C+B+P", "C+B"),
    ]
    for variant in cindex_all["variant"].unique():
        for cancer in cindex_all[cindex_all.variant == variant]["cancer"].unique():
            sub = cindex_all[(cindex_all.variant == variant) & (cindex_all.cancer == cancer)]
            for a, b in comparisons:
                da = sub[sub.feature_set == a].set_index(["cohort", "repeat", "outer_fold"])["uno_c"]
                db = sub[sub.feature_set == b].set_index(["cohort", "repeat", "outer_fold"])["uno_c"]
                common = da.index.intersection(db.index)
                if len(common) == 0:
                    continue
                d = (da.loc[common] - db.loc[common]).to_numpy()
                n_tr = sub[sub.feature_set == a]["n_tr"].mean()
                n_te = sub[sub.feature_set == a]["n_te"].mean()
                mean, lo, hi, t, p = surv_mod.nb_ttest(d, n_tr, n_te)
                delta_rows.append(
                    {"variant": variant, "cancer": cancer, "comparison": f"{a} vs {b}",
                     "mean_delta_c": mean, "ci_low": lo, "ci_high": hi, "p": p, "n_splits": len(common)}
                )
    delta_df = pd.DataFrame(delta_rows)
    from epathst.analysis.stats_utils import bh

    delta_df["q"] = bh(delta_df["p"].to_numpy())
    delta_df.to_csv(out / "delta_c_sensitivity.csv", index=False)

    if deviations:
        dev_path = out_root / "DEVIATIONS.md"
        existing = dev_path.read_text() if dev_path.exists() else "# Deviations\n"
        dev_path.write_text(
            existing.rstrip("\n") + "\n\n## 2026-10-10 -- Step 5.7 sensitivity: tau lowered by Step 5.4's rule\n\n"
            + "\n".join(deviations) + "\n"
        )

    log.info("wrote %s", out)
    log.info("primary-equivalent, BRCA IDC-only, C+P vs C:\n%s",
              delta_df[(delta_df.variant == "idc_only") & (delta_df.comparison == "C+P vs C")].to_string(index=False))


if __name__ == "__main__":
    main()
