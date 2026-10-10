#!/usr/bin/env python
"""Descriptive secondary analyses: per-cancer consistency of the A2/A5/A6/A7 ablations and the
pooled multiscale-minus-local (A7 - A6) difference in Fisher-z patient-macro R.

Reads cv_ablation_metrics.csv from src/scripts/10_run_patient_cv.py; no hypothesis tests here
(see 12_primary_hypothesis_test.py). The Moran's I and per-pathway analyses are not wired in yet,
since they need per-gene, per-patient r tables saved during the script 10 run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.evaluation.analysis import per_cancer_consistency
from epathst.utils import load_config, resolve_run_paths, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    df = pd.read_csv(paths["results"] / "cv_ablation_metrics.csv")

    by_cancer = {}
    for cancer, sub in df.groupby("cancer"):
        by_cancer[cancer] = {
            "A2_distilled_patch": sub[sub["ablation"] == "A2_distilled_patch"][
                "fisher_z_mean_patient_R"
            ].mean(),
            "A5_shuffled_graph": sub[sub["ablation"] == "A5_shuffled_graph"][
                "fisher_z_mean_patient_R"
            ].mean(),
            "A6_local_pna": sub[sub["ablation"] == "A6_local_pna"][
                "fisher_z_mean_patient_R"
            ].mean(),
            "A7_multiscale_pna": sub[sub["ablation"] == "A7_multiscale_pna"][
                "fisher_z_mean_patient_R"
            ].mean(),
        }
    consistency_df = per_cancer_consistency(by_cancer)
    consistency_df.to_csv(paths["results"] / "per_cancer_consistency.csv", index=False)
    print(f"[13_additional_analyses] per-cancer consistency:\n{consistency_df}")

    # Pooled over all cancers/folds/seeds, so cancers with more runs weigh more.
    local_mean = df[df["ablation"] == "A6_local_pna"]["fisher_z_mean_patient_R"].mean()
    multi_mean = df[df["ablation"] == "A7_multiscale_pna"]["fisher_z_mean_patient_R"].mean()
    summary = {
        "multiscale_minus_local": (
            float(multi_mean - local_mean)
            if pd.notna(local_mean) and pd.notna(multi_mean)
            else None
        ),
        "note": (
            "Moran's I vs delta_r correlation and per-pathway delta ranking need the full "
            "per-gene/per-pathway per-patient r tables from "
            "epathst.evaluation.metrics.per_patient_per_column_pearson during script 10's run -- "
            "not yet wired to epathst.evaluation.analysis.ground_truth_morans_i_per_gene / "
            "spatial_benefit_vs_autocorrelation / per_pathway_delta."
        ),
    }
    save_json(summary, paths["results"] / "additional_analyses_summary.json")
    print(f"[13_additional_analyses] {summary}")


if __name__ == "__main__":
    main()
