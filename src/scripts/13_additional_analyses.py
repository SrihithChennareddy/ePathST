#!/usr/bin/env python
"""Secondary analyses: multiscale-vs-local contribution and per-cancer
consistency, computed from the per-ablation/per-fold metrics scripts 10/12
already wrote to disk."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathology_st.analysis import per_cancer_consistency
from pathology_st.utils import load_config, resolve_run_paths, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    df = pd.read_csv(paths["results"] / "cv_ablation_metrics.csv")

    # per-cancer consistency
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

    # local vs. multiscale delta, pooled across cancers
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
            "pathology_st.metrics.per_patient_per_column_pearson during script 10's run -- "
            "not yet wired to pathology_st.analysis.ground_truth_morans_i_per_gene / "
            "spatial_benefit_vs_autocorrelation / per_pathway_delta."
        ),
    }
    save_json(summary, paths["results"] / "additional_analyses_summary.json")
    print(f"[13_additional_analyses] {summary}")


if __name__ == "__main__":
    main()
