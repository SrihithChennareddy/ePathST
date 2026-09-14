#!/usr/bin/env python
"""Builds the MODEL x {HVG50, HVG1000, pathway} results table (mean +/-
95% CI at the patient level), the per-cancer breakdown table, and figures
1/4/5/7 -- the figures directly derivable from tabular results already on
disk (2/3/6/8 are produced inline by scripts 02/06/13 since they need raw
images/embeddings)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.reporting import figures
from epathst.statistics.stats_tests import patient_bootstrap_ci
from epathst.utils import load_config, resolve_run_paths

MODEL_ORDER = [
    "A0_mean_expression", "A1_imagenet_patch", "A2_distilled_patch", "A5_shuffled_graph",
    "A6_local_pna", "A7_multiscale_pna", "A8_scale_aware_pna", "A3_matched_mlp",
    "A4_random_graph", "A9_gatv2_multiscale",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    df = pd.read_csv(paths["results"] / "cv_ablation_metrics.csv")

    def summarize(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) > 1:
            ci = patient_bootstrap_ci(vals, n_iterations=cfg["statistics"]["n_bootstrap"])
            return ci.get("mean_delta", ci.get("mean")), ci["ci_low"], ci["ci_high"], len(vals)
        return (float(vals.mean()) if len(vals) else np.nan), np.nan, np.nan, len(vals)

    rows = []
    for model in MODEL_ORDER:
        sub = df[df["ablation"] == model]
        row = {"model": model}
        targets = [
            ("hvg50", "hvg50_fisher_z_mean_patient_R"),
            ("hvg1000", "hvg1000_fisher_z_mean_patient_R"),
            ("pathway", "pathway_fisher_z_mean_patient_R"),
        ]
        for target, col in targets:
            if sub.empty or col not in sub.columns:
                row.update({
                    f"{target}_r_mean": None,
                    f"{target}_r_ci_low": None,
                    f"{target}_r_ci_high": None,
                })
                continue
            mean, lo, hi, n = summarize(sub[col].values)
            row.update({
                f"{target}_r_mean": mean,
                f"{target}_r_ci_low": lo,
                f"{target}_r_ci_high": hi,
            })
        row["n_folds_seeds"] = len(sub)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(paths["results"] / "final_results_table.csv", index=False)
    print(f"[17_results_and_figures] final results table:\n{table.to_string(index=False)}")

    per_cancer_models = [
        "A2_distilled_patch", "A7_multiscale_pna", "A8_scale_aware_pna", "A9_gatv2_multiscale",
    ]
    per_cancer_rows = []
    for cancer in cfg["cancers"]["hest"]:
        for model in per_cancer_models:
            sub = df[(df["ablation"] == model) & (df["cancer"] == cancer)]
            per_cancer_rows.append({
                "cancer": cancer,
                "model": model,
                "hvg50_mean_R": sub["hvg50_fisher_z_mean_patient_R"].mean() if len(sub) else None,
                "hvg1000_mean_R": sub["hvg1000_fisher_z_mean_patient_R"].mean() if len(sub) else None,
                "pathway_mean_R": sub["pathway_fisher_z_mean_patient_R"].mean() if len(sub) else None,
            })
    per_cancer_path = paths["results"] / "per_cancer_results_table.csv"
    pd.DataFrame(per_cancer_rows).to_csv(per_cancer_path, index=False)

    # Figures directly derivable from `table` -- HVG1000 is the primary reported target
    valid = table.dropna(subset=["hvg1000_r_mean"])
    figures.fig1_pipeline_diagram(paths["figures"] / "fig1_pipeline.png")
    if len(valid):
        renamed = valid.rename(columns={"hvg1000_r_mean": "fisher_z_mean_patient_R"})
        figures.fig4_ablation_bars(renamed, paths["figures"] / "fig4_ablation_comparison.png")

    # figure 5: dose-response (optional)
    dose = paths["results"] / "spatial_destruction_dose_response.json"
    if dose.exists():
        r_per_q = json.load(open(dose))
        qs = sorted(float(k) for k in r_per_q)
        rs = [r_per_q[str(q)] if str(q) in r_per_q else r_per_q[q] for q in qs]
        figures.fig5_destruction_dose_response(qs, rs, rs, rs, paths["figures"] / "fig5_dose_response.png")

    # figure 7: efficiency vs performance (optional)
    eff = paths["results"] / "model_efficiency.json"
    if eff.exists():
        e = json.load(open(eff))
        params = [e["student"]["total_parameters"]]
        a2_rows = valid[valid["model"] == "A2_distilled_patch"]
        perf = [a2_rows["hvg1000_r_mean"].values[0]] if len(a2_rows) else [0]
        labels = ["distilled student"]
        # H-Optimus-0 teacher point intentionally not added here -- its performance lives in a
        # separate PCA256-based comparison (src/epathst/evaluation/hoptimus_comparison.py), not in cv_ablation_metrics.csv.
        figures.fig7_efficiency_vs_performance(
            params, perf, labels, paths["figures"] / "fig7_efficiency.png"
        )

    print(f"[17_results_and_figures] wrote tables and figures under {paths['results']} / {paths['figures']}")


if __name__ == "__main__":
    main()
