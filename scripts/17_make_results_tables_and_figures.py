#!/usr/bin/env python
"""Steps 26-27 (Sections 30-31): the final MODEL x {HVG50, HVG1000,
PATHWAY} results table (mean +/- 95% CI at the patient level), the
per-cancer breakdown table, and figures 1/4/5/7 (the figures directly
derivable from tabular results already on disk; 2/3/6/8 are produced
inline by scripts 02/06/13 since they need raw images/embeddings)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts import figures
from sts.stats_tests import patient_bootstrap_ci
from sts.utils import load_config, resolve_run_paths, save_json

MODEL_ORDER = ["A0_mean_expression", "A1_imagenet_patch", "A2_distilled_patch", "A3_matched_mlp",
               "A4_random_graph", "A5_shuffled_graph", "A6_local_pna", "A7_multiscale_pna",
               "A8_genbio_patch", "A9_genbio_pna"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    df = pd.read_csv(paths["results"] / "cv_ablation_metrics.csv")

    rows = []
    for model in MODEL_ORDER:
        sub = df[df["ablation"] == model]
        if sub.empty:
            rows.append({"model": model, "hvg50_r_mean": None, "hvg50_r_ci": None})
            continue
        vals = sub["fisher_z_mean_patient_R"].dropna().values
        ci = patient_bootstrap_ci(vals, n_iterations=cfg["statistics"]["n_bootstrap"]) if len(vals) > 1 else \
            {"mean_delta": float(vals.mean()) if len(vals) else np.nan, "ci_low": np.nan, "ci_high": np.nan}
        rows.append({"model": model, "hvg50_r_mean": ci.get("mean_delta", ci.get("mean")),
                     "hvg50_r_ci_low": ci["ci_low"], "hvg50_r_ci_high": ci["ci_high"], "n_folds_seeds": len(vals)})
    table = pd.DataFrame(rows)
    table.to_csv(paths["results"] / "final_results_table.csv", index=False)
    print(f"[17_results_and_figures] final results table:\n{table.to_string(index=False)}")

    per_cancer_rows = []
    for cancer in cfg["cancers"]["hest"]:
        for model in ["A2_distilled_patch", "A7_multiscale_pna", "A8_genbio_patch", "A9_genbio_pna"]:
            sub = df[(df["ablation"] == model) & (df["cancer"] == cancer)]
            per_cancer_rows.append({"cancer": cancer, "model": model,
                                     "mean_R": sub["fisher_z_mean_patient_R"].mean() if len(sub) else None})
    pd.DataFrame(per_cancer_rows).to_csv(paths["results"] / "per_cancer_results_table.csv", index=False)

    # Figures directly derivable from `table`
    valid = table.dropna(subset=["hvg50_r_mean"])
    figures.fig1_pipeline_diagram(paths["figures"] / "fig1_pipeline.png")
    if len(valid):
        figures.fig4_ablation_bars(valid.rename(columns={"hvg50_r_mean": "fisher_z_mean_patient_R"}),
                                    paths["figures"] / "fig4_ablation_comparison.png")

    dose = paths["results"] / "spatial_destruction_dose_response.json"
    if dose.exists():
        import json
        r_per_q = json.load(open(dose))
        qs = sorted(float(k) for k in r_per_q)
        rs = [r_per_q[str(q)] if str(q) in r_per_q else r_per_q[q] for q in qs]
        figures.fig5_destruction_dose_response(qs, rs, rs, rs, paths["figures"] / "fig5_dose_response.png")

    eff = paths["results"] / "model_efficiency.json"
    if eff.exists():
        import json
        e = json.load(open(eff))
        params = [e["student"]["total_parameters"]]
        perf = [valid[valid["model"] == "A2_distilled_patch"]["hvg50_r_mean"].values[0]] if len(
            valid[valid["model"] == "A2_distilled_patch"]) else [0]
        labels = ["distilled student"]
        if e["teacher"].get("total_parameters"):
            params.append(e["teacher"]["total_parameters"])
            genbio_perf = valid[valid["model"] == "A8_genbio_patch"]["hvg50_r_mean"].values
            perf.append(genbio_perf[0] if len(genbio_perf) else 0)
            labels.append("GenBio teacher")
        figures.fig7_efficiency_vs_performance(params, perf, labels, paths["figures"] / "fig7_efficiency.png")

    print(f"[17_results_and_figures] wrote tables and figures under {paths['results']} / {paths['figures']}")


if __name__ == "__main__":
    main()
