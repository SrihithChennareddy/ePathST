#!/usr/bin/env python
"""Steps 20-21 (Sections 24-25): aggregate per-patient Fisher-z Pearson from
results/cv_ablation_metrics.csv into the required results tables, then run
the PRIMARY hypothesis test: real multiscale graph (A7) vs degree-preserving
shuffled graph (A5), patient-level bootstrap (10,000 reps), one-sided."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts.stats_tests import paired_sign_flip_test, patient_bootstrap_ci
from sts.utils import load_config, resolve_run_paths, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    df = pd.read_csv(paths["results"] / "cv_ablation_metrics.csv")
    stats_cfg = cfg["statistics"]

    def patient_scores(ablation: str) -> dict:
        sub = df[df["ablation"] == ablation]
        return sub.groupby("cancer")["fisher_z_mean_patient_R"].mean().to_dict()

    r_real = patient_scores("A7_multiscale_pna")
    r_shuffle = patient_scores("A5_shuffled_graph")
    common_cancers = sorted(set(r_real) & set(r_shuffle))
    deltas = np.array([r_real[c] - r_shuffle[c] for c in common_cancers])

    bootstrap = patient_bootstrap_ci(deltas, n_iterations=stats_cfg["n_bootstrap"], ci=tuple(stats_cfg["ci"]))
    sign_flip = paired_sign_flip_test(deltas, n_iterations=stats_cfg["n_bootstrap"])
    result = {
        "comparison": "A7_multiscale_pna vs A5_shuffled_graph (PRIMARY HYPOTHESIS)",
        "units": common_cancers,
        "deltas": deltas.tolist(),
        "bootstrap": bootstrap,
        "sign_flip_test": sign_flip,
        "h1_supported_ci_above_zero": bool(bootstrap["ci_low"] > 0.0),
    }
    save_json(result, paths["results"] / "primary_hypothesis_test.json")
    print(f"[12_primary_hypothesis_test] mean delta R = {bootstrap['mean_delta']:.4f}, "
          f"95% CI [{bootstrap['ci_low']:.4f}, {bootstrap['ci_high']:.4f}] -> "
          f"H1 {'SUPPORTED' if result['h1_supported_ci_above_zero'] else 'NOT SUPPORTED'}")

    # Section 21 secondary comparisons, same bootstrap machinery
    secondary = {}
    for name, (a, b) in {
        "multiscale_vs_local": ("A7_multiscale_pna", "A6_local_pna"),
        "real_vs_random": ("A7_multiscale_pna", "A4_random_graph"),
        "distilled_vs_imagenet_patch": ("A2_distilled_patch", "A1_imagenet_patch"),
        "graph_vs_matched_mlp": ("A7_multiscale_pna", "A3_matched_mlp"),
    }.items():
        ra, rb = patient_scores(a), patient_scores(b)
        common = sorted(set(ra) & set(rb))
        if not common:
            continue
        d = np.array([ra[c] - rb[c] for c in common])
        secondary[name] = patient_bootstrap_ci(d, n_iterations=stats_cfg["n_bootstrap"], ci=tuple(stats_cfg["ci"]))
    save_json(secondary, paths["results"] / "secondary_hypothesis_tests.json")
    print(f"[12_primary_hypothesis_test] wrote secondary_hypothesis_tests.json ({list(secondary)})")


if __name__ == "__main__":
    main()
