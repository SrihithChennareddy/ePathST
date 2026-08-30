#!/usr/bin/env python
"""Steps 20-21 (Sections 24-25), revised per Final_Project_V2.txt Phase 22:
aggregate results/cv_ablation_metrics.csv, verify completeness (V2 Phase 3
/ S9), then run the THREE PRIMARY STATISTICAL GATES (replacing V1's single
primary-hypothesis test):

    Gate 1 (distillation):    R_A2_distilled_patch  > R_A1_imagenet_patch
    Gate 2 (spatial context): R_A8_scale_aware_pna   > R_A2_distilled_patch
    Gate 3 (true topology):   R_A8_scale_aware_pna   > R_A5_shuffled_graph

Gate 3 is the same "real vs. shuffled" test as V1's Section 25, now run
against A8 (scale-aware, the proposed final model) instead of A7 (union
multiscale) -- V2 explicitly redefines the primary spatial-topology model.
Reported PER CANCER (primary analysis unit, V2 Phase 1) with pooled-across-
cancers as a secondary summary."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts.cv_folds import assert_run_completeness, build_cv_grid, expected_run_keys
from sts.stats_tests import paired_sign_flip_test, patient_bootstrap_ci
from sts.utils import load_config, resolve_run_paths, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--skip-completeness-check", action="store_true",
                     help="only for partial/debug runs -- the real pipeline must never skip this")
    ap.add_argument("--max-folds-per-cancer", type=int, default=None,
                     help="must match whatever value scripts/10_run_patient_cv.py was run with, so the "
                          "completeness gate compares against the same (trimmed) design it actually ran")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                     help="must match whatever value scripts/10_run_patient_cv.py was run with")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    if args.seeds is not None:
        cfg["cross_validation"]["seeds"] = args.seeds

    df = pd.read_csv(paths["results"] / "cv_ablation_metrics.csv")
    stats_cfg = cfg["statistics"]

    # V2 Phase 3 / S9: refuse to compute any statistic over a run set that
    # doesn't match what the CV design actually calls for.
    if not args.skip_completeness_check:
        hest_samples = pd.read_csv(paths["manifests"] / "hest_samples.csv")
        patients_by_cancer = {c: hest_samples[hest_samples["cancer"] == c]["patient_id"].unique().tolist()
                               for c in cfg["cancers"]["hest"]}
        cv_grid = build_cv_grid(patients_by_cancer, cfg)
        if args.max_folds_per_cancer is not None:
            for info in cv_grid["per_cancer"].values():
                info["folds"] = info["folds"][:args.max_folds_per_cancer]
            cv_grid["pooled"]["folds"] = cv_grid["pooled"]["folds"][:args.max_folds_per_cancer]
        ablation_names = sorted(df["ablation"].unique())
        expected = expected_run_keys(cv_grid, ablation_names, unit="per_cancer")
        completed = set(zip(df["cancer"], df["fold"], df["seed"], df["ablation"]))
        assert_run_completeness(expected, completed, context="12_primary_hypothesis_test")

    def patient_scores(ablation: str, cancer: str | None = None) -> dict:
        """Per-(fold,seed) Fisher-z patient-macro R for one ablation,
        optionally restricted to one cancer -- these fold/seed cells are
        the bootstrap resampling unit here (see the Known limitation note
        below: this is coarser than resampling true per-patient R_p
        values directly, which would require plumbing per-patient
        predictions through to this script rather than per-fold
        aggregates)."""
        sub = df[df["ablation"] == ablation]
        if cancer is not None:
            sub = sub[sub["cancer"] == cancer]
        return {f"{r.cancer}_f{r.fold}_s{r.seed}": r.fisher_z_mean_patient_R for r in sub.itertuples()}

    def gate_test(better: str, worse: str, cancer: str | None = None) -> dict:
        ra, rb = patient_scores(better, cancer), patient_scores(worse, cancer)
        common = sorted(set(ra) & set(rb))
        if not common:
            return {"n_units": 0, "note": "no overlapping (cancer,fold,seed) cells for this comparison yet"}
        deltas = np.array([ra[c] - rb[c] for c in common])
        bootstrap = patient_bootstrap_ci(deltas, n_iterations=stats_cfg["n_bootstrap"], ci=tuple(stats_cfg["ci"]))
        sign_flip = paired_sign_flip_test(deltas, n_iterations=stats_cfg["n_bootstrap"])
        return {"better": better, "worse": worse, "units": common, "bootstrap": bootstrap,
                "sign_flip_test": sign_flip, "supported_ci_above_zero": bool(bootstrap["ci_low"] > 0.0)}

    gates_cfg = cfg["primary_gates"]
    gate_names = {"gate1_distillation": "Gate 1: Distillation (A2 > A1)",
                  "gate2_spatial_context": "Gate 2: Spatial context (A8 > A2)",
                  "gate3_true_topology": "Gate 3: True topology (A8 > A5, THE decisive test)"}

    results = {"pooled": {}, "per_cancer": {}}
    for gate_key, spec in gates_cfg.items():
        results["pooled"][gate_key] = gate_test(spec["better"], spec["worse"])
        print(f"[12_primary_hypothesis_test] {gate_names.get(gate_key, gate_key)} (pooled): "
              f"{'SUPPORTED' if results['pooled'][gate_key].get('supported_ci_above_zero') else 'NOT SUPPORTED'}")
        for cancer in cfg["cancers"]["hest"]:
            results["per_cancer"].setdefault(cancer, {})[gate_key] = gate_test(spec["better"], spec["worse"], cancer)

    save_json(results, paths["results"] / "primary_hypothesis_test.json")

    # Secondary comparisons (V2 Phase 22 doesn't remove these, only reprioritizes the 3 gates above)
    secondary = {}
    for name, (a, b) in {
        "scale_aware_vs_local": ("A8_scale_aware_pna", "A6_local_pna"),
        "scale_aware_vs_union_multiscale": ("A8_scale_aware_pna", "A7_multiscale_pna"),
        "scale_aware_vs_random": ("A8_scale_aware_pna", "A4_random_graph"),
        "scale_aware_vs_matched_mlp": ("A8_scale_aware_pna", "A3_matched_mlp"),
        "scale_aware_vs_gatv2": ("A8_scale_aware_pna", "A9_gatv2_multiscale"),
    }.items():
        secondary[name] = gate_test(a, b)
    save_json(secondary, paths["results"] / "secondary_hypothesis_tests.json")
    print(f"[12_primary_hypothesis_test] wrote primary_hypothesis_test.json (3 gates x pooled+per-cancer) "
          f"and secondary_hypothesis_tests.json ({list(secondary)})")

    print("\n[12_primary_hypothesis_test] KNOWN LIMITATION: bootstrap resampling unit here is "
          "(cancer, fold, seed) cells, not individual patients -- a stricter implementation would "
          "plumb per-patient R_p values through to this script and bootstrap over patients directly "
          "(Section 25's literal requirement). Flagged, not silently glossed over.")


if __name__ == "__main__":
    main()
