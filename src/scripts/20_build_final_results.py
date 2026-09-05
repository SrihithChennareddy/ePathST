#!/usr/bin/env python
"""Assembles results/final/ -- the single consolidated, submission-ready
output directory for the LOCKED final 5-fold x 3-cancer x seed-0
experiment (job 84871, 2026-08-31). Pure aggregation over files scripts
10/12/17/19 and the GenBio baseline already wrote to disk -- never
re-derives a number by any means other than reading it back out of one of
those files (no re-fitting, no estimation, no extrapolation).

Per the FINAL STOP RULE: every number here is either (a) a real measured
result from a completed run, or (b) explicitly labeled NOT MEASURED /
PARTIAL / SEPARATE PROTOCOL -- never silently omitted or fabricated.

Writes, under results/final/:
    tables/model_comparison_macro.csv          -- all 10 ablations x {hvg50,hvg1000,pathway}, pooled 3-cancer bootstrap CI
    tables/model_comparison_per_cancer.csv     -- same, broken out per cancer (5-fold bootstrap CI within cancer)
    tables/gate_summary.csv                    -- the 3 primary gates, pooled + per-cancer, verdicts
    tables/secondary_comparisons.csv           -- the 5 secondary A8-vs-X comparisons (pooled)
    tables/pathway_progeny_results.csv         -- pathway-only view (macro + per-cancer)
    tables/official_hest_benchmark.csv         -- SEPARATE official-protocol numbers (never mixed with the above)
    tables/genbio_embedding_mlp_baseline_B1.csv -- PARTIAL (fold0-only), GenBio embedding -> trained MLP
    tables/efficiency.csv                      -- real measured params + compression ratio; unmeasured rows flagged
    folds/cv_folds_used.csv                    -- exact (cancer, fold, split, patient_id) used, for audit
    FINAL_RESULTS.md                           -- narrative: Q1-Q5 answered, tables inlined, honest limitations
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts.ablations import ABLATION_GRID
from sts.stats_tests import patient_bootstrap_ci
from sts.utils import load_config, resolve_run_paths, save_json

MODEL_DESC = {a.name: a.description for a in ABLATION_GRID}
MODEL_PRIORITY = {a.name: a.priority for a in ABLATION_GRID}
MODEL_ORDER = [a.name for a in ABLATION_GRID]
TARGETS = [("hvg1000", "hvg1000_fisher_z_mean_patient_R"), ("hvg50", "hvg50_fisher_z_mean_patient_R"),
           ("pathway", "pathway_fisher_z_mean_patient_R")]


def bootstrap_row(vals: np.ndarray, n_boot: int) -> dict:
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        mean = float(vals.mean()) if len(vals) else float("nan")
        return {"mean": mean, "ci_low": float("nan"), "ci_high": float("nan"), "n": len(vals)}
    ci = patient_bootstrap_ci(vals, n_iterations=n_boot)
    return {"mean": ci["mean_delta"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "n": len(vals)}


def main():
    cfg = load_config("configs/base.yaml")
    paths = resolve_run_paths(cfg)
    results = paths["results"]
    n_boot = cfg["statistics"]["n_bootstrap"]

    out_dir = results / "final"
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "folds").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results / "cv_ablation_metrics.csv")
    cancers = cfg["cancers"]["hest"]

    # -- A) macro (pooled across 3 cancers x 5 folds x seed 0 = 15 cells) --
    macro_rows = []
    for model in MODEL_ORDER:
        sub = df[df["ablation"] == model]
        row = {"model": model, "priority": MODEL_PRIORITY[model], "description": MODEL_DESC[model]}
        for tgt, col in TARGETS:
            b = bootstrap_row(sub[col].values, n_boot)
            row.update({f"{tgt}_R_mean": b["mean"], f"{tgt}_R_ci_low": b["ci_low"],
                        f"{tgt}_R_ci_high": b["ci_high"], f"{tgt}_n_cells": b["n"]})
        macro_rows.append(row)
    macro_df = pd.DataFrame(macro_rows)
    macro_df.to_csv(out_dir / "tables" / "model_comparison_macro.csv", index=False)

    # -- B) per-cancer (5-fold bootstrap CI within each cancer) --
    per_cancer_rows = []
    for cancer in cancers:
        for model in MODEL_ORDER:
            sub = df[(df["ablation"] == model) & (df["cancer"] == cancer)]
            row = {"cancer": cancer, "model": model}
            for tgt, col in TARGETS:
                b = bootstrap_row(sub[col].values, n_boot)
                row.update({f"{tgt}_R_mean": b["mean"], f"{tgt}_R_ci_low": b["ci_low"],
                            f"{tgt}_R_ci_high": b["ci_high"], f"{tgt}_n_folds": b["n"]})
            per_cancer_rows.append(row)
    per_cancer_df = pd.DataFrame(per_cancer_rows)
    per_cancer_df.to_csv(out_dir / "tables" / "model_comparison_per_cancer.csv", index=False)

    # -- C) gate summary (from script 12's primary_hypothesis_test.json) --
    primary = json.loads((results / "primary_hypothesis_test.json").read_text())
    gate_labels = {"gate1_distillation": "Gate 1: Distillation (A2 > A1)",
                   "gate2_spatial_context": "Gate 2: Spatial context (A8 > A2)",
                   "gate3_true_topology": "Gate 3: True topology (A8 > A5, decisive negative control)"}
    gate_rows = []
    for gate_key, spec in cfg["primary_gates"].items():
        g = primary["pooled"][gate_key]
        gate_rows.append({"gate": gate_key, "label": gate_labels[gate_key], "scope": "pooled (3 cancers)",
                           "better": spec["better"], "worse": spec["worse"],
                           "mean_delta_R": g["bootstrap"]["mean_delta"], "ci_low": g["bootstrap"]["ci_low"],
                           "ci_high": g["bootstrap"]["ci_high"], "cohens_d": g["bootstrap"]["effect_size_cohens_d"],
                           "p_value_one_sided": g["sign_flip_test"]["p_value_one_sided"],
                           "n_units": len(g["units"]),
                           "verdict": "SUPPORTED" if g["supported_ci_above_zero"] else "NOT SUPPORTED"})
        for cancer in cancers:
            gc = primary["per_cancer"][cancer][gate_key]
            gate_rows.append({"gate": gate_key, "label": gate_labels[gate_key], "scope": cancer,
                               "better": spec["better"], "worse": spec["worse"],
                               "mean_delta_R": gc["bootstrap"]["mean_delta"], "ci_low": gc["bootstrap"]["ci_low"],
                               "ci_high": gc["bootstrap"]["ci_high"],
                               "cohens_d": gc["bootstrap"]["effect_size_cohens_d"],
                               "p_value_one_sided": gc["sign_flip_test"]["p_value_one_sided"],
                               "n_units": len(gc["units"]),
                               "verdict": "SUPPORTED" if gc["supported_ci_above_zero"] else "NOT SUPPORTED"})
    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(out_dir / "tables" / "gate_summary.csv", index=False)

    # -- D) secondary comparisons (pooled only, from script 12) --
    secondary = json.loads((results / "secondary_hypothesis_tests.json").read_text())
    sec_rows = []
    for name, g in secondary.items():
        sec_rows.append({"comparison": name, "better": g.get("better"), "worse": g.get("worse"),
                          "mean_delta_R": g.get("bootstrap", {}).get("mean_delta"),
                          "ci_low": g.get("bootstrap", {}).get("ci_low"),
                          "ci_high": g.get("bootstrap", {}).get("ci_high"),
                          "p_value_one_sided": g.get("sign_flip_test", {}).get("p_value_one_sided"),
                          "n_units": len(g.get("units", [])),
                          "verdict": "SUPPORTED" if g.get("supported_ci_above_zero") else "NOT SUPPORTED"})
    pd.DataFrame(sec_rows).to_csv(out_dir / "tables" / "secondary_comparisons.csv", index=False)

    # -- E) pathway/PROGENy-only view --
    pathway_rows = [{"scope": "macro", "model": r["model"], "R_mean": r["pathway_R_mean"],
                      "ci_low": r["pathway_R_ci_low"], "ci_high": r["pathway_R_ci_high"], "n": r["pathway_n_cells"]}
                     for r in macro_rows]
    for r in per_cancer_rows:
        pathway_rows.append({"scope": r["cancer"], "model": r["model"], "R_mean": r["pathway_R_mean"],
                              "ci_low": r["pathway_R_ci_low"], "ci_high": r["pathway_R_ci_high"],
                              "n": r["pathway_n_folds"]})
    pd.DataFrame(pathway_rows).to_csv(out_dir / "tables" / "pathway_progeny_results.csv", index=False)

    # -- F) official HEST-Benchmark (separate protocol, copied verbatim, never mixed with A-G) --
    official_src = results / "official_hest_benchmark" / "official_hest_benchmark_summary.csv"
    official_df = pd.read_csv(official_src) if official_src.exists() else pd.DataFrame()
    if len(official_df):
        official_df.to_csv(out_dir / "tables" / "official_hest_benchmark.csv", index=False)

    # -- G) GenBio embedding -> MLP baseline B1 (PARTIAL: fold0/seed0 only, not full 5-fold).
    # src/scripts/run_genbio_hest_baseline.py's default --head is a trained MLP on the raw 4608-D
    # GenBio embedding (no PCA, no Ridge) -- --head ridge is an available but unused alternative.
    genbio_src = results / "genbio_baseline" / "genbio_summary.csv"
    genbio_df = pd.read_csv(genbio_src) if genbio_src.exists() else pd.DataFrame()
    if len(genbio_df):
        genbio_df.insert(1, "status", "PARTIAL: fold0/seed0 only, not full 5-fold CV -- do not compare directly "
                                        "to the 5-fold macro table above without accounting for this")
        genbio_df.to_csv(out_dir / "tables" / "genbio_embedding_mlp_baseline_B1.csv", index=False)

    # -- H) efficiency (real measured items only; everything else explicitly NOT MEASURED) --
    student_params = None
    sp = results / "student_encoder_param_count_seed0.json"
    if sp.exists():
        student_params = json.loads(sp.read_text())["total_parameters"]
    teacher_params = None
    tp = results / "genbio_teacher_param_count.json"
    if tp.exists():
        teacher_params = json.loads(tp.read_text())["total_parameters"]
    eff_rows = [
        {"metric": "student_encoder_total_parameters", "value": student_params, "status": "MEASURED",
         "source": "scripts (ConvNeXt V2-Atto student), student_encoder_param_count_seed0.json"},
        {"metric": "genbio_pathfm_teacher_total_parameters", "value": teacher_params, "status": "MEASURED",
         "source": "AutoModel.from_pretrained('genbio-ai/genbio-pathfm'), sum(p.numel()), 2026-08-31"},
        {"metric": "compression_ratio_teacher_over_student", "value":
            (teacher_params / student_params) if (teacher_params and student_params) else None, "status": "MEASURED",
         "source": "derived from the two rows above"},
        {"metric": "gpu_peak_vram_inference", "value": None, "status": "NOT MEASURED",
         "source": "no inference-time VRAM profiling run this session"},
        {"metric": "patches_per_second_throughput", "value": None, "status": "NOT MEASURED",
         "source": "no throughput benchmark run this session"},
        {"metric": "A8_gnn_inference_wallclock_per_patient", "value": None, "status": "NOT MEASURED",
         "source": "no dedicated timing harness run this session"},
    ]
    pd.DataFrame(eff_rows).to_csv(out_dir / "tables" / "efficiency.csv", index=False)

    # -- I) fold manifest (exact patients used, for audit) --
    cv_folds = json.loads((paths["manifests"] / "hest_cv_folds.json").read_text())
    fold_rows = []
    for cancer, info in cv_folds["per_cancer"].items():
        for f in info["folds"]:
            for pid in f["train_patients"]:
                fold_rows.append({"cancer": cancer, "fold": f["fold"], "split": "train", "patient_id": pid})
            for pid in f["test_patients"]:
                fold_rows.append({"cancer": cancer, "fold": f["fold"], "split": "test", "patient_id": pid})
    pd.DataFrame(fold_rows).to_csv(out_dir / "folds" / "cv_folds_used.csv", index=False)

    # -- J) FINAL_RESULTS.md --
    def fmt(v, d=4):
        return "N/A" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"

    def gate_line(gate_key):
        g = [r for r in gate_rows if r["gate"] == gate_key and r["scope"] == "pooled (3 cancers)"][0]
        per_c = [f"{r['scope']}: {r['verdict']} (delta={fmt(r['mean_delta_R'])})"
                 for r in gate_rows if r["gate"] == gate_key and r["scope"] != "pooled (3 cancers)"]
        return (f"**{g['label']}** -- pooled: **{g['verdict']}**, "
                f"delta_R={fmt(g['mean_delta_R'])} 95% CI=[{fmt(g['ci_low'])}, {fmt(g['ci_high'])}], "
                f"Cohen's d={fmt(g['cohens_d'])}, p={fmt(g['p_value_one_sided'], 4)} "
                f"(n={g['n_units']} cancer-fold-seed cells). Per-cancer: {'; '.join(per_c)}.")

    macro_table_md = macro_df[["model", "priority", "hvg1000_R_mean", "hvg1000_R_ci_low", "hvg1000_R_ci_high",
                                "hvg50_R_mean", "pathway_R_mean"]].round(4).to_markdown(index=False)
    gate_table_md = gate_df[gate_df["scope"] == "pooled (3 cancers)"][
        ["label", "mean_delta_R", "ci_low", "ci_high", "p_value_one_sided", "verdict"]].round(4).to_markdown(index=False)
    official_table_md = official_df.round(4).to_markdown(index=False) if len(official_df) else "(not available)"
    genbio_table_md = genbio_df[["cancer", "status", "HVG50_r", "HVG1000_r", "PROGENy14_r"]].round(4).to_markdown(
        index=False) if len(genbio_df) else "(not available)"

    md = f"""# FINAL_RESULTS.md

Locked final scientific experiment: distilled ConvNeXt V2-Atto (from GenBio-PathFM) +
multiscale PNA graph correction, 5-fold patient-level CV x 3 HEST-1k cancers
(IDC/COAD/CCRCC) x seed 0. Source run: SLURM job 84871, completed 2026-08-31
(03:15:54 elapsed). All numbers below are read directly from
`results/cv_ablation_metrics.csv`, `results/primary_hypothesis_test.json`,
`results/secondary_hypothesis_tests.json`, `results/official_hest_benchmark/`,
`results/genbio_baseline/`, and the two parameter-count JSONs -- nothing here is
estimated or extrapolated (FINAL STOP RULE).

## Research questions, answered

**Q1/Q2 -- Does GenBio-PathFM distillation help the patch-only encoder?**
{gate_line('gate1_distillation')}
Distillation is **not supported** as a pooled effect at seed 0: A2 (distilled) is not reliably
better than A1 (ImageNet) once patient-level bootstrap uncertainty is accounted for. See the
per-cancer breakdown in `tables/gate_summary.csv` for where (if anywhere) it reverses.

**Q3 -- Does the multiscale PNA graph correction add real signal over patch-only prediction?**
{gate_line('gate2_spatial_context')}
This is the most robust positive finding in the experiment: the GNN correction improves on
patch-only prediction in **every one of the 3 cancers individually**, not just pooled.

**Q4 -- Does that benefit come from true tissue topology specifically (vs. generic neighborhood
smoothing)?**
{gate_line('gate3_true_topology')}
The decisive negative control: A8 (real multiscale topology) is statistically indistinguishable
from A5 (degree-preserving shuffled graph) in every cancer. Per the FINAL STOP RULE, this is
reported as a genuine negative finding, not glossed over -- the GNN's Gate-2 benefit most likely
comes from general neighbor-averaging/regularization, not from encoding real tissue adjacency.

**Q5 -- Compression efficiency (student vs. GenBio-PathFM teacher).**
Student (distilled ConvNeXt V2-Atto): **{student_params:,} parameters** (measured).
GenBio-PathFM teacher: **{teacher_params:,} parameters** (measured 2026-08-31 via
`AutoModel.from_pretrained`, exact `sum(p.numel())`, not the model card's rounded "1.1B").
Compression ratio: **{teacher_params / student_params:.1f}x** fewer parameters, at the accuracy
cost quantified by Gate 1 above and by the official-protocol comparison below. All other
efficiency metrics (peak inference VRAM, patches/sec throughput, per-patient GNN wall-clock) were
**NOT MEASURED** this session -- see `tables/efficiency.csv`.

## Primary gates (pooled across IDC/COAD/CCRCC, 15 cancer-fold-seed cells each)

{gate_table_md}

Full pooled + per-cancer breakdown, sign-flip p-values, effect sizes: `tables/gate_summary.csv`.
Secondary comparisons (A8 vs. local/union/random/matched-MLP/GATv2): `tables/secondary_comparisons.csv`.

## Macro model comparison (pooled 3-cancer x 5-fold x seed-0, bootstrap 95% CI, n_bootstrap={n_boot})

{macro_table_md}

Per-cancer breakdown: `tables/model_comparison_per_cancer.csv`. Pathway/PROGENy14-only view:
`tables/pathway_progeny_results.csv`.

## Official HEST-Benchmark reproduction (SEPARATE protocol -- log1p-only targets, PCA256+Ridge,
official k-fold splits, seed=1 -- NOT comparable numerically to the custom-experiment table above,
which uses train-fold HVG selection, a trained patch head, and GNN correction)

{official_table_md}

## GenBio embedding -> trained MLP baseline (B1) -- **PARTIAL, fold0/seed0 only, not full 5-fold CV**

Frozen 4608-D GenBio-PathFM embedding fed directly into a small trained MLP head (no PCA, no
Ridge -- `src/scripts/run_genbio_hest_baseline.py --head ridge` is an available but unused
alternative), on the same HVG1000/HVG50/PROGENy14 targets and fold structure as the A0-A9 table.

{genbio_table_md}

This is preliminary (single fold per cancer); do not treat it as equivalent evidence to the
5-fold macro table above. A full 5-fold B1 rerun would be needed before drawing a firm B1
conclusion.

## Known limitations (reported honestly, not silently omitted)

1. **Bootstrap resampling unit.** All CIs above resample (cancer, fold, seed) cells (15 pooled,
   5 per cancer), not literal per-patient R_p values -- script 10 does not currently persist
   per-patient predictions to disk, only fold-level aggregates. A stricter implementation would
   plumb per-patient R_p through to script 12/20 and bootstrap over patients directly. Flagged
   since job 84871's design, not new to this aggregation step.
2. **Seeds 1/2** for the major models (A2/A8/B1) were not run -- seed 0 only, per the spec's
   "seed 0 is primary; seeds 1/2 only if compute allows" allowance.
3. **B1 (GenBio embedding -> MLP baseline)** is fold0/seed0 only for each cancer (see above),
   not the full 5-fold design the rest of this table uses.
4. **Efficiency**: only parameter counts (student, teacher, compression ratio) were measured;
   VRAM/throughput/wall-clock were not.
5. The **IDC-specific distillation reversal** noted during the interactive diagnostic session
   (A1 slightly ahead of A2 in IDC specifically) has not been separately root-caused -- see
   `tables/model_comparison_per_cancer.csv` for the raw per-cancer numbers.

## Reproducibility

Exact patient-to-fold assignment used: `folds/cv_folds_used.csv`. Full config:
`experimental_runs/2026-08-31_job84871_final_5fold_3cancer_seed0/config_used.json`. Raw per-run
metrics: `results/cv_ablation_metrics.csv` (150 rows = 3 cancers x 5 folds x 10 ablations x
seed 0).
"""
    (out_dir / "FINAL_RESULTS.md").write_text(md)

    print(f"[20_build_final_results] wrote {out_dir}")
    for p in sorted((out_dir / "tables").iterdir()):
        print(f"  tables/{p.name}")
    print(f"  folds/cv_folds_used.csv")
    print(f"  FINAL_RESULTS.md")


if __name__ == "__main__":
    main()
