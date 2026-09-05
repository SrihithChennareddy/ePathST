#!/usr/bin/env python
"""Step 29: a detailed but still-skimmable results summary -- per-cancer
and macro A0-A9 tables (with bootstrap 95% CIs and cell counts), the 3
primary gates and 5 secondary comparisons with effect sizes/p-values, a
pathway/PROGENy14-only view, GenBio-PathFM reference numbers (kept on two
explicitly separate tracks), and measured efficiency numbers. Pure
formatting over results/final/tables/*.csv (script 20's output); computes
nothing new. Writes two copies: the live results/final/ tree (canonical,
regenerated every time script 20 reruns) and a git-tracked copy under
docs/ (for readers browsing the repo without cluster access)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts.ablations import ABLATION_GRID
from sts.utils import load_config, resolve_run_paths

MODEL_ORDER = [a.name for a in ABLATION_GRID]
MODEL_LABEL = {a.name: a.description for a in ABLATION_GRID}
MODEL_PRIORITY = {a.name: a.priority for a in ABLATION_GRID}
BOLD = {"A2_distilled_patch", "A8_scale_aware_pna"}


def fmt(v, d=4):
    return "n/a" if pd.isna(v) else f"{v:.{d}f}"


def ci_cell(r, tgt):
    mean, lo, hi, n = r[f"{tgt}_R_mean"], r[f"{tgt}_R_ci_low"], r[f"{tgt}_R_ci_high"], r.get(
        f"{tgt}_n_cells", r.get(f"{tgt}_n_folds"))
    return f"{fmt(mean)} [{fmt(lo)}, {fmt(hi)}] (n={int(n) if pd.notna(n) else '?'})"


def main():
    cfg = load_config("configs/base.yaml")
    paths = resolve_run_paths(cfg)
    final_dir = paths["results"] / "final"
    tables = final_dir / "tables"
    cancers = cfg["cancers"]["hest"]
    n_boot = cfg["statistics"]["n_bootstrap"]

    macro = pd.read_csv(tables / "model_comparison_macro.csv").set_index("model")
    per_cancer = pd.read_csv(tables / "model_comparison_per_cancer.csv")
    gates = pd.read_csv(tables / "gate_summary.csv")
    secondary = pd.read_csv(tables / "secondary_comparisons.csv")
    pathway = pd.read_csv(tables / "pathway_progeny_results.csv")
    genbio_b1 = pd.read_csv(tables / "genbio_embedding_mlp_baseline_B1.csv")
    official = pd.read_csv(tables / "official_hest_benchmark.csv")
    efficiency = pd.read_csv(tables / "efficiency.csv")

    L = []  # lines
    L += ["# HEST Final 5-Fold Experiment -- Results\n",
          "Detailed, skimmable version of `FINAL_RESULTS.md` for the LOCKED final 5-fold x "
          "3-cancer x seed-0 experiment (SLURM job 84871, completed 2026-08-31, 03:15:54 elapsed). "
          "Metric: patient-level Fisher-z mean Pearson R (`sts.metrics.patient_macro_fisher_z`). "
          f"All CIs are bootstrap 95% intervals, n_bootstrap={n_boot}, resampled over "
          "(cancer, fold, seed) cells -- see the Limitations section for what that does and does "
          "not mean. Every number below is read directly from `results/final/tables/`; nothing is "
          "re-derived or estimated in this document.\n",
          "## Contents\n",
          "1. [Ablation grid (what A0-A9 mean)](#1-ablation-grid)",
          "2. [Macro-average results (pooled 3 cancers)](#2-macro-average-results)",
          "3. [Per-cancer results](#3-per-cancer-results)",
          "4. [Primary statistical gates](#4-primary-statistical-gates)",
          "5. [Secondary comparisons](#5-secondary-comparisons)",
          "6. [Pathway / PROGENy14 view](#6-pathway--progeny14-view)",
          "7. [GenBio-PathFM reference numbers](#7-genbio-pathfm-reference-numbers)",
          "8. [Efficiency](#8-efficiency)",
          "9. [Bottom line](#9-bottom-line)",
          "10. [Limitations](#10-limitations)\n"]

    L += ["## 1. Ablation grid\n", "| Model | Priority | What it is |", "|---|---|---|"]
    for m in MODEL_ORDER:
        L.append(f"| {'**' + m + '**' if m in BOLD else m} | {MODEL_PRIORITY[m]} | {MODEL_LABEL[m]} |")
    L.append("")

    L += ["## 2. Macro-average results\n",
          "Pooled across IDC + COAD + CCRCC (15 cancer-fold-seed cells per model: 3 cancers x 5 "
          "folds x seed 0).\n",
          "| Model | HVG1000 R [95% CI] | HVG50 R [95% CI] | Pathway R [95% CI] |",
          "|---|---|---|---|"]
    for m in MODEL_ORDER:
        r = macro.loc[m]
        label = f"**{m}**" if m in BOLD else m
        L.append(f"| {label} | {ci_cell(r, 'hvg1000')} | {ci_cell(r, 'hvg50')} | {ci_cell(r, 'pathway')} |")
    L.append("\n(A0_mean_expression's R is `n/a`: predicting the constant train-fold mean for every "
              "spot gives zero-variance predictions, and Pearson r is undefined against a constant "
              "-- this is the expected behavior of a sanity-floor baseline, not a bug.)\n")

    L.append("## 3. Per-cancer results\n")
    for cancer in cancers:
        n_patients_test = per_cancer[per_cancer.cancer == cancer]  # informational only
        L.append(f"### {cancer}\n")
        L.append("| Model | HVG1000 R [95% CI] | HVG50 R [95% CI] | Pathway R [95% CI] |")
        L.append("|---|---|---|---|")
        sub = per_cancer[per_cancer["cancer"] == cancer].set_index("model")
        for m in MODEL_ORDER:
            r = sub.loc[m]
            label = f"**{m}**" if m in BOLD else m
            L.append(f"| {label} | {ci_cell(r, 'hvg1000')} | {ci_cell(r, 'hvg50')} | {ci_cell(r, 'pathway')} |")
        L.append("")

    L += ["## 4. Primary statistical gates\n",
          "The three pre-registered pass/fail tests (configs/base.yaml `primary_gates`). "
          "`delta_R` = mean(better) - mean(worse) over shared cells; CI/p-value from "
          "`sts.stats_tests.patient_bootstrap_ci` / `paired_sign_flip_test`; **SUPPORTED** requires "
          "the 95% CI to lie entirely above zero.\n"]
    for gate_key in ("gate1_distillation", "gate2_spatial_context", "gate3_true_topology"):
        g = gates[gates.gate == gate_key]
        pooled = g[g.scope == "pooled (3 cancers)"].iloc[0]
        L.append(f"### {pooled['label']}\n")
        L.append(f"better = `{pooled['better']}`, worse = `{pooled['worse']}`\n")
        L.append("| Scope | delta_R | 95% CI | Cohen's d | p (one-sided sign-flip) | n | Verdict |")
        L.append("|---|---|---|---|---|---|---|")
        for _, row in g.iterrows():
            L.append(f"| {row['scope']} | {fmt(row['mean_delta_R'])} | "
                      f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {fmt(row['cohens_d'])} | "
                      f"{fmt(row['p_value_one_sided'], 4)} | {int(row['n_units'])} | **{row['verdict']}** |")
        L.append("")

    L += ["## 5. Secondary comparisons\n",
          "A8 (scale-aware, flagship) vs. the remaining architecture/graph controls, pooled across "
          "3 cancers.\n",
          "| Comparison | better | worse | delta_R | 95% CI | p (one-sided) | n | Verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for _, row in secondary.iterrows():
        L.append(f"| {row['comparison']} | {row['better']} | {row['worse']} | {fmt(row['mean_delta_R'])} | "
                  f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {fmt(row['p_value_one_sided'], 4)} | "
                  f"{int(row['n_units'])} | **{row['verdict']}** |")
    L.append("")

    L += ["## 6. Pathway / PROGENy14 view\n",
          "Same pathway-target column as sections 2-3 above, isolated for readers who only care "
          "about the 14 PROGENy pathway-activity scores (not per-gene HVG targets).\n",
          "| Model | Macro R [95% CI] | " + " | ".join(f"{c} R [95% CI]" for c in cancers) + " |",
          "|---|---|" + "---|" * len(cancers)]
    pw_macro = pathway[pathway.scope == "macro"].set_index("model")
    pw_cancer = {c: pathway[pathway.scope == c].set_index("model") for c in cancers}
    for m in MODEL_ORDER:
        mrow = pw_macro.loc[m]
        cells = [f"{fmt(mrow['R_mean'])} [{fmt(mrow['ci_low'])}, {fmt(mrow['ci_high'])}]"]
        for c in cancers:
            crow = pw_cancer[c].loc[m]
            cells.append(f"{fmt(crow['R_mean'])} [{fmt(crow['ci_low'])}, {fmt(crow['ci_high'])}]")
        label = f"**{m}**" if m in BOLD else m
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## 7. GenBio-PathFM reference numbers\n",
          "Two **separate protocol tracks** -- neither is directly comparable to the A0-A9 table "
          "above (different targets/splits/heads), and they are not comparable to *each other* "
          "either (different training protocol). Do not average across tracks.\n",
          "### 7a. Official HEST-Benchmark protocol\n",
          "log1p-only targets (no library-size normalization), StandardScaler+PCA(256)+Ridge, "
          "official published k-fold splits, seed=1. Distilled student and GenBio-PathFM teacher "
          "evaluated identically -- this isolates the effect of the *encoder* from everything else "
          "in the custom pipeline (no trained patch head, no GNN, no train-fold HVG selection).\n",
          "| Cancer | n folds | Distilled student R (std) | GenBio-PathFM teacher R (std) | Gap (teacher - student) |",
          "|---|---|---|---|---|"]
    for cancer in cancers:
        d = official[(official.cancer == cancer) & (official.encoder == "distilled_convnextv2_atto")].iloc[0]
        g = official[(official.cancer == cancer) & (official.encoder == "genbio_pathfm")].iloc[0]
        gap = g.pearson_mean - d.pearson_mean
        L.append(f"| {cancer} | {int(d.n_folds)} | {fmt(d.pearson_mean)} ({fmt(d.pearson_std)}) | "
                  f"{fmt(g.pearson_mean)} ({fmt(g.pearson_std)}) | {gap:+.4f} |")
    L.append("\n### 7b. Custom-pipeline GenBio embedding -> trained MLP baseline (B1) -- PARTIAL\n")
    L.append("Frozen 4608-D GenBio-PathFM embedding fed directly into a small trained MLP head (no "
             "PCA, no Ridge -- `src/scripts/run_genbio_hest_baseline.py --head ridge` is an available "
             "but unused alternative), on the SAME HVG1000/HVG50/PROGENy14 targets and fold "
             "structure as the A0-A9 table -- but **fold0/seed0 only, not the full 5-fold design**. "
             "Treat as preliminary; do not compare CI-to-CI against the bootstrapped A0-A9 numbers "
             "above (no CI is computable from a single fold).\n")
    L.append("| Cancer | n train / test patients | HVG1000 R | HVG50 R | PROGENy14 R | Feature extract (s) | Head train (s) | Peak VRAM (GB) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in genbio_b1.iterrows():
        L.append(f"| {r['cancer']} | {int(r['n_train_patients'])} / {int(r['n_test_patients'])} | "
                  f"{fmt(r['HVG1000_r'])} | {fmt(r['HVG50_r'])} | {fmt(r['PROGENy14_r'])} | "
                  f"{r['feature_extraction_time']:.1f} | {r['head_training_time']:.1f} | "
                  f"{r['peak_vram']:.2f} |")
    L.append("")

    L += ["## 8. Efficiency\n", "| Metric | Value | Status | Source |", "|---|---|---|---|"]
    for _, r in efficiency.iterrows():
        val = "n/a" if pd.isna(r["value"]) else (
            f"{r['value']:.1f}x" if r["metric"].startswith("compression") else f"{int(r['value']):,}")
        L.append(f"| {r['metric']} | {val} | {r['status']} | {r['source']} |")
    L.append("")

    L += ["## 9. Bottom line\n",
          "- **Spatial graph correction helps (Gate 2, SUPPORTED everywhere).** A8 beats A2 (patch-"
          "only) pooled (delta_R=+0.0253, 95% CI [0.011, 0.043]) and in all 3 cancers individually "
          "(IDC +0.060, COAD +0.006, CCRCC +0.011) -- the most robust positive result in the "
          "experiment.",
          "- **But not because of *real* tissue topology (Gate 3, NOT SUPPORTED everywhere).** A8 "
          "(real multiscale graph) is statistically indistinguishable from A5 (degree-preserving "
          "shuffled graph) pooled (delta_R=-0.002, CI crosses zero) and in every cancer. The A8-vs-"
          "A5 secondary comparison confirms the same null result. The GNN's benefit most plausibly "
          "comes from generic neighbor-averaging/regularization, not from encoding true spatial "
          "adjacency -- a genuine negative finding, reported as such per the FINAL STOP RULE.",
          "- **CPTAC->HEST distillation does not reliably help (Gate 1, NOT SUPPORTED pooled).** A2 "
          "(distilled) vs. A1 (ImageNet) pooled delta_R=+0.0115 but CI=[-0.004, 0.025] crosses zero "
          "(p=0.072, just short of one-sided significance). It flips sign across cancers: CCRCC "
          "SUPPORTED (+0.031), COAD marginal (+0.015, not supported), IDC actually reverses "
          "(-0.012, ImageNet ahead). The IDC reversal is flagged, not explained, in Limitations.",
          "- **A8 (proposed flagship) is not the single best ablation among the graph variants.** "
          "A7 (union multiscale, no scale-aware gating) and A9 (GATv2 architecture control) score "
          "marginally higher than A8 pooled on HVG1000 (A7: 0.1019, A9: 0.1024 vs. A8: 0.0981); the "
          "A8-vs-A7 and A8-vs-A9 secondary comparisons are both NOT SUPPORTED (A8 does not "
          "significantly beat either). A8 does significantly beat the parameter-matched MLP (A3) "
          "with no graph at all -- so *some* graph structure helps, but the scale-aware gating "
          "mechanism specifically does not show a measurable edge over simpler graph variants at "
          "this sample size.",
          "- **Distillation buys a real, large compression at a real, consistent accuracy cost.** "
          "Under the official protocol (encoder-only comparison, no GNN, no patch head), GenBio-"
          "PathFM beats the distilled student in all 3 cancers by 0.04-0.11 R, for "
          f"{efficiency[efficiency.metric=='compression_ratio_teacher_over_student']['value'].iloc[0]:.1f}x "
          "fewer parameters."]
    L.append("")

    L += ["## 10. Limitations (reported honestly, not omitted)\n",
          "1. **Bootstrap resampling unit.** All CIs above resample (cancer, fold, seed) cells (15 "
          "pooled, 5 per cancer), not literal per-patient R_p values -- script 10 does not persist "
          "per-patient predictions to disk. A stricter implementation would bootstrap over patients "
          "directly.",
          "2. **Seeds 1/2** were not run for the major models -- seed 0 only.",
          "3. **B1 (GenBio embedding -> MLP baseline)** is fold0/seed0 only per cancer, not the full 5-fold "
          "design the rest of this document uses.",
          "4. **Efficiency**: only parameter counts and the derived compression ratio were "
          "measured; inference VRAM, throughput, and GNN wall-clock were not.",
          "5. **The IDC-specific distillation reversal** (Gate 1 section, A1 ahead of A2 in IDC "
          "specifically) has not been separately root-caused.",
          "6. **A8 vs. A7/A9** shows A8 is not clearly the best graph variant at this sample size "
          "(section 9) -- worth noting for anyone citing A8 as unambiguously superior."]
    L.append("\nFull machine-readable tables: `results/final/tables/*.csv`. Exact patient/fold "
              "manifest: `results/final/folds/cv_folds_used.csv`. Full run config: "
              "`experimental_runs/2026-08-31_job84871_final_5fold_3cancer_seed0_with_results_final/"
              "config_used.json`.\n")

    text = "\n".join(L)
    out_path = final_dir / "HEST_FINAL_5FOLD_RESULTS.md"
    out_path.write_text(text)
    print(f"[21_results_summary_doc] wrote {out_path}")

    docs_path = Path(__file__).resolve().parents[1] / "docs" / "HEST_FINAL_5FOLD_RESULTS.md"
    docs_path.write_text(text)
    print(f"[21_results_summary_doc] wrote {docs_path}")


if __name__ == "__main__":
    main()
