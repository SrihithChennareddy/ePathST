#!/usr/bin/env python
"""Step 29: a short, skimmable results summary -- per-cancer A0-A9 averages
plus GenBio-PathFM reference numbers (B1 custom baseline, kept separate
from the official-protocol GenBio number) -- for readers who want the
headline numbers without reading all of FINAL_RESULTS.md. Pure formatting
over results/final/tables/*.csv (script 20's output); computes nothing new."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts.ablations import ABLATION_GRID
from sts.utils import load_config, resolve_run_paths

MODEL_ORDER = [a.name for a in ABLATION_GRID]
MODEL_LABEL = {a.name: a.description for a in ABLATION_GRID}


def main():
    cfg = load_config("configs/base.yaml")
    paths = resolve_run_paths(cfg)
    final_dir = paths["results"] / "final"
    tables = final_dir / "tables"
    cancers = cfg["cancers"]["hest"]

    macro = pd.read_csv(tables / "model_comparison_macro.csv").set_index("model")
    per_cancer = pd.read_csv(tables / "model_comparison_per_cancer.csv")
    genbio_b1 = pd.read_csv(tables / "genbio_ridge_baseline_B1.csv")
    official = pd.read_csv(tables / "official_hest_benchmark.csv")

    def model_row(model, r):
        label = f"**{model}**" if model in ("A2_distilled_patch", "A8_scale_aware_pna") else model
        return f"| {label} | {r['hvg1000_R_mean']:.4f} | {r['hvg50_R_mean']:.4f} | {r['pathway_R_mean']:.4f} |"

    lines = ["# Results Summary\n",
             "Quick-reference version of `FINAL_RESULTS.md` (job 84871, LOCKED final 5-fold x "
             "3-cancer x seed-0 run, 2026-08-31). Numbers are patient-level Fisher-z mean Pearson R, "
             "pooled over 5 folds (seed 0). All figures read directly from `results/final/tables/`, "
             "nothing re-derived here.\n",
             "## Macro average (pooled across IDC + COAD + CCRCC)\n",
             "| Model | HVG1000 R | HVG50 R | Pathway (PROGENy14) R |",
             "|---|---|---|---|"]
    for model in MODEL_ORDER:
        r = macro.loc[model]
        lines.append(model_row(model, r))
    lines.append("\n**A2_distilled_patch** = the patch-only anchor (no graph). "
                  "**A8_scale_aware_pna** = the proposed flagship model (multiscale PNA graph correction).\n")

    lines.append("## Per-cancer averages, A0-A9 (5-fold mean, seed 0)\n")
    for cancer in cancers:
        lines.append(f"### {cancer}\n")
        lines.append("| Model | HVG1000 R | HVG50 R | Pathway R |")
        lines.append("|---|---|---|---|")
        sub = per_cancer[per_cancer["cancer"] == cancer].set_index("model")
        for model in MODEL_ORDER:
            r = sub.loc[model]
            lines.append(model_row(model, r))
        lines.append("")

    lines.append("## GenBio-PathFM reference numbers (two separate tracks -- do not average with the "
                  "A0-A9 table above)\n")
    lines.append("**(1) Official HEST-Benchmark protocol** (log1p-only targets, PCA256+Ridge, official "
                  "k-fold splits, seed=1) -- distilled student vs. its own teacher, same protocol for both:\n")
    lines.append("| Cancer | Distilled student R | GenBio-PathFM teacher R | Gap |")
    lines.append("|---|---|---|---|")
    for cancer in cancers:
        d = official[(official.cancer == cancer) & (official.encoder == "distilled_convnextv2_atto")]
        g = official[(official.cancer == cancer) & (official.encoder == "genbio_pathfm")]
        if len(d) and len(g):
            dv, gv = d.pearson_mean.iloc[0], g.pearson_mean.iloc[0]
            lines.append(f"| {cancer} | {dv:.4f} | {gv:.4f} | {gv - dv:+.4f} |")

    lines.append("\n**(2) Custom-pipeline GenBio-PCA256-Ridge baseline (B1)** -- fold0/seed0 only, "
                  "**PARTIAL**, not the full 5-fold design the A0-A9 table above uses:\n")
    lines.append("| Cancer | HVG1000 R | HVG50 R | PROGENy14 R |")
    lines.append("|---|---|---|---|")
    for _, r in genbio_b1.iterrows():
        lines.append(f"| {r['cancer']} | {r['HVG1000_r']:.4f} | {r['HVG50_r']:.4f} | {r['PROGENy14_r']:.4f} |")

    lines.append("\n## Bottom line\n")
    lines.append("- The spatial graph correction (A8 vs. A2) helps in all 3 cancers (Gate 2: SUPPORTED "
                  "pooled and per-cancer).")
    lines.append("- That benefit does not require *real* tissue topology -- A8 is statistically "
                  "indistinguishable from A5 (shuffled-graph control) everywhere (Gate 3: NOT SUPPORTED).")
    lines.append("- CPTAC->HEST distillation (A2 vs. A1) is not reliably better pooled (Gate 1: NOT "
                  "SUPPORTED); it wins in CCRCC, loses in IDC, is roughly a wash in COAD.")
    lines.append("- Under the official HEST-Benchmark protocol, GenBio-PathFM (1.13B params) beats the "
                  "distilled student (3.39M params, 334.7x smaller) by a real, consistent margin in all "
                  "3 cancers -- the accuracy cost of the 334.7x compression.")
    lines.append("\nFull detail, per-cancer gate statistics, CIs, and limitations: `FINAL_RESULTS.md` "
                  "(same directory).\n")

    out_path = final_dir / "RESULTS_SUMMARY.md"
    out_path.write_text("\n".join(lines))
    print(f"[21_results_summary_doc] wrote {out_path}")


if __name__ == "__main__":
    main()
