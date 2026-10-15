#!/usr/bin/env python
"""Step 10 of FINAL_GUIDE.md: figures 2-5 (figure 1 is a hand-drawn study-design schematic, not
generated here), main tables, supplement index, and the numbers sheet. One fixed colour per
system (E/P/T/B) across every figure. Vector PDF + 300-dpi PNG. No number appears in a figure
that is not in a results file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLOR = {"E": "#4C72B0", "P": "#DD8452", "T": "#55A868", "B": "#C44E52"}
CANCERS = ["BRCA", "COAD", "KIRC"]


def fig2(out: Path, root: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hest = pd.read_csv(root / "hest" / "hest_main.csv")
    hest = hest[hest.cancer != "three_cancer_mean"]
    screen = pd.read_csv(root / "screen" / "screen_full.csv")
    u3 = pd.read_csv(root / "survival" / "bottleneck_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    width = 0.25
    x = np.arange(len(CANCERS))
    for i, s in enumerate(["E", "P", "T"]):
        vals, s0, s1 = [], [], []
        for c in CANCERS:
            row = hest[(hest.cancer == c) & (hest.system == s)].iloc[0]
            vals.append(row["pcc_mean_all_folds_2seed_mean"])
            s0.append(row["pcc_mean_all_folds_seed0"]); s1.append(row["pcc_mean_all_folds_seed1"])
        ax.bar(x + (i - 1) * width, vals, width, color=COLOR[s], label=s)
        ax.scatter(x + (i - 1) * width, s0, color="black", marker="_", s=40, zorder=3)
        ax.scatter(x + (i - 1) * width, s1, color="black", marker="_", s=40, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(CANCERS)
    ax.set_ylabel("per-gene PCC, 2-seed mean (ticks = each seed)")
    ax.set_title("(a) HEST same-head (A15)")
    ax.legend()

    ax = axes[1]
    for i, s in enumerate(["E", "P", "T"]):
        vals = [screen[(screen.cancer == c) & (screen.system == s)]["rho"].median() for c in CANCERS]
        ax.bar(x + (i - 1) * width, vals, width, color=COLOR[s], label=s)
    ax.set_xticks(x); ax.set_xticklabels(CANCERS)
    ax.set_ylabel("median Spearman rho (50 genes)")
    ax.set_title("(b) TCGA screen (Step 4)")
    ax.axhline(0, color="grey", lw=0.5)

    ax = axes[2]
    comps = ["C+E vs C+HS", "C+P vs C+HS", "C+T vs C+HT", "C+HS vs C+HT"]
    y = np.arange(len(comps))
    for ci, cancer in enumerate(CANCERS):
        vals, lo, hi = [], [], []
        for comp in comps:
            row = u3[(u3.cancer == cancer) & (u3.comparison == comp)]
            if len(row):
                vals.append(row["mean_delta_c"].iloc[0]); lo.append(row["ci_low"].iloc[0]); hi.append(row["ci_high"].iloc[0])
            else:
                vals.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        yy = y + (ci - 1) * 0.2
        ax.errorbar(vals, yy, xerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)],
                    fmt="o", label=cancer, capsize=3)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(comps)
    ax.set_xlabel("Delta Uno's C (genes vs embeddings), corrected 95% CI")
    ax.set_title("(c) U3 bottleneck (Step 6)")
    ax.legend(fontsize=7)

    fig.suptitle("Figure 2: compression")
    fig.tight_layout()
    fig.savefig(out / "fig2_compression.pdf", bbox_inches="tight")
    fig.savefig(out / "fig2_compression.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3(out: Path, root: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hest = pd.read_csv(root / "hest" / "hest_main.csv")
    hest = hest[hest.cancer != "three_cancer_mean"]
    deltas = pd.read_csv(root / "screen" / "system_deltas.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    x = np.arange(len(CANCERS))
    for ci, c in enumerate(CANCERS):
        e0 = hest[(hest.cancer == c) & (hest.system == "E")]["pcc_mean_all_folds_seed0"].iloc[0]
        e1 = hest[(hest.cancer == c) & (hest.system == "E")]["pcc_mean_all_folds_seed1"].iloc[0]
        p0 = hest[(hest.cancer == c) & (hest.system == "P")]["pcc_mean_all_folds_seed0"].iloc[0]
        p1 = hest[(hest.cancer == c) & (hest.system == "P")]["pcc_mean_all_folds_seed1"].iloc[0]
        ax.plot([ci, ci], [p0 - e0, p1 - e1], "o", color=COLOR["P"])
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(CANCERS)
    ax.set_ylabel("P - E, per-gene PCC (both seeds shown)")
    ax.set_title("(a) context gain on HEST")

    ax = axes[1]
    d = deltas[(deltas.comparison == "P-E") & (deltas.cancer == "BRCA")].sort_values("delta_rho")
    ax.errorbar(d["delta_rho"], np.arange(len(d)), xerr=[d["delta_rho"] - d["ci_low"], d["ci_high"] - d["delta_rho"]],
                fmt=".", color=COLOR["P"], ecolor="lightgrey", elinewidth=0.7, capsize=0)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks([]); ax.set_ylabel(f"{len(d)} genes, sorted (BRCA)")
    ax.set_xlabel("P - E, TCGA screen rho")
    ax.set_title("(b) per-gene context gain on TCGA (BRCA)")

    fig.suptitle("Figure 3: context")
    fig.tight_layout()
    fig.savefig(out / "fig3_context.pdf", bbox_inches="tight"); fig.savefig(out / "fig3_context.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4(out: Path, root: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    screen = pd.read_csv(root / "screen" / "screen_full.csv")
    gl = pd.read_csv(root / "programs" / "gene_level_outcomes.csv")

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])

    ax = fig.add_subplot(gs[0, :])
    brca = screen[screen.cancer == "BRCA"].copy()
    order = brca[brca.system == "P"].sort_values("rho")["gene"].tolist()
    yb = {g: i for i, g in enumerate(order)}
    for s in ["E", "P", "T"]:
        d = brca[brca.system == s].set_index("gene").loc[order]
        yy = [yb[g] + {"E": -0.25, "P": 0, "T": 0.25}[s] for g in order]
        sig = d["q"] < 0.05
        ax.errorbar(d["rho"], yy, xerr=[d["rho"] - d["ci_low"], d["ci_high"] - d["rho"]],
                    fmt="o", color=COLOR[s], markersize=np.where(sig, 4, 2.5).mean(), ecolor=COLOR[s],
                    elinewidth=0.5, alpha=0.8, label=s)
        ax.scatter(d["prho"], yy, facecolors="none", edgecolors=COLOR[s], marker="o", s=15, linewidths=0.6)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("Spearman rho (filled=screen, open=partial rho); BRCA, all 50 genes")
    ax.set_title("(a) per-gene transfer, BRCA")
    ax.legend(fontsize=7, loc="lower right")

    ax = fig.add_subplot(gs[1, 0])
    rows = []
    for c in CANCERS:
        for prog in ["identity", "state", "other"]:
            vals = gl[(gl.cancer == c) & (gl.program == prog)]["O2_P"].dropna()
            rows += [(c, prog, v) for v in vals]
    dfp = pd.DataFrame(rows, columns=["cancer", "program", "O2_P"])
    programs = ["identity", "state", "other"]
    positions, data = [], []
    for ci, c in enumerate(CANCERS):
        for pi, prog in enumerate(programs):
            positions.append(ci * 4 + pi)
            data.append(dfp[(dfp.cancer == c) & (dfp.program == prog)]["O2_P"].to_numpy())
    bp = ax.boxplot(data, positions=positions, widths=0.8, showfliers=False, patch_artist=True)
    for patch, prog in zip(bp["boxes"], programs * 3):
        patch.set_facecolor({"identity": "#8DA0CB", "state": "#FC8D62", "other": "#E5C494"}[prog])
    ax.set_xticks([ci * 4 + 1 for ci in range(3)]); ax.set_xticklabels(CANCERS)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("O2 (P), TCGA transfer rho")
    ax.set_title("(b) program type (U4)")
    from matplotlib.patches import Patch

    prog_color = {"identity": "#8DA0CB", "state": "#FC8D62", "other": "#E5C494"}
    ax.legend(handles=[Patch(facecolor=c, label=p) for p, c in prog_color.items()], fontsize=7, loc="lower left")

    ax = fig.add_subplot(gs[1, 1])
    brca_p = screen[(screen.cancer == "BRCA") & (screen.system == "P") & (screen.q < 0.05)]
    gene = brca_p.iloc[(brca_p["rho"] - brca_p["rho"].median()).abs().argsort().iloc[0]]["gene"]
    pred = pd.read_parquet(root / "features" / "BRCA_P_mean.parquet").set_index("patient")[gene]
    bulk = pd.read_parquet(root / "features" / "BRCA_B.parquet").set_index("patient")[gene]
    common = pred.index.intersection(bulk.index)
    ax.scatter(bulk.loc[common], pred.loc[common], s=6, alpha=0.4, color=COLOR["P"])
    r = brca_p[brca_p.gene == gene]["rho"].iloc[0]
    ax.set_xlabel(f"measured log2(TPM+1), {gene}"); ax.set_ylabel("predicted (P)")
    ax.set_title(f"(c) {gene}, BRCA, rho={r:.2f} (median of q<0.05 genes)")

    fig.suptitle("Figure 4: transfer and biology")
    fig.tight_layout()
    fig.savefig(out / "fig4_transfer_biology.pdf", bbox_inches="tight"); fig.savefig(out / "fig4_transfer_biology.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig5(out: Path, root: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shutil

    delta = pd.read_csv(root / "survival" / "delta_c.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    d = delta.sort_values(["family", "q"]).reset_index(drop=True)
    y = np.arange(len(d))
    for yi, row in zip(y, d.itertuples()):
        c = "crimson" if row.family == "primary" else "grey"
        ax.errorbar([row.mean_delta_c], [yi], xerr=[[row.mean_delta_c - row.ci_low], [row.ci_high - row.mean_delta_c]],
                    fmt="o", ecolor=c, color=c, markersize=4, capsize=2)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([f"{r.cancer}: {r.comparison}" for r in d.itertuples()], fontsize=6)
    ax.set_xlabel("Delta Uno's C, corrected 95% CI (red = primary test)")
    ax.set_title("(a) survival forest plot")

    # (b) reuse the already-generated, corrected BRCA KM figure (C vs C+P side by side, repeat 0)
    src = root / "survival_exploratory" / "km_BRCA.png"
    if src.exists():
        img = plt.imread(src)
        axes[1].imshow(img); axes[1].axis("off")
        axes[1].set_title("(b) BRCA KM: C vs C+P (see Step 8)")
    fig.suptitle("Figure 5: prognosis")
    fig.tight_layout()
    fig.savefig(out / "fig5_prognosis.pdf", bbox_inches="tight"); fig.savefig(out / "fig5_prognosis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main_tables(out: Path, root: Path):
    cohort_rows = []
    for c in CANCERS:
        consort = pd.read_csv(root / "cohort" / f"consort_{c}.csv")
        fu = pd.read_csv(root / "cohort" / "followup.csv")
        frow = fu[fu.cancer == c].iloc[0]
        r = consort.iloc[0]
        cohort_rows.append(
            {"cancer": c, "candidate": r["1_candidate"], "step1_eligible": r["after_step1"],
             "survival_eligible": r["4_survival_eligible"], "events": frow["events"],
             "median_followup_years": round(frow["median_followup_years"], 2)}
        )
    pd.DataFrame(cohort_rows).to_csv(out / "table1_cohorts.csv", index=False)

    hest = pd.read_csv(root / "hest" / "hest_main.csv")
    hest.to_csv(out / "table2_hest_same_head.csv", index=False)

    cindex = pd.read_csv(root / "survival" / "cindex_by_split.csv")
    t3 = cindex.groupby(["cancer", "cohort", "feature_set"]).agg(
        uno_c_mean=("uno_c", "mean"), uno_c_sd=("uno_c", "std")
    ).reset_index()
    t3.to_csv(out / "table3_cindex.csv", index=False)
    delta = pd.read_csv(root / "survival" / "delta_c.csv")
    delta.to_csv(out / "table3b_delta_c.csv", index=False)
    u3 = pd.read_csv(root / "survival" / "bottleneck_summary.csv")
    u3.to_csv(out / "table3c_u3.csv", index=False)


def numbers_sheet(out_root: Path, commit: str) -> str:
    entries = [
        ("Primary test: BRCA C+P vs C, delta Uno's C", "0.0163 [-0.028, 0.060], p=0.460",
         "survival/delta_c_primary.csv", "54_tcga_survival.py"),
        ("Pooled (conditional) BRCA C+P vs C", "0.0212 [-0.007, 0.053]", "survival/pooled_delta_c.csv", "54b_pooled_ci.py"),
        ("Secondary: KIRC C+P vs C", "0.0197 [0.004, 0.035], p=0.0136, q=0.231", "survival/delta_c_secondary.csv", "54_tcga_survival.py"),
        ("Secondary: KIRC C+B vs C", "0.0255 [0.003, 0.048], p=0.0271, q=0.231", "survival/delta_c_secondary.csv", "54_tcga_survival.py"),
        ("Independence HR, KIRC (P score, per SD)", "2.54 [1.64, 3.94], p=3e-5", "survival/independence.csv", "54_tcga_survival.py"),
        ("U3 retention ratio R_student, KIRC", "0.650 [0.167, 1.095]", "survival/retention_ratio_ci.csv", "54b_pooled_ci.py"),
        ("U3 retention ratio R_teacher, KIRC", "1.182 [0.234, 2.307]", "survival/retention_ratio_ci.csv", "54b_pooled_ci.py"),
        ("HEST main table, 3-cancer mean (P, 2-seed)", "0.341 (all folds)", "hest/hest_main.csv", "57_hest_tables.py"),
        ("Compute: student MACs/patch", "0.552 GMACs", "hest/compute.csv", "57_hest_tables.py"),
        ("Compute: H-Optimus-0 MACs/patch", "296.0 GMACs (536x student)", "hest/compute.csv", "57_hest_tables.py"),
        ("Compute: ResNet-50 MACs/patch", "4.112 GMACs (7.45x student)", "hest/compute.csv", "57_hest_tables.py"),
        ("ePathST-E linear-probe (own run)", "0.3144 (IDC 0.482/COAD 0.238/CCRCC 0.223)", "hest/provenance_search.json", "57_hest_tables.py"),
        ("BRCA cohort: candidate/eligible/survival-eligible/events", "1045/1030/994/130", "cohort/consort_BRCA.csv", "51_tcga_cohort.py"),
        ("Screen: BRCA genes q<0.05 (P)", "31/50", "screen/screen_full.csv", "53_tcga_screen.py"),
        ("Specificity: BRCA median matched rank (P)", "9.0 (null 25.5)", "screen/specificity_summary.csv", "53_tcga_screen.py"),
        ("U4 pooled permutation, O1_T", "p=0.011, q=0.050", "programs/program_tests.csv", "55_program_transfer.py"),
        ("Sensitivity: BRCA IDC-only, C+P vs C", "0.0087 [-0.044, 0.062], p=0.744, q=0.934 (primary stays null)",
         "survival_sensitivity/delta_c_sensitivity.csv", "54c_sensitivity.py"),
        ("Sensitivity: median aggregation, KIRC C+B vs C", "0.0255 [0.003, 0.048], p=0.027, q=0.934 (matches main analysis, doesn't survive correction)",
         "survival_sensitivity/delta_c_sensitivity.csv", "54c_sensitivity.py"),
        ("Sensitivity: spots-covariate, COAD C+P vs C+E", "0.0263 [0.001, 0.052], p=0.044, q=0.934 (new nominal signal, doesn't survive correction)",
         "survival_sensitivity/delta_c_sensitivity.csv", "54c_sensitivity.py"),
    ]
    lines = ["# Numbers sheet", "", f"Generated from commit `{commit}`.", "",
             "## Headline numbers", "",
             "| Number | Value | File | Script |", "|---|---|---|---|"]
    for name, val, f, s in entries:
        lines.append(f"| {name} | {val} | `results/tcga_final/{f}` | `src/scripts/{s}` |")

    lines += [
        "", "## Full table/figure provenance map",
        "",
        "Every number in every main table and every figure panel traces to one of these files. "
        "Each file's own rows/columns are the per-number index (so, e.g., 'screen_full.csv' is "
        "the row/column answer for all 450 of its own numbers, not spelled out cell by cell here).",
        "",
        "| Table/Figure | Panel/column | File | Script |",
        "|---|---|---|---|",
        "| Table I | cohorts (candidate/eligible/survival-eligible/events/followup), 3 cancers x 6 cols | `figures/table1_cohorts.csv` | `51_tcga_cohort.py` -> `58_figures.py` |",
        "| Table II | HEST same-head, 3 cancers x 3 systems (+3-cancer mean), seed0/seed1/2-seed mean, all-folds/excl-collapsed | `figures/table2_hest_same_head.csv` (= `hest/hest_main.csv`) | `57_hest_tables.py` |",
        "| Table III | C-index, 3 cancers x 2 cohorts x up to 8 feature sets, mean+SD | `figures/table3_cindex.csv` (= `survival/cindex_by_split.csv` aggregated) | `54_tcga_survival.py` |",
        "| Table III (delta C) | all 18 primary+secondary comparisons, corrected CI, p, q | `figures/table3b_delta_c.csv` (= `survival/delta_c.csv`) | `54_tcga_survival.py` |",
        "| Table III (U3) | 12 bottleneck comparisons, corrected CI, p, q | `figures/table3c_u3.csv` (= `survival/bottleneck_summary.csv`) | `54_tcga_survival.py` |",
        "| Fig 2a | HEST same-head bars, 3 cancers x 3 systems x 2 seeds | `hest/hest_main.csv` | `57_hest_tables.py` |",
        "| Fig 2b | TCGA screen median rho, 3 cancers x 3 systems (9 numbers, derived: median of `rho` per cancer x system) | `screen/screen_full.csv` | `53_tcga_screen.py` |",
        "| Fig 2c | U3 bottleneck, 4 comparisons x 3 cancers, corrected CI | `survival/bottleneck_summary.csv` | `54_tcga_survival.py` |",
        "| Fig 3a | P-E per-gene PCC on HEST, 3 cancers x 2 seeds (derived: `pcc_mean_all_folds_seed{0,1}` for P minus E) | `hest/hest_main.csv` | `57_hest_tables.py` |",
        "| Fig 3b | per-gene P-E on TCGA screen, BRCA, 50 genes with CI | `screen/system_deltas.csv` (comparison=P-E, cancer=BRCA) | `53_tcga_screen.py` |",
        "| Fig 4a | per-gene rho + partial rho + CI, BRCA, 50 genes x 3 systems (150 numbers) | `screen/screen_full.csv` (cancer=BRCA) | `53_tcga_screen.py` |",
        "| Fig 4b | O2 (P) by program type, 3 cancers x 3 programs (box five-number summaries) | `programs/gene_level_outcomes.csv` (column O2_P, grouped by cancer/program) | `55_program_transfer.py` |",
        "| Fig 4c | AQP1 scatter, BRCA, 994 patients (predicted vs measured) + rho annotation | `features/BRCA_P_mean.parquet`, `features/BRCA_B.parquet`, `screen/screen_full.csv` (gene=AQP1) | `52_tcga_features.py`, `53_tcga_screen.py` |",
        "| Fig 5a | survival forest, all 18 primary+secondary comparisons | `survival/delta_c.csv` | `54_tcga_survival.py` |",
        "| Fig 5b | BRCA KM, C and C+P panels, repeat 0, numbers at risk in the legend | `survival_exploratory/km_summary.csv` + `survival_exploratory/km_BRCA.png` | `56_survival_exploratory.py` |",
        "",
        "## Step 5.7 sensitivity analyses (supplement)",
        "",
        "| Analysis | File | Script |",
        "|---|---|---|",
        "| BRCA IDC-only (primary + secondary set) | `survival_sensitivity/delta_c_sensitivity.csv` (variant=idc_only) | `54c_sensitivity.py` |",
        "| Cancer-specific covariates (BRCA ER+HER2, KIRC grade) | `survival_sensitivity/delta_c_sensitivity.csv` (variant=cancer_covariates) | `54c_sensitivity.py` |",
        "| Median instead of mean aggregation | `survival_sensitivity/delta_c_sensitivity.csv` (variant=median_aggregation) | `54c_sensitivity.py` |",
        "| Spots-per-patient as covariate | `survival_sensitivity/delta_c_sensitivity.csv` (variant=spots_covariate) | `54c_sensitivity.py` |",
    ]
    lines.append("")
    lines.append(
        "**Scope note:** figure/table panels that are themselves large tables (e.g. Fig 4a's 150 "
        "numbers) are mapped to their one source file rather than enumerated cell by cell here -- "
        "that file's own rows/columns are the per-number index. Sensitivity analyses added "
        "2026-10-10; AUC/integrated-Brier secondary survival metrics and per-gene sensitivity "
        "tables beyond this list are not yet computed."
    )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/tcga_final")
    ap.add_argument("--commit", default="uncommitted")
    args = ap.parse_args()
    root = Path(args.out)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)

    fig2(out, root)
    fig3(out, root)
    fig4(out, root)
    fig5(out, root)
    main_tables(out, root)
    print("wrote", out, "(numbers sheet: 59_numbers_sheet.py)")


if __name__ == "__main__":
    main()
