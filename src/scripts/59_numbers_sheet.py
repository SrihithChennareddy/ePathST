#!/usr/bin/env python
"""TCGA numbers sheet: every TCGA number for the paper, its value and its source file, read from
results/tcga_final (nothing typed by hand). Also writes cohort/cohort_table.csv (sizes, events,
exclusions). Output: docs/reports/NUMBERS_SHEET.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CANCERS = ["BRCA", "COAD", "KIRC"]


def ci(r, lo="ci_low", hi="ci_high", k=3):
    return f"[{r[lo]:.{k}f}, {r[hi]:.{k}f}]"


def pq(r):
    s = f"p={r['p']:.3g}"
    return s + (f", q={r['q']:.3g}" if "q" in r and pd.notna(r["q"]) else "")


def cohort_table(root: Path) -> pd.DataFrame:
    fu = pd.read_csv(root / "cohort" / "followup.csv").set_index("cancer")
    rows = []
    for c in CANCERS:
        d = pd.read_csv(root / "cohort" / f"{c}_patients.csv")
        ok = d[d.step1_eligible]
        lost = ok[~ok.survival_eligible]
        surv = ok[ok.survival_eligible]
        rows.append({
            "cancer": c, "endpoint": fu.at[c, "endpoint"], "candidate_patients": len(d),
            "excl_slide_qc": int((~d.step1_eligible).sum()),
            "excl_slide_qc_reasons": "; ".join(f"{k}={v}" for k, v in d.loc[~d.step1_eligible, "status"].value_counts().items()),
            "slide_eligible": len(ok),
            "excl_survival": len(lost),
            "excl_missing_stage": int(lost.stage_III_IV.isna().sum()),
            "excl_missing_age": int(lost.age_per_10y.isna().sum()),
            "excl_missing_or_nonpositive_time": int((lost.time_days.isna() | (lost.time_days <= 0)).sum()),
            "survival_eligible": len(surv), "events": int(surv.event.sum()),
            "event_rate": round(surv.event.mean(), 3),
            "with_rna": int(surv.has_rna.sum()),
            "median_followup_years": round(fu.at[c, "median_followup_years"], 2),
            "at_risk_3y": int(fu.at[c, "at_risk_3y"]), "at_risk_5y": int(fu.at[c, "at_risk_5y"]),
            "idc_only": int((surv.histological_type == "Infiltrating Ductal Carcinoma").sum()) if c == "BRCA" else None,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/tcga_final")
    ap.add_argument("--sheet", default="docs/reports/NUMBERS_SHEET.md")
    args = ap.parse_args()
    root = Path(args.out)
    rows = []  # (section, number, value, file)

    def add(sec, name, val, f):
        rows.append((sec, name, val, f))

    # cohort
    ct = cohort_table(root)
    ct.to_csv(root / "cohort" / "cohort_table.csv", index=False)
    for r in ct.itertuples():
        add("Cohort", f"{r.cancer} candidate / slide-eligible / survival-eligible / events ({r.endpoint})",
            f"{r.candidate_patients} / {r.slide_eligible} / {r.survival_eligible} / {r.events}", "cohort/cohort_table.csv")
        add("Cohort", f"{r.cancer} exclusions: slide QC / survival (stage, age, time)",
            f"{r.excl_slide_qc} / {r.excl_survival} ({r.excl_missing_stage}, {r.excl_missing_age}, {r.excl_missing_or_nonpositive_time})",
            "cohort/cohort_table.csv")
        add("Cohort", f"{r.cancer} median follow-up (y), at risk 3y / 5y",
            f"{r.median_followup_years}, {r.at_risk_3y} / {r.at_risk_5y}", "cohort/followup.csv")
    add("Cohort", "BRCA IDC-only survival-eligible", str(int(ct.loc[ct.cancer == "BRCA", "idc_only"].iloc[0])), "cohort/cohort_table.csv")
    ov = json.loads((root / "cohort" / "overlap_check.json").read_text())
    add("Cohort", "TCGA patients overlapping HEST / CPTAC",
        f"{ov['hest_rows_with_any_TCGA_barcode']} / {ov['cptac_patients_with_TCGA_id']}", "cohort/overlap_check.json")
    plan = json.loads(Path("configs/analysis_plan.json").read_text())
    for proj, e in plan["tcga_ensemble"].items():
        if proj.startswith("TCGA-"):
            add("Cohort", f"{proj} slides predicted, folds averaged", f"{e['slides']}, {len(e['folds_used'])}", "configs/analysis_plan.json")

    # screen
    sc = pd.read_csv(root / "screen" / "screen_full.csv")
    for (c, s), g in sc.groupby(["cancer", "system"]):
        add("Screen (transfer)", f"{c} {s}: genes q<0.05 / median rho / genes partial-rho q<0.05",
            f"{(g.q < 0.05).sum()}/50, {g.rho.median():.3f}, {(g.q_prho < 0.05).sum()}/50", "screen/screen_full.csv")
    spec = pd.read_csv(root / "screen" / "specificity_summary.csv")
    for r in spec.itertuples():
        add("Screen (specificity)", f"{r.cancer} {r.system}: median matched rank (null 25.5), share rank<=5, Wilcoxon p",
            f"{r.median_matched_rank}, {r.share_rank_le_5:.2f}, {r.wilcoxon_p:.2g}", "screen/specificity_summary.csv")
    sd = pd.read_csv(root / "screen" / "system_deltas_summary.csv")
    for r in sd.itertuples():
        add("Screen (systems)", f"{r.cancer} {r.comparison}: median delta rho, Wilcoxon p",
            f"{r.median_delta_rho:.4f}, {r.wilcoxon_p:.3g}", "screen/system_deltas_summary.csv")

    # survival
    cidx = pd.read_csv(root / "survival" / "cindex_by_split.csv")
    t = cidx[cidx.cohort == "full"].groupby(["cancer", "feature_set"]).uno_c.agg(["mean", "std"])
    for (c, f), r in t.iterrows():
        add("Survival C-index (Uno, 25 splits)", f"{c} {f}", f"{r['mean']:.3f} ± {r['std']:.3f}", "survival/cindex_by_split.csv")
    tau = cidx.drop_duplicates(["cancer"]).set_index("cancer").tau_days
    for c in CANCERS:
        add("Survival C-index (Uno, 25 splits)", f"{c} horizon tau", f"{tau[c] / 365.25:.0f} y", "survival/cindex_by_split.csv")
    dc = pd.read_csv(root / "survival" / "delta_c.csv")
    for r in dc.to_dict("records"):
        sec = "Survival ΔC — PRIMARY" if r["family"] == "primary" else "Survival ΔC — secondary (BH within family)"
        add(sec, f"{r['cancer']} {r['comparison']} ({r['cohort']}, J={r['n_splits']})",
            f"{r['mean_delta_c']:.4f} {ci(r)}, {pq(r)}", "survival/delta_c.csv")
    pdc = pd.read_csv(root / "survival" / "pooled_delta_c.csv")
    for r in pdc.to_dict("records"):
        add("Survival ΔC — pooled (conditional on fitted models)", f"{r['cancer']} {r['comparison']}",
            f"{r['pooled_delta_c']:.4f} {ci(r)}", "survival/pooled_delta_c.csv")
    ind = pd.read_csv(root / "survival" / "independence.csv")
    for r in ind.to_dict("records"):
        add("Independence (Cox: age + stage + P score)", f"{r['cancer']} HR per SD (n={r['n']})",
            f"{r['hr_per_sd']:.2f} {ci(r, k=2)}, Wald p={r['wald_p']:.2g}, LR p={r['lr_p']:.2g}", "survival/independence.csv")
    ph = pd.read_csv(root / "survival" / "ph_tests.csv")
    for c, g in ph.groupby("cancer"):
        add("Schoenfeld PH test", f"{c} p (" + ", ".join(g.covariate) + ")",
            ", ".join(f"{p:.2f}" for p in g.p), "survival/ph_tests.csv")

    # U3
    u3 = pd.read_csv(root / "survival" / "bottleneck_summary.csv")
    for r in u3.to_dict("records"):
        add("U3 bottleneck ΔC", f"{r['cancer']} {r['comparison']}", f"{r['mean_delta_c']:.4f} {ci(r)}, {pq(r)}",
            "survival/bottleneck_summary.csv")
    rr = pd.read_csv(root / "survival" / "retention_ratio_ci.csv")
    for r in rr.to_dict("records"):
        add("U3 bottleneck ΔC", f"KIRC retention ratio R_{r['embedding']}", f"{r['R']:.3f} {ci(r)}",
            "survival/retention_ratio_ci.csv")

    # U4
    prog = pd.read_csv(root / "programs" / "program_tests.csv")
    for r in prog.sort_values("q").head(10).to_dict("records"):
        label = " ".join(str(r[k]) for k in ("test", "cancer", "outcome", "system") if pd.notna(r[k]))
        eff = f"effect={r['effect_size']:.3f}, " if pd.notna(r["effect_size"]) else ""
        add("U4 program transfer (10 lowest q)", label, f"{eff}p={r['p']:.3g}, q={r['q']:.3g}", "programs/program_tests.csv")

    # exploratory
    stab = pd.read_csv(root / "survival_exploratory" / "stability.csv")
    for (c, coh, f), g in stab.groupby(["cancer", "cohort", "feature_set"]):
        if (coh, f) in (("full", "C+P"), ("full", "C+T"), ("rna", "C+B")):
            drivers = ", ".join(x.split("__")[-1] for x in g[g.stable_driver].feature)
            add("Exploratory: stable drivers (>=80% of 25 fits, one sign)", f"{c} {f}", drivers or "none",
                "survival_exploratory/stability.csv")
    hr = pd.read_csv(root / "survival_exploratory" / "per_gene_hr.csv")
    for (c, s), g in hr.groupby(["cancer", "system"]):
        add("Exploratory: per-gene HR", f"{c} {s}: genes q<0.05", f"{(g.q < 0.05).sum()}/50",
            "survival_exploratory/per_gene_hr.csv")
    ag = pd.read_csv(root / "survival_exploratory" / "loghr_agreement.csv")
    for r in ag.to_dict("records"):
        add("Exploratory: P vs RNA agreement", f"{r['cancer']} Spearman log-HR(P) vs log-HR(B); top-10 overlap",
            f"{r['spearman_loghr_P_vs_B']:.3f} (p={r['p']:.2g}); {r['top10_overlap_P_B']}/10", "survival_exploratory/loghr_agreement.csv")
    for c in CANCERS:
        go = pd.read_csv(root / "survival_exploratory" / f"gene_overlap_{c}.csv")
        add("Exploratory: gene overlap", f"{c}: transfer q<0.05 & pred-HR q<0.05; same HR direction P vs RNA",
            f"{((go.transfer_q < 0.05) & (go.pred_hr_q < 0.05)).sum()}; {int(go.same_direction.sum())}/50",
            f"survival_exploratory/gene_overlap_{c}.csv")
    km = pd.read_csv(root / "survival_exploratory" / "km_summary.csv")
    for r in km.to_dict("records"):
        add("Exploratory: KM (repeat 0, train-fold median)", f"{r['cancer']} log-rank p, C / C+P",
            f"{r['C_logrank_p']:.2g} / {r['C+P_logrank_p']:.2g}", "survival_exploratory/km_summary.csv")
    sg = pd.read_csv(root / "survival_exploratory" / "subgroups.csv")
    for r in sg.to_dict("records"):
        add("Exploratory: subgroups C+P vs C", f"{r['cancer']} {r['subgroup']} (n={r['n']}, events={r['events']})",
            f"{r['delta_c']:.4f} {ci(r)}, p={r['p']:.3g}", "survival_exploratory/subgroups.csv")

    # sensitivity
    sens = pd.read_csv(root / "survival_sensitivity" / "delta_c_sensitivity.csv")
    for r in sens.to_dict("records"):
        add("Sensitivity (Step 5.7, BH over all rows)", f"{r['variant']}: {r['cancer']} {r['comparison']}",
            f"{r['mean_delta_c']:.4f} {ci(r)}, {pq(r)}", "survival_sensitivity/delta_c_sensitivity.csv")

    # HEST same-head (heads used for TCGA)
    hm = pd.read_csv(root / "hest" / "hest_main.csv")
    for r in hm.to_dict("records"):
        add("HEST same-head PCC (2-seed mean)", f"{r['cancer']} {r['system']}",
            f"{r['pcc_mean_all_folds_2seed_mean']:.4f} (excl. collapsed {r['pcc_mean_excl_collapsed_2seed_mean']:.4f})",
            "hest/hest_main.csv")
    comp = pd.read_csv(root / "hest" / "compute.csv")
    for r in comp.to_dict("records"):
        add("Compute (fvcore MACs/patch)", r["model"], f"{r['GMACs_per_patch']:.3f} GMACs", "hest/compute.csv")

    # post-hoc null controls (60_null_controls.py)
    nd = root / "null_controls"
    if (nd / "transfer_null_summary.csv").exists():
        for r in pd.read_csv(nd / "transfer_null_summary.csv").to_dict("records"):
            add("Null control (post-hoc): shuffled-patient transfer", f"{r['cancer']} {r['system']}: genes q<0.05 obs vs null mean / max",
                f"{r['obs_n_q05']} vs {r['null_n_q05_mean']:.3f} / {r['null_n_q05_max']}, perm p={r['p_n_q05']:.3g} ({r['n_perm']} perms)",
                "null_controls/transfer_null_summary.csv")
    if (nd / "kirc_hr_null_summary.csv").exists():
        r = pd.read_csv(nd / "kirc_hr_null_summary.csv").iloc[0]
        n = int(r["n_perm"])
        add("Null control (post-hoc): KIRC shuffled predicted genes", "P-score HR per SD: obs, null median, null 95th pct, perm p",
            f"{r['obs_hr_per_sd']:.2f}, {r['null_hr_median']:.2f}, {r['null_hr_95pct']:.2f}, p={r['p_hr']:.3g} ({n} perms)", "null_controls/kirc_hr_null_summary.csv")
        add("Null control (post-hoc): KIRC shuffled predicted genes", "LR stat vs age+stage: obs, null 95th pct, perm p",
            f"{r['obs_lr_stat']:.2f}, {r['null_lr_95pct']:.2f}, p={r['p_lr']:.3g}", "null_controls/kirc_hr_null_summary.csv")
        add("Null control (post-hoc): KIRC shuffled predicted genes", "ΔC C+P vs C: obs, null median, null 95th pct, perm p",
            f"{r['obs_delta_c']:.4f}, {r['null_delta_c_median']:.4f}, {r['null_delta_c_95pct']:.4f}, p={r['p_delta_c']:.3g}", "null_controls/kirc_hr_null_summary.csv")

    lines = ["# Numbers sheet (TCGA)", "",
             "Generated by `src/scripts/59_numbers_sheet.py` from `results/tcga_final/` "
             "Values are read from the files, not typed.", "",
             "ΔC tables use J=25 splits paired within one cohort (fixed 2026-10-14; the earlier C+P vs C rows "
             "used J=50 by counting the identical full and RNA cohorts twice).", ""]
    sec_prev = None
    for sec, name, val, f in rows:
        if sec != sec_prev:
            lines += ["", f"## {sec}", "", "| Number | Value | Source |", "|---|---|---|"]
            sec_prev = sec
        lines.append(f"| {name} | {val} | `{f}` |")
    Path(args.sheet).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.sheet} ({len(rows)} numbers) and {root / 'cohort' / 'cohort_table.csv'}")


if __name__ == "__main__":
    main()
