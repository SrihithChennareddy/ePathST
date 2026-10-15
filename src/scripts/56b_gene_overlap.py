#!/usr/bin/env python
"""Exploratory per-gene overlap table: does a gene that transfers to TCGA also carry prognosis?
Joins the Step 4 screen (P), the Step 8.2 per-gene HRs (P and measured RNA, B) and the Step 8.1
selection frequency (C+P and C+B). No fitting. Output:
results/tcga_final/survival_exploratory/gene_overlap_{BRCA,COAD,KIRC}.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HEAD_OF = {"BRCA": "IDC", "COAD": "COAD", "KIRC": "CCRCC"}


def program_of(gene: str, programs: dict, head: str) -> str:
    for ptype, genes in programs["program_type"][head].items():
        if gene in genes:
            return ptype
    return "unlabeled"


def selection(stab: pd.DataFrame, cancer: str, cohort: str, fset: str, prefix: str) -> pd.DataFrame:
    s = stab[(stab.cancer == cancer) & (stab.cohort == cohort) & (stab.feature_set == fset)
             & stab.feature.str.startswith(f"{prefix}__")].copy()
    s["gene"] = s["feature"].str[len(prefix) + 2:]
    return s.set_index("gene")[["share_nonzero", "sign", "stable_driver"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="configs/analysis_plan.json")
    ap.add_argument("--out", default="results/tcga_final")
    args = ap.parse_args()
    root = Path(args.out)
    plan = json.loads(Path(args.plan).read_text())
    programs = json.loads(Path("configs/gene_programs.json").read_text())
    screen = pd.read_csv(root / "screen" / "screen_full.csv")
    hr = pd.read_csv(root / "survival_exploratory" / "per_gene_hr.csv")
    stab = pd.read_csv(root / "survival_exploratory" / "stability.csv")
    out = root / "survival_exploratory"

    for cancer, head in HEAD_OF.items():
        panel = plan["panels"][head]
        sc = screen[(screen.cancer == cancer) & (screen.system == "P")].set_index("gene")
        hp = hr[(hr.cancer == cancer) & (hr.system == "P")].set_index("gene")
        hb = hr[(hr.cancer == cancer) & (hr.system == "B")].set_index("gene")
        sp = selection(stab, cancer, "full", "C+P", "P")
        sb = selection(stab, cancer, "rna", "C+B", "B")
        rows = []
        for g in panel:
            r = {"cancer": cancer, "gene": g, "program": program_of(g, programs, head)}
            if g in sc.index:
                r.update({"transfer_rho": sc.at[g, "rho"], "transfer_ci_low": sc.at[g, "ci_low"],
                          "transfer_ci_high": sc.at[g, "ci_high"], "transfer_q": sc.at[g, "q"],
                          "transfer_partial_rho": sc.at[g, "prho"], "transfer_partial_q": sc.at[g, "q_prho"]})
            for tag, h in (("pred", hp), ("rna", hb)):
                if g in h.index:
                    r.update({f"{tag}_hr_per_sd": h.at[g, "hr_per_sd"], f"{tag}_hr_ci_low": h.at[g, "ci_low"],
                              f"{tag}_hr_ci_high": h.at[g, "ci_high"], f"{tag}_hr_p": h.at[g, "wald_p"],
                              f"{tag}_hr_q": h.at[g, "q"]})
            lp, lb = np.log(r.get("pred_hr_per_sd", np.nan)), np.log(r.get("rna_hr_per_sd", np.nan))
            r["same_direction"] = bool(np.sign(lp) == np.sign(lb)) if np.isfinite(lp) and np.isfinite(lb) else np.nan
            # absent from coefficients.csv = never selected
            r["selfreq_CP"] = sp["share_nonzero"].get(g, 0.0)
            r["stable_driver_CP"] = bool(sp["stable_driver"].get(g, False))
            r["selfreq_CB"] = sb["share_nonzero"].get(g, 0.0)
            r["stable_driver_CB"] = bool(sb["stable_driver"].get(g, False))
            r["label"] = "exploratory"
            rows.append(r)
        df = pd.DataFrame(rows).sort_values("transfer_rho", ascending=False)
        df.to_csv(out / f"gene_overlap_{cancer}.csv", index=False)
        print(f"wrote {out / f'gene_overlap_{cancer}.csv'} ({len(df)} genes)")


if __name__ == "__main__":
    main()
