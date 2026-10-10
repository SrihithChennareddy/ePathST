#!/usr/bin/env python
"""Step 4 of FINAL_GUIDE.md: molecular transfer screen. For each cancer x gene x system (E, P, T):
Spearman rho of patient-mean prediction vs measured RNA (screen-eligible patients: slide + RNA),
a purity-adjusted partial correlation, a gene-specificity control, and a descriptive system
comparison. BH correction is per system over the 150 gene-cancer pairs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.analysis.stats_utils import bh, partial_spearman, perm_p_single

PROJECTS = {"TCGA-BRCA": "BRCA", "TCGA-COAD": "COAD", "TCGA-KIRC": "KIRC"}
SYSTEMS = ("E", "P", "T")
N_BOOT = 2000

log = logging.getLogger("53_tcga_screen")


def load_cancer(features: Path, cohort: Path, cancer: str, genes: list[str]):
    co = pd.read_csv(cohort / f"{cancer}_patients.csv")
    screen = co[co["step1_eligible"] & co["has_rna"]][["patient", "CPE"]].set_index("patient")
    pred = {
        s: pd.read_parquet(features / f"{cancer}_{s}_mean.parquet").set_index("patient")
        for s in SYSTEMS
    }
    bulk = pd.read_parquet(features / f"{cancer}_B.parquet").set_index("patient")
    patients = screen.index
    for s in SYSTEMS:
        pred[s] = pred[s].reindex(patients)[genes]
    bulk = bulk.reindex(patients)[genes]
    purity = screen["CPE"]
    return patients, pred, bulk, purity


def screen_full(cancer: str, patients, pred, bulk, purity, genes: list[str], seed: int = 0):
    n = len(patients)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    rows, boot_rho = [], {}  # boot_rho[(system, gene)] -> (n_boot,) array, for 4.4
    for s in SYSTEMS:
        x_all = pred[s].to_numpy()
        y_all = bulk.to_numpy()
        pur = purity.to_numpy()
        for gi, g in enumerate(genes):
            x, y = x_all[:, gi], y_all[:, gi]
            mask = np.isfinite(x) & np.isfinite(y)
            n_g = int(mask.sum())
            if n_g < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
                rows.append(
                    {"cancer": cancer, "gene": g, "system": s, "n": n_g, "n_purity": 0,
                     "rho": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan,
                     "prho": np.nan, "p_prho": np.nan, "status": "zero-variance"}
                )
                boot_rho[(s, g)] = np.full(N_BOOT, np.nan)
                continue
            xm, ym = x[mask], y[mask]
            rho = float(stats.spearmanr(xm, ym)[0])
            p = float(stats.spearmanr(xm, ym)[1]) if n_g >= 100 else perm_p_single(xm, ym, n_perm=10000, seed=seed)
            # shared draws over the full (unmasked) patient index, so E/P/T and 4.4's deltas are paired
            boot = np.array(
                [
                    (lambda xb, yb: stats.spearmanr(xb, yb)[0] if np.std(xb) > 0 and np.std(yb) > 0 else np.nan)(
                        x[d], y[d]
                    )
                    for d in idx
                ]
            )
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            boot_rho[(s, g)] = boot
            m_pur = mask & np.isfinite(pur)
            prho, p_prho, n_pur = partial_spearman(x[m_pur], y[m_pur], pur[m_pur])
            rows.append(
                {
                    "cancer": cancer, "gene": g, "system": s, "n": n_g, "n_purity": n_pur,
                    "rho": rho, "ci_low": lo, "ci_high": hi, "p": p,
                    "prho": prho, "p_prho": p_prho, "status": "ok",
                }
            )
    return pd.DataFrame(rows), boot_rho


def specificity(cancer: str, pred, bulk, genes: list[str]):
    rows = []
    for s in SYSTEMS:
        x_all = pred[s].to_numpy()
        y_all = bulk.to_numpy()
        mat = np.full((len(genes), len(genes)), np.nan)
        for gi in range(len(genes)):
            x = x_all[:, gi]
            for hi in range(len(genes)):
                y = y_all[:, hi]
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
                    continue
                mat[gi, hi] = stats.spearmanr(x[mask], y[mask])[0]
        ranks = (-mat).argsort(axis=1).argsort(axis=1) + 1  # 1 = highest rho in the row
        matched_rank = np.array([ranks[i, i] for i in range(len(genes))])
        margin = np.array(
            [mat[i, i] - np.median(np.delete(mat[i], i)) for i in range(len(genes))]
        )
        w_stat, w_p = stats.wilcoxon(margin - 0) if np.any(margin != 0) else (np.nan, np.nan)
        rows.append(
            {
                "cancer": cancer, "system": s,
                "median_matched_rank": float(np.median(matched_rank)),
                "share_rank_le_5": float((matched_rank <= 5).mean()),
                "median_margin": float(np.median(margin)),
                "wilcoxon_p": w_p,
            }
        )
        for gi, g in enumerate(genes):
            rows[-1].setdefault("_per_gene", []).append((g, matched_rank[gi], margin[gi]))
    per_gene = pd.DataFrame(
        [
            {"cancer": r["cancer"], "system": r["system"], "gene": g, "matched_rank": mr, "margin": mg}
            for r in rows
            for g, mr, mg in r.pop("_per_gene")
        ]
    )
    summary = pd.DataFrame(rows)
    return summary, per_gene


def system_deltas(cancer: str, genes: list[str], boot_rho: dict, full: pd.DataFrame):
    rows = []
    pairs = [("P", "E"), ("E", "T"), ("P", "T")]
    for a, b in pairs:
        for g in genes:
            boot_a, boot_b = boot_rho[(a, g)], boot_rho[(b, g)]
            delta_boot = boot_a - boot_b
            lo, hi = np.nanpercentile(delta_boot, [2.5, 97.5])
            ra = full[(full.gene == g) & (full.system == a)]["rho"].iloc[0]
            rb = full[(full.gene == g) & (full.system == b)]["rho"].iloc[0]
            rows.append(
                {"cancer": cancer, "gene": g, "comparison": f"{a}-{b}", "delta_rho": ra - rb,
                 "ci_low": lo, "ci_high": hi}
            )
    deltas = pd.DataFrame(rows)
    med_rows = []
    for a, b in pairs:
        d = deltas[deltas.comparison == f"{a}-{b}"]["delta_rho"]
        stat, p = stats.wilcoxon(d) if np.any(d != 0) else (np.nan, np.nan)
        med_rows.append({"cancer": cancer, "comparison": f"{a}-{b}", "median_delta_rho": float(d.median()), "wilcoxon_p": p})
    return deltas, pd.DataFrame(med_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="configs/analysis_plan.json")
    ap.add_argument("--out", default="results/tcga_final")
    args = ap.parse_args()
    out_root = Path(args.out)
    out = out_root / "screen"
    out.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(out_root / "logs" / "53_tcga_screen.log"),
            logging.StreamHandler(),
        ],
    )
    plan = json.loads(Path(args.plan).read_text())
    panels = {c: plan["panels"][cfg["head"]] for c, cfg in plan["cohorts"].items()}
    features, cohort = out_root / "features", out_root / "cohort"

    full_all, deltas_all, delta_sum_all, spec_sum_all, spec_gene_all = [], [], [], [], []
    for cancer, genes in panels.items():
        patients, pred, bulk, purity = load_cancer(features, cohort, cancer, genes)
        full, boot_rho = screen_full(cancer, patients, pred, bulk, purity, genes, seed=plan["screen"]["seed"])
        full_all.append(full)
        deltas, delta_sum = system_deltas(cancer, genes, boot_rho, full)
        deltas_all.append(deltas)
        delta_sum_all.append(delta_sum)
        spec_sum, spec_gene = specificity(cancer, pred, bulk, genes)
        spec_sum_all.append(spec_sum)
        spec_gene_all.append(spec_gene)
        log.info("%s: screened %d genes x %d systems over %d patients", cancer, len(genes), len(SYSTEMS), len(patients))

    full_all = pd.concat(full_all, ignore_index=True)
    # BH per system over all 150 gene-cancer pairs (not within-cancer)
    for s in SYSTEMS:
        m = full_all["system"] == s
        full_all.loc[m, "q"] = bh(full_all.loc[m, "p"].to_numpy())
        full_all.loc[m, "q_prho"] = bh(full_all.loc[m, "p_prho"].to_numpy())
    full_all.to_csv(out / "screen_full.csv", index=False)

    pd.concat(spec_sum_all, ignore_index=True).to_csv(out / "specificity_summary.csv", index=False)
    pd.concat(spec_gene_all, ignore_index=True).to_csv(out / "specificity_per_gene.csv", index=False)
    pd.concat(deltas_all, ignore_index=True).to_csv(out / "system_deltas.csv", index=False)
    pd.concat(delta_sum_all, ignore_index=True).to_csv(out / "system_deltas_summary.csv", index=False)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
