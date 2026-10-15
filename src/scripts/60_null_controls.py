#!/usr/bin/env python
"""Post-hoc null controls (added 2026-10-15 after review; see DEVIATIONS.md). Shuffled-patient
permutations on the existing predictions, no retraining of the expression models.

transfer: one patient permutation applied to the whole predicted matrix (keeps gene-gene
  correlation), Spearman vs measured RNA per gene, BH per system over the 150 gene-cancer pairs as
  in 53_tcga_screen.py; null of #genes q<0.05 and median rho.
kirc_hr: predicted-gene rows shuffled across KIRC patients (age, stage, outcome fixed), then the
  full Step 5 C/C+P nested CV and Step 5.6 independence fit rerun; null of the P-score HR, LR stat
  and delta C. Answers whether the clinical covariates inside the C+P score explain HR 2.54.
Output in results/tcga_final/null_controls/."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.analysis.stats_utils import bh

SYSTEMS = ["E", "P", "T"]
log = logging.getLogger("60_null_controls")


def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rank_cols(a):
    return np.apply_along_axis(stats.rankdata, 0, a)


def spearman_cols(rx, ry):
    """Column-wise Spearman from pre-ranked, complete matrices; t-approx p (all n >= 100 here)."""
    n = rx.shape[0]
    rx = (rx - rx.mean(0)) / rx.std(0)
    ry = (ry - ry.mean(0)) / ry.std(0)
    rho = (rx * ry).mean(0)
    t = rho * np.sqrt((n - 2) / np.clip(1 - rho**2, 1e-12, None))
    return rho, 2 * stats.t.sf(np.abs(t), n - 2)


def transfer_null(root: Path, panels: dict, n_perm: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    screen = load_mod("53_tcga_screen")
    data = {}
    for cancer, genes in panels.items():
        _, pred, bulk, _ = screen.load_cancer(root / "features", root / "cohort", cancer, genes)
        keep = bulk.notna().all(1) & np.all([pred[s].notna().all(1) for s in SYSTEMS], axis=0)
        data[cancer] = ({s: pred[s][keep].to_numpy() for s in SYSTEMS}, rank_cols(bulk[keep].to_numpy()))

    def stats_for(perms):
        out = {}
        for s in SYSTEMS:
            rhos, ps, cs = [], [], []
            for cancer, (pred, ry) in data.items():
                x = pred[s] if perms is None else pred[s][perms[cancer]]
                rho, p = spearman_cols(rank_cols(x), ry)
                rhos.append(rho); ps.append(p); cs += [cancer] * len(rho)
            q = bh(np.concatenate(ps))
            rho_all, cs = np.concatenate(rhos), np.array(cs)
            for cancer in data:
                m = cs == cancer
                out[(cancer, s)] = (int((q[m] < 0.05).sum()), float(np.median(rho_all[m])))
        return out

    obs = stats_for(None)
    rng = np.random.default_rng(seed)
    null_rows = []
    for k in range(n_perm):
        perms = {c: rng.permutation(d[1].shape[0]) for c, d in data.items()}
        for (c, s), (nq, med) in stats_for(perms).items():
            null_rows.append({"perm": k, "cancer": c, "system": s, "n_q05": nq, "median_rho": med})
    null = pd.DataFrame(null_rows)
    summ = []
    for (c, s), (nq, med) in obs.items():
        g = null[(null.cancer == c) & (null.system == s)]
        summ.append({"cancer": c, "system": s, "n_patients": data[c][1].shape[0],
                     "obs_n_q05": nq, "null_n_q05_mean": g.n_q05.mean(), "null_n_q05_95pct": g.n_q05.quantile(0.95),
                     "null_n_q05_max": g.n_q05.max(), "p_n_q05": (1 + (g.n_q05 >= nq).sum()) / (n_perm + 1),
                     "obs_median_rho": med, "null_median_rho_95pct": g.median_rho.quantile(0.95),
                     "p_median_rho": (1 + (g.median_rho >= med).sum()) / (n_perm + 1), "n_perm": n_perm})
    return pd.DataFrame(summ), null


def one_kirc_perm(k: int, seed: int, plan_path: str, root: str, smoke: bool):
    logging.disable(logging.CRITICAL)
    surv = load_mod("54_tcga_survival")
    plan = json.loads(Path(plan_path).read_text())
    panel = plan["panels"]["CCRCC"]
    root = Path(root)
    df = surv.load_frame("KIRC", "TCGA-KIRC", root / "features", root / "cohort")
    seeds = plan["survival"]["repeat_seeds"][:2] if smoke else plan["survival"]["repeat_seeds"]
    if k >= 0:
        pcols = [f"P__{g}" for g in panel]
        perm = np.random.default_rng([seed, k]).permutation(len(df))
        df[pcols] = df[pcols].to_numpy()[perm]
    cindex, _, _, _, oof, _, _ = surv.run_cohort("KIRC", df, panel, ["C", "C+P"], "full", seeds, 32, root, [])
    d = surv.delta_table(cindex, [("C+P", "C", ["KIRC"])])
    ind, _ = surv.independence_and_ph("KIRC", df, oof["C+P"])
    return {"perm": k, "delta_c": d.mean_delta_c.iloc[0], "delta_p": d.p.iloc[0],
            "hr_per_sd": ind["hr_per_sd"], "lr_stat": ind["lr_stat"], "lr_p": ind["lr_p"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="configs/analysis_plan.json")
    ap.add_argument("--out", default="results/tcga_final")
    ap.add_argument("--part", choices=["transfer", "kirc_hr", "both"], default="both")
    ap.add_argument("--n-perm-transfer", type=int, default=1000)
    ap.add_argument("--n-perm-hr", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--smoke", action="store_true", help="2 repeats in the survival CV (timing only)")
    ap.add_argument("--dest", default=None, help="output dir (default <out>/null_controls)")
    args = ap.parse_args()
    root = Path(args.out)
    out = Path(args.dest) if args.dest else root / "null_controls"
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "60_null_controls.log"), logging.StreamHandler()])
    plan = json.loads(Path(args.plan).read_text())
    panels = {c: plan["panels"][cfg["head"]] for c, cfg in plan["cohorts"].items()}

    if args.part in ("transfer", "both"):
        summ, null = transfer_null(root, panels, args.n_perm_transfer, seed=0)
        summ.to_csv(out / "transfer_null_summary.csv", index=False)
        null.to_csv(out / "transfer_null_draws.csv", index=False)
        log.info("transfer null:\n%s", summ.to_string(index=False))

    if args.part in ("kirc_hr", "both"):
        rows = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(one_kirc_perm)(k, 0, args.plan, str(root), args.smoke) for k in range(-1, args.n_perm_hr)
        )
        draws = pd.DataFrame(rows)
        obs, null = draws[draws.perm == -1].iloc[0], draws[draws.perm >= 0]
        draws.to_csv(out / "kirc_hr_null_draws.csv", index=False)
        n = len(null)
        summ = pd.DataFrame([{
            "obs_hr_per_sd": obs.hr_per_sd, "null_hr_median": null.hr_per_sd.median(),
            "null_hr_95pct": null.hr_per_sd.quantile(0.95), "p_hr": (1 + (null.hr_per_sd >= obs.hr_per_sd).sum()) / (n + 1),
            "obs_lr_stat": obs.lr_stat, "null_lr_95pct": null.lr_stat.quantile(0.95),
            "p_lr": (1 + (null.lr_stat >= obs.lr_stat).sum()) / (n + 1),
            "obs_delta_c": obs.delta_c, "null_delta_c_median": null.delta_c.median(),
            "null_delta_c_95pct": null.delta_c.quantile(0.95), "p_delta_c": (1 + (null.delta_c >= obs.delta_c).sum()) / (n + 1),
            "n_perm": n, "repeats": 2 if args.smoke else 5,
        }])
        summ.to_csv(out / "kirc_hr_null_summary.csv", index=False)
        log.info("KIRC HR null:\n%s", summ.T.to_string())


if __name__ == "__main__":
    main()
