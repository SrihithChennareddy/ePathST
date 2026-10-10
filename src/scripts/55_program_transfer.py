#!/usr/bin/env python
"""Step 7 (U4) of FINAL_GUIDE.md: which kinds of genes can be read from H&E, which survive
compression (T-E), and which gain from tissue context (P-E). Hypothesis to test, not assume:
cell-identity/lineage genes transfer better than cell-state genes. Exploratory throughout.
Reads only screen_full.csv, the HEST per-spot test predictions and configs/gene_programs.json --
nothing from the survival step. Output in results/tcga_final/programs/."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.analysis.stats_utils import bh, perm_p, rank_biserial

SRIHITH = Path("/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith")
WORK = SRIHITH / "cptac_hoptimus0_distill_hest_pna_work"
HEST_RUNS = {0: WORK / "results" / "leiden_official_hvg50", 1: WORK / "results" / "leiden_official_hvg50_seed1"}
HEAD_OF = {"BRCA": "IDC", "COAD": "COAD", "KIRC": "CCRCC"}
SYSTEMS = {"E": "ePathST-E", "P": "ePathST", "T": "H-Optimus-0"}
# Step 1 A7: fold models that collapsed (flag_pcc_lt_0.05), as (cancer_head, fold, system, seed)
COLLAPSED = {("IDC", 2, "E", 0), ("CCRCC", 4, "T", 1)}

log = logging.getLogger("55_program_transfer")


def hest_per_gene_pcc(head: str, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """O1: per-gene test PCC, mean over folds then over seeds 0/1, per system. Returns
    (all_folds, excluding_collapsed) wide tables indexed by gene, columns E/P/T."""
    rows_all, rows_excl = [], []
    for sys_code, sys_name in SYSTEMS.items():
        per_seed_all, per_seed_excl = [], []
        for seed, run in HEST_RUNS.items():
            files = sorted((run / "preds").glob(f"{head}_fold*_{sys_name}.npz"))
            fold_pccs_all, fold_pccs_excl = [], []
            for f in files:
                fold = int(f.stem.split("_fold")[1].split("_")[0])
                z = np.load(f)
                y_true, y_pred = z["y_true"], z["y_pred"]
                pcc = np.array(
                    [stats.pearsonr(y_true[:, gi], y_pred[:, gi])[0] for gi in range(len(genes))]
                )
                fold_pccs_all.append(pcc)
                if (head, fold, sys_code, seed) not in COLLAPSED:
                    fold_pccs_excl.append(pcc)
            per_seed_all.append(np.mean(fold_pccs_all, axis=0))
            if fold_pccs_excl:
                per_seed_excl.append(np.mean(fold_pccs_excl, axis=0))
        rows_all.append(pd.Series(np.mean(per_seed_all, axis=0), index=genes, name=sys_code))
        rows_excl.append(pd.Series(np.mean(per_seed_excl, axis=0), index=genes, name=sys_code))
    return pd.concat(rows_all, axis=1), pd.concat(rows_excl, axis=1)


def program_label(gene: str, programs: dict, head: str) -> str:
    for ptype, genes in programs["program_type"][head].items():
        if gene in genes:
            return ptype
    return "unlabeled"


def domain_groups_of(gene: str, programs: dict, head: str) -> list[str]:
    return [g for g, genes in programs["domain_markers"][head].items() if gene in genes]


def build_gene_level_outcomes(out_root: Path, programs: dict, panels: dict) -> pd.DataFrame:
    screen = pd.read_csv(out_root / "screen" / "screen_full.csv")
    gene_props = pd.read_csv(out_root / "features" / "gene_properties.csv")
    rows = []
    for cancer, genes in panels.items():
        head = HEAD_OF[cancer]
        o1_all, o1_excl = hest_per_gene_pcc(head, genes)
        for gene in genes:
            dmarks = domain_groups_of(gene, programs, head)
            row = {
                "cancer": cancer, "gene": gene,
                "program": program_label(gene, programs, head),
                "domain_groups": ";".join(dmarks), "n_domain_groups": len(dmarks),
            }
            props = gene_props[(gene_props.cancer == cancer) & (gene_props.gene == gene)]
            row["mean_log2tpm1"] = props["mean_log2tpm1"].iloc[0] if len(props) else np.nan
            row["sd_log2tpm1"] = props["sd_log2tpm1"].iloc[0] if len(props) else np.nan
            row["zero_share"] = props["zero_share"].iloc[0] if len(props) else np.nan
            for sys_code in SYSTEMS:
                row[f"O1_{sys_code}"] = o1_all.loc[gene, sys_code]
                row[f"O1_excl_{sys_code}"] = o1_excl.loc[gene, sys_code]
                s = screen[(screen.cancer == cancer) & (screen.gene == gene) & (screen.system == sys_code)]
                row[f"O2_{sys_code}"] = s["rho"].iloc[0] if len(s) else np.nan
                row[f"O3_{sys_code}"] = s["prho"].iloc[0] if len(s) else np.nan
            row["O4_hest"] = row["O1_T"] - row["O1_E"]
            row["O4_tcga"] = row["O2_T"] - row["O2_E"]
            row["O5_hest"] = row["O1_P"] - row["O1_E"]
            row["O5_tcga"] = row["O2_P"] - row["O2_E"]
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_rank_biserial_ci(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ab = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        if len(np.unique(ab)) < 2 or len(np.unique(bb)) < 2 or np.std(ab) == 0 or np.std(bb) == 0:
            vals.append(rank_biserial(ab, bb) if len(ab) and len(bb) else np.nan)
        else:
            vals.append(rank_biserial(ab, bb))
    return np.nanpercentile(vals, [2.5, 97.5])


def per_cancer_program_tests(gl: pd.DataFrame, outcome_cols: list[str], n_perm: int) -> list[dict]:
    rows = []
    for outcome in outcome_cols:
        for cancer, g in gl.groupby("cancer"):
            g = g.dropna(subset=[outcome, "program"])
            g = g[g["program"] != "unlabeled"]
            groups = {p: gg[outcome].to_numpy() for p, gg in g.groupby("program")}
            if len(groups) >= 2 and all(len(v) >= 1 for v in groups.values()):
                try:
                    kw_stat, kw_p = stats.kruskal(*groups.values())
                except ValueError:
                    kw_stat, kw_p = np.nan, np.nan
                rows.append(
                    {"test": "kruskal_program_type", "cancer": cancer, "outcome": outcome,
                     "system": None, "statistic": kw_stat, "effect_size": np.nan,
                     "ci_low": np.nan, "ci_high": np.nan, "p": kw_p}
                )
            if "identity" in groups and "state" in groups:
                a, b = groups["identity"], groups["state"]
                small = len(a) < 8 or len(b) < 8
                eff = rank_biserial(a, b)
                lo, hi = bootstrap_rank_biserial_ci(a, b, seed=0)
                p = np.nan
                if not small:
                    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                rows.append(
                    {"test": "mwu_identity_vs_state", "cancer": cancer, "outcome": outcome,
                     "system": None, "statistic": np.median(a) - np.median(b), "effect_size": eff,
                     "ci_low": lo, "ci_high": hi, "p": p,
                     "note": "medians/effect size only, n<8 in a group" if small else ""}
                )
    return rows


def pooled_permutation_test(gl: pd.DataFrame, outcome_cols: list[str], n_perm: int) -> list[dict]:
    rows = []
    for outcome in outcome_cols:
        g = gl.dropna(subset=[outcome, "program"])
        g = g[g["program"].isin(["identity", "state"])]
        if g["cancer"].nunique() < 2:
            continue

        def stat_fn(labels, _g=g):
            d = _g.assign(_lab=labels)
            per_cancer = []
            for cancer, gg in d.groupby("cancer"):
                idn = gg.loc[gg["_lab"] == "identity", outcome]
                st = gg.loc[gg["_lab"] == "state", outcome]
                if len(idn) and len(st):
                    per_cancer.append(idn.median() - st.median())
            return np.mean(per_cancer) if per_cancer else 0.0

        observed = stat_fn(g["program"].to_numpy())
        p = perm_p(stat_fn, g["program"].to_numpy(), g["cancer"].to_numpy(), n_perm=n_perm, seed=0)
        rows.append(
            {"test": "pooled_permutation_identity_vs_state", "cancer": "pooled", "outcome": outcome,
             "system": None, "statistic": observed, "effect_size": np.nan, "ci_low": np.nan,
             "ci_high": np.nan, "p": p}
        )
    return rows


def ols_expression_control(gl: pd.DataFrame, outcome_cols: list[str]) -> tuple[list[dict], dict]:
    rows, tables = [], {}
    for outcome in outcome_cols:
        d = gl.dropna(subset=[outcome, "program", "mean_log2tpm1", "sd_log2tpm1", "zero_share"]).copy()
        d = d[d["program"] != "unlabeled"]
        if d["cancer"].nunique() < 1 or d["program"].nunique() < 2:
            continue
        formula = f'{outcome} ~ C(program, Treatment("identity")) + mean_log2tpm1 + sd_log2tpm1 + zero_share + C(cancer)'
        try:
            model = smf.ols(formula, data=d).fit(cov_type="HC3")
        except Exception as e:
            log.warning("OLS failed for %s: %s", outcome, e)
            continue
        tables[outcome] = model.summary2().tables[1]
        state_rows = [i for i in model.params.index if "state" in i.lower()]
        for sr in state_rows:
            rows.append(
                {"test": "ols_expression_control", "cancer": "pooled", "outcome": outcome, "system": None,
                 "statistic": model.params[sr], "effect_size": model.params[sr],
                 "ci_low": model.conf_int().loc[sr, 0], "ci_high": model.conf_int().loc[sr, 1],
                 "p": model.pvalues[sr]}
            )
    return rows, tables


def domain_group_tests(gl: pd.DataFrame, outcome_cols: list[str]) -> list[dict]:
    rows = []
    for outcome in outcome_cols:
        for cancer, g in gl.groupby("cancer"):
            g = g[g["n_domain_groups"] == 1].dropna(subset=[outcome])
            if len(g) == 0:
                continue
            groups = {dg: gg[outcome].to_numpy() for dg, gg in g.groupby("domain_groups") if len(gg) > 0}
            if len(groups) < 2:
                continue
            try:
                stat, p = stats.kruskal(*groups.values())
            except ValueError:
                continue
            rows.append(
                {"test": "kruskal_domain_groups", "cancer": cancer, "outcome": outcome, "system": None,
                 "statistic": stat, "effect_size": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": p}
            )
    return rows


def hest_vs_tcga(gl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sys_code in SYSTEMS:
        for cancer, g in gl.groupby("cancer"):
            g = g.dropna(subset=[f"O1_{sys_code}", f"O2_{sys_code}"])
            if len(g) < 3:
                continue
            rho, p = stats.spearmanr(g[f"O1_{sys_code}"], g[f"O2_{sys_code}"])
            rows.append({"cancer": cancer, "system": sys_code, "n": len(g), "rho": rho, "p": p})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="configs/analysis_plan.json")
    ap.add_argument("--out", default="results/tcga_final")
    ap.add_argument("--smoke", action="store_true", help="200 permutations instead of 10000/2000")
    args = ap.parse_args()
    out_root = Path(args.out)
    out = out_root / "programs"
    out.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(out_root / "logs" / "55_program_transfer.log"), logging.StreamHandler()],
    )
    plan = json.loads(Path(args.plan).read_text())
    panels = {c: plan["panels"][cfg["head"]] for c, cfg in plan["cohorts"].items()}
    programs = json.loads(Path("configs/gene_programs.json").read_text())
    n_perm = 200 if args.smoke else 10000

    gl = build_gene_level_outcomes(out_root, programs, panels)
    gl.to_csv(out / "gene_level_outcomes.csv", index=False)
    log.info("gene_level_outcomes.csv: %d rows", len(gl))

    outcome_cols = (
        [f"O1_{s}" for s in SYSTEMS] + [f"O1_excl_{s}" for s in SYSTEMS]
        + [f"O2_{s}" for s in SYSTEMS] + [f"O3_{s}" for s in SYSTEMS]
        + ["O4_hest", "O4_tcga", "O5_hest", "O5_tcga"]
    )
    tests = []
    tests += per_cancer_program_tests(gl, outcome_cols, n_perm)
    tests += pooled_permutation_test(gl, outcome_cols, n_perm)
    ols_rows, ols_tables = ols_expression_control(gl, outcome_cols)
    tests += ols_rows
    tests += domain_group_tests(gl, outcome_cols)
    program_tests = pd.DataFrame(tests)
    program_tests["q"] = bh(program_tests["p"].to_numpy())
    program_tests.sort_values("q", na_position="last").to_csv(out / "program_tests.csv", index=False)

    with open(out / "ols_tables.txt", "w") as f:
        for outcome, table in ols_tables.items():
            f.write(f"=== {outcome} ===\n{table.to_string()}\n\n")

    hv = hest_vs_tcga(gl)
    hv.to_csv(out / "hest_vs_tcga.csv", index=False)

    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
