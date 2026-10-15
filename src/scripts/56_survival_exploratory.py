#!/usr/bin/env python
"""Step 8 of FINAL_GUIDE.md: exploratory analyses that explain the Step 5 result -- never replace
or rescue it. All labeled exploratory. Output in results/tcga_final/survival_exploratory/."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy import stats
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.analysis.stats_utils import bh

SRIHITH = Path("/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith")
COMPANION = SRIHITH / "tcga" / "companion_processed"
PROJECTS = {"TCGA-BRCA": "BRCA", "TCGA-COAD": "COAD", "TCGA-KIRC": "KIRC"}
HEAD_OF = {"BRCA": "IDC", "COAD": "COAD", "KIRC": "CCRCC"}
CLIN = ["age_per_10y", "stage_III_IV"]

log = logging.getLogger("56_survival_exploratory")


def load_survival_module():
    spec = importlib.util.spec_from_file_location(
        "survmod", Path(__file__).resolve().parent / "54_tcga_survival.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def program_label(gene, programs, head):
    for ptype, genes in programs["program_type"][head].items():
        if gene in genes:
            return ptype
    return "unlabeled"


def domain_groups_of(gene, programs, head):
    return [g for g, genes in programs["domain_markers"][head].items() if gene in genes]


def selection_stability(coef: pd.DataFrame, cohort_of: dict) -> pd.DataFrame:
    """8.1: per cancer x system(feature_set), each feature's share of the 25 outer fits with a
    non-zero coefficient and the sign of its mean coefficient; stable = share >= 0.80, one sign."""
    rows = []
    for (cancer, cohort, fset), g in coef.groupby(["cancer", "cohort", "feature_set"]):
        if fset not in cohort_of:
            continue
        n_splits = 25
        for feature, gg in g.groupby("feature"):
            share = len(gg) / n_splits
            sign = np.sign(gg["coef"].mean())
            consistent = (np.sign(gg["coef"]) == sign).mean() == 1.0
            rows.append(
                {"cancer": cancer, "cohort": cohort, "feature_set": fset, "feature": feature,
                 "share_nonzero": share, "mean_coef": gg["coef"].mean(), "sign": int(sign),
                 "stable_driver": bool(share >= 0.80 and consistent)}
            )
    return pd.DataFrame(rows)


def per_gene_hr(surv, df_full: pd.DataFrame, df_rna: pd.DataFrame, panels: dict) -> pd.DataFrame:
    """8.2: per cancer x system(P,T,B) x gene, CoxPHFitter(age, stage, gene_z), BH within cancer x
    system over the 50 genes."""
    rows = []
    sources = {"P": ("P", df_full), "T": ("T", df_full), "B": ("B", df_rna)}
    for cancer, genes in panels.items():
        for sys_code, (prefix, df) in sources.items():
            for gene in genes:
                col = f"{prefix}__{gene}"
                if col not in df.columns:
                    continue
                d = df[["time_days", "event"] + CLIN + [col]].dropna()
                z = (d[col] - d[col].mean()) / d[col].std()
                d = d.assign(**{col: z})
                try:
                    cph = CoxPHFitter()
                    cph.fit(d[["time_days", "event"] + CLIN + [col]], duration_col="time_days", event_col="event")
                    hr = cph.summary.loc[col]
                    rows.append(
                        {"cancer": cancer, "system": sys_code, "gene": gene, "n": len(d),
                         "hr_per_sd": float(hr["exp(coef)"]), "ci_low": float(hr["exp(coef) lower 95%"]),
                         "ci_high": float(hr["exp(coef) upper 95%"]), "wald_p": float(hr["p"]),
                         "log_hr": float(hr["coef"])}
                    )
                except Exception as e:
                    log.warning("per-gene HR failed %s %s %s: %s", cancer, sys_code, gene, e)
    df_hr = pd.DataFrame(rows)
    for (cancer, sys_code), g in df_hr.groupby(["cancer", "system"]):
        df_hr.loc[g.index, "q"] = bh(g["wald_p"].to_numpy())
    return df_hr


def agreement_with_sequencing(per_gene: pd.DataFrame, stability: pd.DataFrame, panels: dict) -> pd.DataFrame:
    """8.3: Spearman of log-HR(P) vs log-HR(B) across genes; top-10 selection-frequency overlap
    (P vs B) with the hypergeometric expectation."""
    rows = []
    for cancer, genes in panels.items():
        p = per_gene[(per_gene.cancer == cancer) & (per_gene.system == "P")].set_index("gene")["log_hr"]
        b = per_gene[(per_gene.cancer == cancer) & (per_gene.system == "B")].set_index("gene")["log_hr"]
        common = p.index.intersection(b.index)
        if len(common) < 3:
            continue
        rho, pval = stats.spearmanr(p.loc[common], b.loc[common])

        def top10(fset):
            s = stability[(stability.cancer == cancer) & (stability.feature_set == fset)]
            return set(s.sort_values("share_nonzero", ascending=False).head(10)["feature"].str.replace(f"{fset[-1]}__", "", regex=False))

        top_p = top10("C+P")
        top_b = top10("C+B")
        overlap = len(top_p & top_b)
        n_genes = len(genes)
        # hypergeometric: P(X >= overlap) for drawing 10 from 10 "successes" out of n_genes
        p_hyper = hypergeom.sf(overlap - 1, n_genes, 10, 10) if overlap > 0 else 1.0
        rows.append(
            {"cancer": cancer, "n_genes_common": len(common), "spearman_loghr_P_vs_B": rho,
             "p": pval, "top10_overlap_P_B": overlap, "hypergeom_p_overlap_ge_observed": p_hyper}
        )
    return pd.DataFrame(rows)


def biology_of_drivers(stability: pd.DataFrame, programs: dict, panels: dict) -> pd.DataFrame:
    """8.4: tabulate stable drivers by program type and domain group."""
    rows = []
    stable = stability[stability["stable_driver"]]
    for _, r in stable.iterrows():
        prefix = r["feature_set"].split("+")[-1]
        gene = r["feature"].replace(f"{prefix}__", "")
        if gene not in panels.get(r["cancer"], []):
            continue
        head = HEAD_OF[r["cancer"]]
        rows.append(
            {"cancer": r["cancer"], "feature_set": r["feature_set"], "gene": gene,
             "program": program_label(gene, programs, head),
             "domain_groups": ";".join(domain_groups_of(gene, programs, head)),
             "share_nonzero": r["share_nonzero"], "sign": r["sign"]}
        )
    return pd.DataFrame(rows)


def _km_groups_for_set(surv_mod, df_full: pd.DataFrame, panel: list, fset: str, splits):
    """Train-fold median-risk threshold applied to its own test fold, pooled over the 5 TE folds
    of repeat 0, for one feature set (C or C+P)."""
    from sklearn.preprocessing import StandardScaler
    from sksurv.util import Surv

    pooled_time, pooled_event, pooled_group = [], [], []
    cols = surv_mod.feature_cols(df_full, fset, panel)
    for r, fold, tr_i, te_i in splits:
        tr, te = df_full.iloc[tr_i], df_full.iloc[te_i]
        y_tr = Surv.from_arrays(event=tr["event"].astype(bool).to_numpy(), time=tr["time_days"].to_numpy())
        scaler = StandardScaler().fit(tr[cols].to_numpy(dtype=float))
        X_tr, X_te = scaler.transform(tr[cols].to_numpy(dtype=float)), scaler.transform(te[cols].to_numpy(dtype=float))
        if fset == "C":
            model = surv_mod.CoxPHSurvivalAnalysis(alpha=1e-4, ties="efron")
            model.fit(X_tr, y_tr)
        else:
            model, _ = surv_mod.fit_coxnet(X_tr, y_tr, n_clin=len(surv_mod.CLIN), seed=100)
        risk_tr, risk_te = model.predict(X_tr), model.predict(X_te)
        thresh = np.median(risk_tr)
        group = np.where(risk_te > thresh, "high", "low")
        pooled_time.extend(te["time_days"].tolist())
        pooled_event.extend(te["event"].tolist())
        pooled_group.extend(group.tolist())
    d = pd.DataFrame({"time": pooled_time, "event": pooled_event, "group": pooled_group})
    hi, lo = d[d.group == "high"], d[d.group == "low"]
    result = logrank_test(hi["time"], lo["time"], event_observed_A=hi["event"], event_observed_B=lo["event"])
    return hi, lo, result


def km_illustration_simple(surv_mod, cancer: str, df_full: pd.DataFrame, panel: list, out: Path):
    """8.5: repeat 0 only, train-fold median-risk threshold applied to its test fold, pooled over
    the 5 TE folds. Two panels side by side: C (clinical only: age + stage) and C+P, so the
    reader can see how much of the separation C alone already gives. The curves separate because
    the risk score is a function of age and stage (and, for C+P, the predicted genes) -- a
    threshold-split Cox risk score is *expected* to separate the very outcome it was fit to
    predict; it is not evidence of a training-on-outcome artifact, and it is not informative about
    C+P specifically unless compared against C's separation, which is what this figure does."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = surv_mod.build_splits(df_full, [0])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    row = {"cancer": cancer}
    for ax, fset in zip(axes, ["C", "C+P"]):
        hi, lo, result = _km_groups_for_set(surv_mod, df_full, panel, fset, splits)
        kmf = KaplanMeierFitter()
        for label, dd in (("high", hi), ("low", lo)):
            kmf.fit(dd["time"] / 365.25, event_observed=dd["event"], label=f"{label} (n={len(dd)}, events={int(dd['event'].sum())})")
            kmf.plot_survival_function(ax=ax)
        ax.set_xlabel("years")
        ax.set_title(f"{fset}: log-rank p={result.p_value:.3g}")
        prefix = "" if fset == "C" else "P_"
        row.update(
            {f"{fset}_logrank_p": result.p_value, f"{fset}_n_high": len(hi),
             f"{fset}_events_high": int(hi["event"].sum()), f"{fset}_n_low": len(lo),
             f"{fset}_events_low": int(lo["event"].sum())}
        )
    fig.suptitle(
        f"{cancer}: train-fold median-risk split, repeat 0 (separation reflects the risk score's "
        f"own covariates -- age+stage for C, +predicted genes for C+P -- not the outcome training itself)"
    )
    fig.tight_layout()
    fig.savefig(out / f"km_{cancer}.pdf")
    fig.savefig(out / f"km_{cancer}.png", dpi=300)
    plt.close(fig)
    return row


def subgroup_analysis(surv_mod, cancer: str, df_full: pd.DataFrame, subgroup_mask: pd.Series, panel: list, name: str, seeds: list):
    d = df_full[subgroup_mask].reset_index(drop=True)
    n_events = int(d["event"].sum())
    result = {"cancer": cancer, "subgroup": name, "n": len(d), "events": n_events}
    if n_events < 20:
        result["note"] = "fewer than 20 events: KM/event counts only, no model fit"
        return result, d
    deviations: list = []
    cindex, alpha, coef, pcavar, oof, tau, folds = surv_mod.run_cohort(
        cancer, d, panel, ["C", "C+P"], f"subgroup_{name}", seeds, 32, Path("/dev/null"), deviations
    )
    dtab = surv_mod.delta_table(cindex, [("C+P", "C", [cancer])], cohort=f"subgroup_{name}")
    if len(dtab):
        result.update(
            {"delta_c": dtab["mean_delta_c"].iloc[0], "ci_low": dtab["ci_low"].iloc[0],
             "ci_high": dtab["ci_high"].iloc[0], "p": dtab["p"].iloc[0]}
        )
    return result, d


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="configs/analysis_plan.json")
    ap.add_argument("--out", default="results/tcga_final")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out_root = Path(args.out)
    out = out_root / "survival_exploratory"
    out.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(out_root / "logs" / "56_survival_exploratory.log"), logging.StreamHandler()],
    )
    plan = json.loads(Path(args.plan).read_text())
    panels = {c: plan["panels"][cfg["head"]] for c, cfg in plan["cohorts"].items()}
    programs = json.loads(Path("configs/gene_programs.json").read_text())
    seeds = plan["survival"]["repeat_seeds"][:2] if args.smoke else plan["survival"]["repeat_seeds"]
    surv = load_survival_module()

    coef = pd.read_csv(out_root / "survival" / "coefficients.csv")
    stability = selection_stability(coef, {"C+P": "P", "C+T": "T", "C+B": "B"})
    stability.to_csv(out / "stability.csv", index=False)
    log.info("stability.csv: %d rows, %d stable drivers", len(stability), int(stability["stable_driver"].sum()))

    cdr = pd.read_csv(COMPANION / "clinical_tcga_cdr" / "TCGA-CDR.csv").rename(
        columns={"bcr_patient_barcode": "patient", "type": "cancer_cdr"}
    )

    per_gene_rows, km_rows, subgroup_rows = [], [], []
    for project, cancer in PROJECTS.items():
        panel = panels[cancer]
        df_full = surv.load_frame(cancer, project, out_root / "features", out_root / "cohort")
        df_rna = df_full[df_full["has_rna"]].reset_index(drop=True)
        hr = per_gene_hr(surv, df_full, df_rna, {cancer: panel})
        per_gene_rows.append(hr)

        km = km_illustration_simple(surv, cancer, df_full, panel, out)
        km_rows.append(km)

        co = pd.read_csv(out_root / "cohort" / f"{cancer}_patients.csv")
        if cancer == "BRCA":
            mask_patients = set(co.loc[(co["ER"] == 1) & (co["HER2"] == 0), "patient"])
            name = "ER+/HER2-"
        else:
            raw_stage = cdr[cdr["cancer_cdr"] == cancer][["patient", "ajcc_pathologic_tumor_stage"]]
            if cancer == "COAD":
                keep = raw_stage["ajcc_pathologic_tumor_stage"].isin(["Stage II", "Stage IIA", "Stage IIB", "Stage IIC"])
                name = "Stage II"
            else:  # KIRC
                keep = raw_stage["ajcc_pathologic_tumor_stage"].isin(["Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC"])
                name = "Stage III"
            mask_patients = set(raw_stage.loc[keep, "patient"])
        mask = df_full["patient"].isin(mask_patients)
        result, _ = subgroup_analysis(surv, cancer, df_full, mask, panel, name, seeds)
        subgroup_rows.append(result)
        log.info("%s subgroup %s: %s", cancer, name, result)

    per_gene = pd.concat(per_gene_rows, ignore_index=True)
    per_gene.to_csv(out / "per_gene_hr.csv", index=False)

    agreement = agreement_with_sequencing(per_gene, stability, panels)
    agreement.to_csv(out / "loghr_agreement.csv", index=False)

    biology = biology_of_drivers(stability, programs, panels)
    biology.to_csv(out / "stable_driver_biology.csv", index=False)

    pd.DataFrame(km_rows).to_csv(out / "km_summary.csv", index=False)
    pd.DataFrame(subgroup_rows).to_csv(out / "subgroups.csv", index=False)

    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
