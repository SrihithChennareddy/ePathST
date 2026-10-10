#!/usr/bin/env python
"""Step 5 (+ Step 6, U3) of FINAL_GUIDE.md: penalized Cox survival models, nested CV. Expression
models are frozen; the Cox model is trained on TCGA outcomes inside cross-validation (never
"zero-shot"). Output in results/tcga_final/survival/."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.analysis.stats_utils import fit_coxnet, harrell_c, nb_ttest, to_percentile, uno_c

PROJECTS = {"TCGA-BRCA": "BRCA", "TCGA-COAD": "COAD", "TCGA-KIRC": "KIRC"}
HORIZON_YEARS = {"BRCA": 5, "COAD": 3, "KIRC": 5}
CLIN = ["age_per_10y", "stage_III_IV"]
FULL_SETS = ["C", "C+E", "C+P", "C+T", "C+HS", "C+HT"]
RNA_SETS = ["C", "C+P", "C+B", "C+B+P"]
PRIMARY = {"cohort": "BRCA", "comparison": "C+P-C", "cancers": ["BRCA"]}
SECONDARY_COMPARISONS = [
    ("C+P", "C", ["COAD", "KIRC"]),
    ("C+P", "C+E", ["BRCA", "COAD", "KIRC"]),
    ("C+E", "C+T", ["BRCA", "COAD", "KIRC"]),
    ("C+P", "C+T", ["BRCA", "COAD", "KIRC"]),
    ("C+B", "C", ["BRCA", "COAD", "KIRC"]),
    ("C+B+P", "C+B", ["BRCA", "COAD", "KIRC"]),
]
U3_COMPARISONS = [("C+E", "C+HS"), ("C+T", "C+HT"), ("C+P", "C+HS"), ("C+HS", "C+HT")]

log = logging.getLogger("54_tcga_survival")


def load_frame(cancer: str, project: str, features: Path, cohort: Path) -> pd.DataFrame:
    co = pd.read_csv(cohort / f"{cancer}_patients.csv")
    df = co[co["survival_eligible"]][["patient", "event", "time_days"] + CLIN + ["has_rna"]].copy()
    for s in ("E", "P", "T"):
        g = pd.read_parquet(features / f"{cancer}_{s}_mean.parquet").set_index("patient")
        g.columns = [f"{s}__{c}" for c in g.columns]
        df = df.merge(g, left_on="patient", right_index=True, how="left")
    hs = pd.read_parquet(features / f"{cancer}_HS.parquet").set_index("patient")
    ht = pd.read_parquet(features / f"{cancer}_HT.parquet").set_index("patient")
    df = df.merge(hs, left_on="patient", right_index=True, how="left")
    df = df.merge(ht, left_on="patient", right_index=True, how="left")
    b = pd.read_parquet(features / f"{cancer}_B.parquet").set_index("patient")
    b.columns = [f"B__{c}" for c in b.columns]
    df = df.merge(b, left_on="patient", right_index=True, how="left")
    return df.reset_index(drop=True)


def feature_cols(df: pd.DataFrame, fset: str, panel: list[str]) -> list[str]:
    parts = {
        "C": [],
        "E": [f"E__{g}" for g in panel],
        "P": [f"P__{g}" for g in panel],
        "T": [f"T__{g}" for g in panel],
        "B": [f"B__{g}" for g in panel],
        "HS": [c for c in df.columns if c.startswith("hs_")],
        "HT": [c for c in df.columns if c.startswith("ht_")],
    }
    cols = list(CLIN)
    for token in fset.split("+")[1:]:
        cols += parts[token]
    return cols


def build_splits(df: pd.DataFrame, seeds: list[int]) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """One list of (repeat, outer_fold, tr_i, te_i) built once, shared by pick_tau, cv_folds.csv
    and the fitting loop -- so there is exactly one source of truth for what the splits are."""
    splits = []
    for r in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=r)
        for fold, (tr_i, te_i) in enumerate(skf.split(df, df["event"])):
            splits.append((r, fold, tr_i, te_i))
    return splits


def folds_table(df: pd.DataFrame, splits: list) -> pd.DataFrame:
    rows = []
    for r, fold, _, te_i in splits:
        for i in te_i:
            rows.append({"patient": df["patient"].iloc[i], "repeat": r, "outer_fold": fold})
    return pd.DataFrame(rows)


def pick_tau(df: pd.DataFrame, horizon_years: float, splits: list) -> tuple[float, bool]:
    """Step 5.4: lower tau to the largest whole year s.t. every split has >=10% of TE at risk,
    tau doesn't exceed TR's max follow-up, and at least one TE patient has time < tau (required
    for concordance_index_ipcw's internal masking). Fails loudly (raises) if even 1 year does
    not satisfy this in every split -- skipping a fold silently is not allowed. Returns
    (tau_days, was_lowered)."""
    nominal = horizon_years * 365.25
    for year in range(int(horizon_years), 0, -1):
        cand = year * 365.25
        ok = True
        for _, _, tr_i, te_i in splits:
            te_time = df["time_days"].to_numpy()[te_i]
            tr_time = df["time_days"].to_numpy()[tr_i]
            at_risk = (te_time >= cand).mean()
            if at_risk < 0.10 or cand > tr_time.max() or not np.any(te_time < cand):
                ok = False
                break
        if ok:
            return cand, (cand < nominal)
    raise RuntimeError(
        f"no tau down to 1 year satisfies the Step 5.4 rule (>=10% TE at risk, "
        f"tau<=TR max follow-up, >=1 TE patient with time<tau) in every split for this cohort"
    )


def fit_and_score(X_tr, y_tr, X_te, y_te, n_clin: int, tau: float, is_c_only: bool, inner_seed: int):
    if is_c_only:
        m = CoxPHSurvivalAnalysis(alpha=1e-4, ties="efron")
        m.fit(X_tr, y_tr)
        risk_te = m.predict(X_te)
        return risk_te, np.nan, m
    model, best_alpha = fit_coxnet(X_tr, y_tr, n_clin=n_clin, seed=inner_seed)
    risk_te = model.predict(X_te)
    return risk_te, best_alpha, model


def run_cohort(cancer: str, df: pd.DataFrame, panel: list[str], sets: list[str], cohort_tag: str,
               seeds: list[int], pca_components: int, out: Path, deviations: list):
    splits = build_splits(df, seeds)
    folds = folds_table(df, splits)
    folds.insert(0, "cancer", cancer)
    folds.insert(1, "cohort", cohort_tag)
    tau, lowered = pick_tau(df, HORIZON_YEARS[cancer], splits)
    log.info(
        "%s/%s: n=%d events=%d tau=%.1f days (%.2f years)%s", cancer, cohort_tag, len(df),
        int(df["event"].sum()), tau, tau / 365.25, " [LOWERED from nominal horizon]" if lowered else "",
    )
    if lowered:
        deviations.append(
            f"- **{cancer}/{cohort_tag}:** horizon lowered from {HORIZON_YEARS[cancer]} to "
            f"{tau / 365.25:.2f} years ({tau:.1f} days) -- the nominal horizon failed Step 5.4's "
            f"rule (>=10% TE at risk, tau<=TR max follow-up, >=1 TE patient with time<tau) in at "
            f"least one of the {len(splits)} outer splits. Follow-up-driven only, no outcome model result involved."
        )

    cindex_rows, alpha_rows, coef_rows, pca_var_rows = [], [], [], []
    oof_percentile = {s: {r: {} for r in seeds} for s in sets}
    for r, fold, tr_i, te_i in splits:
            tr, te = df.iloc[tr_i], df.iloc[te_i]
            y_tr = Surv.from_arrays(event=tr["event"].astype(bool).to_numpy(), time=tr["time_days"].to_numpy())
            y_te = Surv.from_arrays(event=te["event"].astype(bool).to_numpy(), time=te["time_days"].to_numpy())
            log.info(
                "%s/%s repeat=%d fold=%d: n_tr=%d events_tr=%d n_te=%d events_te=%d",
                cancer, cohort_tag, r, fold, len(tr), int(tr["event"].sum()), len(te), int(te["event"].sum()),
            )
            for fset in sets:
                is_pca = fset in ("C+HS", "C+HT")
                cols = feature_cols(df, fset, panel)
                X_tr_raw, X_te_raw = tr[cols].to_numpy(dtype=float), te[cols].to_numpy(dtype=float)
                scaler = StandardScaler().fit(X_tr_raw[:, len(CLIN):] if is_pca else X_tr_raw)
                if is_pca:
                    clin_tr, clin_te = X_tr_raw[:, : len(CLIN)], X_te_raw[:, : len(CLIN)]
                    emb_tr = scaler.transform(X_tr_raw[:, len(CLIN):])
                    emb_te = scaler.transform(X_te_raw[:, len(CLIN):])
                    # clamp only guards a degenerate split (e.g. a tiny --smoke subset); the full
                    # cohorts have hundreds of patients per fold, well above pca_components=32
                    n_comp = min(pca_components, emb_tr.shape[0] - 1, emb_tr.shape[1])
                    pca = PCA(n_components=n_comp, svd_solver="full", random_state=0).fit(emb_tr)
                    X_tr = np.column_stack([clin_tr, pca.transform(emb_tr)])
                    X_te = np.column_stack([clin_te, pca.transform(emb_te)])
                    pca_var_rows.append(
                        {"cancer": cancer, "cohort": cohort_tag, "feature_set": fset, "repeat": r,
                         "outer_fold": fold, "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum())}
                    )
                else:
                    X_tr, X_te = scaler.transform(X_tr_raw), scaler.transform(X_te_raw)
                # pick_tau already guarantees this for every split in this cohort; if it's wrong,
                # fail loudly here instead of silently dropping the fold from the tables.
                assert np.any(te["time_days"].to_numpy() < tau), (
                    f"{cancer}/{cohort_tag} repeat={r} fold={fold}: 0 TE patients with time < "
                    f"tau={tau} despite pick_tau's guarantee -- this must not be silently skipped"
                )
                risk_te, best_alpha, model = fit_and_score(
                    X_tr, y_tr, X_te, y_te, n_clin=len(CLIN), tau=tau, is_c_only=(fset == "C"),
                    inner_seed=100 + r,
                )
                c_uno = uno_c(y_tr, y_te, risk_te, tau)
                c_harrell = harrell_c(y_te, risk_te)
                cindex_rows.append(
                    {"cancer": cancer, "cohort": cohort_tag, "feature_set": fset, "repeat": r,
                     "outer_fold": fold, "n_tr": len(tr), "n_te": len(te), "tau_days": tau,
                     "uno_c": c_uno, "harrell_c": c_harrell}
                )
                alpha_rows.append(
                    {"cancer": cancer, "cohort": cohort_tag, "feature_set": fset, "repeat": r,
                     "outer_fold": fold, "alpha": best_alpha}
                )
                if fset != "C":
                    coefs = model.coef_.ravel()
                    for name, val in zip(cols, coefs):
                        if val != 0:
                            coef_rows.append(
                                {"cancer": cancer, "cohort": cohort_tag, "feature_set": fset,
                                 "repeat": r, "outer_fold": fold, "feature": name, "coef": float(val)}
                            )
                pct = to_percentile(risk_te)
                for pid, p in zip(te["patient"], pct):
                    oof_percentile[fset][r][pid] = p
    return (
        pd.DataFrame(cindex_rows), pd.DataFrame(alpha_rows), pd.DataFrame(coef_rows),
        pd.DataFrame(pca_var_rows), oof_percentile, tau, folds,
    )


def delta_table(cindex: pd.DataFrame, comparisons, cancers_per_comp=None) -> pd.DataFrame:
    rows = []
    for a, b, *rest in comparisons:
        cancers = rest[0] if rest else cancers_per_comp
        for cancer in cancers:
            da = cindex[(cindex.cancer == cancer) & (cindex.feature_set == a)].set_index(["cohort", "repeat", "outer_fold"])["uno_c"]
            db = cindex[(cindex.cancer == cancer) & (cindex.feature_set == b)].set_index(["cohort", "repeat", "outer_fold"])["uno_c"]
            common = da.index.intersection(db.index)
            if len(common) == 0:
                continue
            deltas = (da.loc[common] - db.loc[common]).to_numpy()
            n_tr = cindex[(cindex.cancer == cancer) & (cindex.feature_set == a)]["n_tr"].mean()
            n_te = cindex[(cindex.cancer == cancer) & (cindex.feature_set == a)]["n_te"].mean()
            mean, lo, hi, t, p = nb_ttest(deltas, n_train=n_tr, n_test=n_te)
            rows.append(
                {"cancer": cancer, "comparison": f"{a} vs {b}", "mean_delta_c": mean,
                 "ci_low": lo, "ci_high": hi, "t": t, "p": p, "n_splits": len(common)}
            )
    df = pd.DataFrame(rows)
    if len(df):
        from epathst.analysis.stats_utils import bh

        df["q"] = bh(df["p"].to_numpy())
    return df


def independence_and_ph(cancer: str, df: pd.DataFrame, oof_percentile_P: dict) -> tuple[dict, dict]:
    avg_pct = {}
    for pid in df["patient"]:
        vals = [oof_percentile_P[r][pid] for r in oof_percentile_P if pid in oof_percentile_P[r]]
        if vals:
            avg_pct[pid] = float(np.mean(vals))
    d = df[df["patient"].isin(avg_pct)].copy()
    d["P_score_z"] = [(avg_pct[p] - np.mean(list(avg_pct.values()))) / np.std(list(avg_pct.values())) for p in d["patient"]]
    cph_full = CoxPHFitter()
    cph_full.fit(d[["time_days", "event"] + CLIN + ["P_score_z"]], duration_col="time_days", event_col="event")
    cph_base = CoxPHFitter()
    cph_base.fit(d[["time_days", "event"] + CLIN], duration_col="time_days", event_col="event")
    lr_stat = 2 * (cph_full.log_likelihood_ - cph_base.log_likelihood_)
    from scipy.stats import chi2

    lr_p = float(chi2.sf(lr_stat, df=1))
    hr = cph_full.summary.loc["P_score_z"]
    independence = {
        "cancer": cancer, "hr_per_sd": float(hr["exp(coef)"]), "ci_low": float(hr["exp(coef) lower 95%"]),
        "ci_high": float(hr["exp(coef) upper 95%"]), "wald_p": float(hr["p"]), "lr_stat": float(lr_stat), "lr_p": lr_p,
        "n": len(d),
    }
    ph = proportional_hazard_test(cph_full, d[["time_days", "event"] + CLIN + ["P_score_z"]], time_transform="rank")
    ph_rows = ph.summary.reset_index().rename(columns={"index": "covariate"})
    ph_rows.insert(0, "cancer", cancer)
    return independence, ph_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="configs/analysis_plan.json")
    ap.add_argument("--out", default="results/tcga_final")
    ap.add_argument("--smoke", action="store_true", help="20 patients/cancer, 2 repeats")
    args = ap.parse_args()
    out_root = Path(args.out)
    out = out_root / "survival"
    out.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(out_root / "logs" / "54_tcga_survival.log"), logging.StreamHandler()],
    )
    plan = json.loads(Path(args.plan).read_text())
    panels = {c: plan["panels"][cfg["head"]] for c, cfg in plan["cohorts"].items()}
    seeds = plan["survival"]["repeat_seeds"][:2] if args.smoke else plan["survival"]["repeat_seeds"]
    pca_k = plan["bottleneck_U3"]["pca_components"]
    features, cohort = out_root / "features", out_root / "cohort"

    all_cindex, all_alpha, all_coef, all_pcavar, all_folds = [], [], [], [], []
    oof_P_full = {}
    independence_rows, ph_rows = [], []
    oof_rows = []
    deviations: list[str] = []

    def flatten_oof(cancer, cohort_tag, oof_dict):
        for fset, per_repeat in oof_dict.items():
            for r, pct_by_patient in per_repeat.items():
                for patient, pct in pct_by_patient.items():
                    oof_rows.append(
                        {"cancer": cancer, "cohort": cohort_tag, "feature_set": fset,
                         "repeat": r, "patient": patient, "percentile": pct}
                    )

    for project, cancer in PROJECTS.items():
        panel = panels[cancer]
        df_full = load_frame(cancer, project, features, cohort)
        if args.smoke:
            df_full = pd.concat(
                [df_full[df_full["event"] == e].head(10) for e in (0, 1)], ignore_index=True
            )
        cindex, alpha, coef, pcavar, oof, tau, folds = run_cohort(
            cancer, df_full, panel, FULL_SETS, "full", seeds, pca_k, out, deviations
        )
        all_cindex.append(cindex); all_alpha.append(alpha); all_coef.append(coef)
        all_pcavar.append(pcavar); all_folds.append(folds)
        oof_P_full[cancer] = oof.get("C+P", {})
        flatten_oof(cancer, "full", oof)

        df_rna = df_full[df_full["has_rna"]].reset_index(drop=True)
        if len(df_rna) >= 20:
            cindex_r, alpha_r, coef_r, pcavar_r, oof_r, tau_r, folds_r = run_cohort(
                cancer, df_rna, panel, RNA_SETS, "rna", seeds, pca_k, out, deviations
            )
            all_cindex.append(cindex_r); all_alpha.append(alpha_r); all_coef.append(coef_r)
            all_pcavar.append(pcavar_r); all_folds.append(folds_r)
            flatten_oof(cancer, "rna", oof_r)

        ind, ph = independence_and_ph(cancer, df_full, oof_P_full[cancer])
        independence_rows.append(ind)
        ph_rows.append(ph)
        log.info("%s done", cancer)

    pd.DataFrame(oof_rows).to_csv(out / "oof_percentiles.csv", index=False)

    if deviations:
        dev_path = out_root / "DEVIATIONS.md"
        existing = dev_path.read_text() if dev_path.exists() else "# Deviations\n"
        dev_path.write_text(
            existing.rstrip("\n") + "\n\n## 2026-10-10 -- Step 5 tau lowered by Step 5.4's rule\n\n"
            + "\n".join(deviations) + "\n"
        )
        log.info("appended %d tau-lowering deviation(s) to DEVIATIONS.md", len(deviations))

    cindex_all = pd.concat(all_cindex, ignore_index=True)
    cindex_all.to_csv(out / "cindex_by_split.csv", index=False)
    pd.concat(all_alpha, ignore_index=True).to_csv(out / "chosen_alpha.csv", index=False)
    pd.concat(all_coef, ignore_index=True).to_csv(out / "coefficients.csv", index=False)
    pd.concat(all_pcavar, ignore_index=True).to_csv(out / "pca_variance.csv", index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(out / "cv_folds.csv", index=False)
    pd.DataFrame(independence_rows).to_csv(out / "independence.csv", index=False)
    pd.concat(ph_rows, ignore_index=True).to_csv(out / "ph_tests.csv", index=False)

    primary = delta_table(cindex_all, [("C+P", "C", ["BRCA"])])
    primary.to_csv(out / "delta_c_primary.csv", index=False)
    secondary = delta_table(cindex_all, SECONDARY_COMPARISONS)
    secondary.to_csv(out / "delta_c_secondary.csv", index=False)
    pd.concat([primary.assign(family="primary"), secondary.assign(family="secondary")], ignore_index=True).to_csv(
        out / "delta_c.csv", index=False
    )

    u3 = delta_table(cindex_all, U3_COMPARISONS, cancers_per_comp=["BRCA", "COAD", "KIRC"])
    u3.to_csv(out / "bottleneck_summary.csv", index=False)

    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
