#!/usr/bin/env python
"""Step 2 of FINAL_GUIDE.md: one row per patient per cancer with slide, RNA, clinical, purity and
receptor fields joined on the 12-character barcode. Outcomes are joined but not analysed here."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SRIHITH = Path("/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith")
COMPANION = SRIHITH / "tcga" / "companion_processed"
PREDICTIONS = SRIHITH / "tcga" / "predictions" / "hex100_leiden_official_hvg50"
HEST_META = SRIHITH / "misc" / "hest_metadata" / "HEST_v1_3_0.csv"
CPTAC_WSI = (
    SRIHITH / "cptac_hoptimus0_distill_hest_pna_work" / "manifests" / "cptac_selected_wsi.csv"
)
PROJECTS = {"TCGA-BRCA": "BRCA", "TCGA-COAD": "COAD", "TCGA-KIRC": "KIRC"}
ENDPOINT = {"BRCA": "PFI", "COAD": "PFI", "KIRC": "OS"}

log = logging.getLogger("51_tcga_cohort")

STAGE_LOW = {"Stage I", "Stage IA", "Stage IB", "Stage II", "Stage IIA", "Stage IIB", "Stage IIC"}
GRADE_MAP = {"G1": 1, "G2": 2, "G3": 3, "G4": 4}


def stage_group(s: str) -> float:
    if pd.isna(s):
        return np.nan
    s = s.strip()
    if s in STAGE_LOW:
        return 0.0
    if s.startswith("Stage III") or s.startswith("Stage IV"):
        return 1.0
    return np.nan  # Stage X, [Not Available], [Discrepancy], [Unknown], empty


def candidate_patients() -> pd.DataFrame:
    """One row per patient per project: Step 1's slide status (any predicted-DX attempt)."""
    rows = []
    for project in PROJECTS:
        qc = pd.read_csv(PREDICTIONS / "qc_all.csv")
        qc = qc[qc["project"] == project][["project", "case", "status"]]
        rows.append(qc)
    df = pd.concat(rows, ignore_index=True)
    df["cancer"] = df["project"].map(PROJECTS)
    df["step1_eligible"] = df["status"] == "ok"
    return df


def join_rna(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for project, cancer in PROJECTS.items():
        rna = pd.read_csv(COMPANION / "rnaseq_processed" / f"{project}_panel50_log2tpm1.csv")
        rna = rna.rename(columns={"case": "patient"}).assign(cancer=cancer)
        frames.append(rna)
    rna_all = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"case": "patient"})
    df["has_rna"] = df["patient"].isin(set(rna_all["patient"]))
    return df, rna_all


def join_clinical(df: pd.DataFrame) -> pd.DataFrame:
    cdr = pd.read_csv(COMPANION / "clinical_tcga_cdr" / "TCGA-CDR.csv")
    cdr = cdr.rename(columns={"bcr_patient_barcode": "patient", "type": "cancer"})
    cdr["age_per_10y"] = cdr["age_at_initial_pathologic_diagnosis"] / 10.0
    cdr["stage_III_IV"] = cdr["ajcc_pathologic_tumor_stage"].map(stage_group)
    cdr["grade_kirc"] = cdr["histological_grade"].map(GRADE_MAP)
    keep = [
        "patient", "cancer", "age_per_10y", "stage_III_IV", "grade_kirc", "histological_type",
        "OS", "OS.time", "PFI", "PFI.time",
    ]
    df = df.merge(cdr[keep], on=["patient", "cancer"], how="left")
    df["endpoint"] = df["cancer"].map(ENDPOINT)
    df["event"] = np.where(df["endpoint"] == "OS", df["OS"], df["PFI"])
    df["time_days"] = np.where(df["endpoint"] == "OS", df["OS.time"], df["PFI.time"])
    return df


def join_purity(df: pd.DataFrame) -> pd.DataFrame:
    cpe = pd.read_csv(COMPANION / "purity" / "aran2015_cpe" / "cpe_aran2015.csv")
    cpe = cpe[cpe["is_primary_tumour"] & (cpe["sample_code"].astype(str).str.zfill(2) == "01")]
    cpe = cpe[["patient", "CPE"]].drop_duplicates("patient")
    return df.merge(cpe, on="patient", how="left")


def her2_call(score, fish) -> float:
    if score == 3 or fish == "Positive":
        return 1.0
    if score in (0, 1) or (score == 2 and fish == "Negative"):
        return 0.0
    return np.nan


def join_brca_receptors(df: pd.DataFrame) -> pd.DataFrame:
    rec = pd.read_csv(COMPANION / "clinical_cbioportal" / "brca_tcga" / "patient_clinical.csv")
    rec = rec.rename(columns={"patientId": "patient"})
    er = rec["ER_STATUS_BY_IHC"].map({"Positive": 1.0, "Negative": 0.0}).rename("ER")
    her2 = [her2_call(s, f) for s, f in zip(rec["HER2_IHC_SCORE"], rec["HER2_FISH_STATUS"])]
    rec = pd.DataFrame({"patient": rec["patient"], "ER": er.values, "HER2": her2})
    return df.merge(rec, on="patient", how="left")


def overlap_check() -> dict:
    """Step 2.6: any TCGA barcode recorded in HEST-1k metadata or the CPTAC distillation cohort."""
    hest = pd.read_csv(HEST_META, low_memory=False)
    hest_tcga_hits = int(
        hest.astype(str).apply(lambda c: c.str.contains("TCGA-", na=False)).any(axis=1).sum()
    )
    cptac = pd.read_csv(CPTAC_WSI)
    cptac_tcga_hits = int(cptac["patient_id"].astype(str).str.startswith("TCGA-").sum())
    return {
        "hest_rows_with_any_TCGA_barcode": hest_tcga_hits,
        "hest_rows_checked": len(hest),
        "cptac_patients_with_TCGA_id": cptac_tcga_hits,
        "cptac_patients_checked": cptac["patient_id"].nunique(),
    }


def followup_summary(df: pd.DataFrame, cancer: str) -> dict:
    surv = df[(df["cancer"] == cancer) & df["survival_eligible"]]
    n = len(surv)
    events = int(surv["event"].sum())
    kmf = KaplanMeierFitter()
    # reverse KM: censoring treated as the event, to estimate median follow-up
    kmf.fit(surv["time_days"], event_observed=1 - surv["event"])
    median_followup_days = float(kmf.median_survival_time_)
    kmf_risk = KaplanMeierFitter()
    kmf_risk.fit(surv["time_days"], event_observed=surv["event"])
    at_risk = kmf_risk.event_table["at_risk"]

    def at_risk_at(days):
        idx = at_risk.index[at_risk.index <= days]
        return int(at_risk.loc[idx.max()]) if len(idx) else np.nan

    return {
        "cancer": cancer,
        "endpoint": ENDPOINT[cancer],
        "n_eligible": n,
        "events": events,
        "event_rate": events / n if n else np.nan,
        "median_followup_years": median_followup_days / 365.25 if np.isfinite(median_followup_days) else np.nan,
        "at_risk_3y": at_risk_at(3 * 365.25),
        "at_risk_5y": at_risk_at(5 * 365.25),
    }


def consort(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cancer in PROJECTS.values():
        g = df[df["cancer"] == cancer]
        rows.append(
            {
                "cancer": cancer,
                "1_candidate": len(g),
                "2_removed_step1_qc": int((~g["step1_eligible"]).sum()),
                "after_step1": int(g["step1_eligible"].sum()),
                "3_flagged_no_rna": int((g["step1_eligible"] & ~g["has_rna"]).sum()),
                "4_survival_eligible": int((g["step1_eligible"] & g["survival_eligible"]).sum()),
                "5_IDC_only_BRCA": int((g["histological_type"] == "Infiltrating Ductal Carcinoma").sum())
                if cancer == "BRCA" else None,
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="configs/analysis_plan.json")
    ap.add_argument("--out", default="results/tcga_final")
    args = ap.parse_args()
    out = Path(args.out) / "cohort"
    out.mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(Path(args.out) / "logs" / "51_tcga_cohort.log"),
            logging.StreamHandler(),
        ],
    )
    json.loads(Path(args.plan).read_text())  # plan is read for provenance only; nothing here uses it

    df = candidate_patients()
    df, rna_all = join_rna(df)
    df = join_clinical(df)
    df = join_purity(df)
    df = join_brca_receptors(df)
    df["survival_eligible"] = (
        df["step1_eligible"]
        & df["event"].notna()
        & df["time_days"].notna()
        & (df["time_days"] > 0)
        & df["age_per_10y"].notna()
        & df["stage_III_IV"].notna()
    )

    for cancer in PROJECTS.values():
        g = df[df["cancer"] == cancer].drop(columns=["project"])
        g.to_csv(out / f"{cancer}_patients.csv", index=False)

    overlap = overlap_check()
    (out / "overlap_check.json").write_text(json.dumps(overlap, indent=1))
    log.info("overlap check: %s", overlap)

    ct = consort(df)
    ct.to_csv(out / "consort_all.csv", index=False)
    for cancer in PROJECTS.values():
        ct[ct["cancer"] == cancer].to_csv(out / f"consort_{cancer}.csv", index=False)
    log.info("CONSORT:\n%s", ct.to_string(index=False))

    fu = pd.DataFrame([followup_summary(df, c) for c in PROJECTS.values()])
    fu.to_csv(out / "followup.csv", index=False)
    log.info("follow-up summary:\n%s", fu.to_string(index=False))

    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
