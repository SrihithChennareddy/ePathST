#!/usr/bin/env python
"""Step 3 of FINAL_GUIDE.md: one 50-value predicted-gene vector per patient per system (mean,
median, SD over retained spots), one embedding vector per patient per encoder (HS, HT), and the
matching bulk RNA table. No correlations yet."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SRIHITH = Path("/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith")
PREDICTIONS = SRIHITH / "tcga" / "predictions" / "hex100_leiden_official_hvg50"
EMBEDDINGS = Path("/lustre/nvwulf/scratch/schennareddy/TCGA/embeddings_hex100")
COMPANION = SRIHITH / "tcga" / "companion_processed" / "rnaseq_processed"
PROJECTS = {"TCGA-BRCA": "BRCA", "TCGA-COAD": "COAD", "TCGA-KIRC": "KIRC"}
SYSTEM_KEY = {"E": "pred__ePathST-E", "P": "pred__ePathST", "T": "pred__H-Optimus-0"}
REBUILT = {"BRCA"}  # cancers where Step 1 rebuilt the ensemble (collapse rule)

log = logging.getLogger("52_tcga_features")


def predictions_dir(project: str, cancer: str, out: Path) -> Path:
    if cancer in REBUILT:
        return out / "predictions_rebuilt" / project
    return PREDICTIONS / project


def eligible_cases(cohort_dir: Path, cancer: str) -> pd.DataFrame:
    df = pd.read_csv(cohort_dir / f"{cancer}_patients.csv")
    return df[df["step1_eligible"]][["patient"]].reset_index(drop=True)


def gene_features(project: str, cancer: str, cases: list[str], out_root: Path, out: Path, genes_expected: list[str]):
    pdir = predictions_dir(project, cancer, out_root)
    mean_rows, median_rows, sd_rows = {s: {} for s in SYSTEM_KEY}, {s: {} for s in SYSTEM_KEY}, {s: {} for s in SYSTEM_KEY}
    genes = None
    for case in cases:
        z = np.load(pdir / f"{case}.npz")
        if genes is None:
            genes = [g for g in z["genes"]]
            assert genes == genes_expected, f"{project}: gene order mismatch vs analysis_plan panel"
        for s, key in SYSTEM_KEY.items():
            v = z[key].astype(np.float64)
            mean_rows[s][case] = v.mean(axis=0)
            median_rows[s][case] = np.median(v, axis=0)
            sd_rows[s][case] = v.std(axis=0, ddof=0)
    for s in SYSTEM_KEY:
        for kind, rows in (("mean", mean_rows[s]), ("median", median_rows[s]), ("sd", sd_rows[s])):
            df = pd.DataFrame.from_dict(rows, orient="index", columns=genes)
            df.index.name = "patient"
            df.reset_index().to_parquet(out / f"{cancer}_{s}_{kind}.parquet", index=False)
    log.info("%s: wrote mean/median/sd for E, P, T over %d patients", cancer, len(cases))


def embedding_features(project: str, cases: list[str], out: Path, cancer: str):
    hs_rows, ht_rows = {}, {}
    for case in cases:
        z = np.load(EMBEDDINGS / project / f"{case}.npz")
        hs_rows[case] = z["student"].astype(np.float64).mean(axis=0)
        ht_rows[case] = z["teacher"].astype(np.float64).mean(axis=0)
    hs = pd.DataFrame.from_dict(hs_rows, orient="index")
    hs.columns = [f"hs_{i}" for i in range(hs.shape[1])]
    hs.index.name = "patient"
    hs.reset_index().to_parquet(out / f"{cancer}_HS.parquet", index=False)
    ht = pd.DataFrame.from_dict(ht_rows, orient="index")
    ht.columns = [f"ht_{i}" for i in range(ht.shape[1])]
    ht.index.name = "patient"
    ht.reset_index().to_parquet(out / f"{cancer}_HT.parquet", index=False)
    log.info("%s: wrote HS (%dd) and HT (%dd) over %d patients", cancer, hs.shape[1], ht.shape[1], len(cases))


def gene_matching(out: Path, panels: dict) -> pd.DataFrame:
    """Reuses the per-project GDC gene-matching table already built when the companion RNA data
    was prepared (symbol / previous-symbol / alias route against the GDC STAR-Counts gene_name
    axis); reformatted to FINAL_GUIDE.md's column spec, Ensembl version suffixes stripped."""
    frames = []
    for project, cancer in PROJECTS.items():
        m = pd.read_csv(COMPANION / f"{project}_panel_gene_mapping.csv")
        m = m[~m["gene_id"].str.startswith("ENSG", na=False) | ~m["gene_id"].str.contains("_PAR_Y")]
        m["ensembl_gene_id"] = m["gene_id"].str.replace(r"\.\d+$", "", regex=True)
        m["matched"] = np.where(m["matched_gene_name"].notna(), "yes", "no")
        m["match_route"] = np.where(m["rule"] == "exact", "symbol", m["rule"])
        m["cancer"] = cancer
        frames.append(
            m.rename(columns={"panel_gene": "panel_gene", "matched_gene_name": "gdc_gene_name"})[
                ["cancer", "panel_gene", "ensembl_gene_id", "gdc_gene_name", "matched", "match_route"]
            ]
        )
    gm = pd.concat(frames, ignore_index=True)
    gm.to_csv(out / "gene_matching.csv", index=False)
    for cancer, genes in panels.items():
        missing = set(genes) - set(gm[gm["cancer"] == cancer]["panel_gene"])
        if missing:
            log.warning("%s: %d panel genes have no matching row: %s", cancer, len(missing), missing)
    return gm


def bulk_rna(out: Path, panels: dict):
    for project, cancer in PROJECTS.items():
        b = pd.read_csv(COMPANION / f"{project}_panel50_log2tpm1.csv").rename(columns={"case": "patient"})
        b = b[["patient"] + panels[cancer]]  # enforce panel order
        b.to_parquet(out / f"{cancer}_B.parquet", index=False)
    log.info("wrote bulk RNA tables (log2(TPM+1), panel order)")


def gene_properties(cohort_dir: Path, out: Path, panels: dict):
    rows = []
    for project, cancer in PROJECTS.items():
        cohort = pd.read_csv(cohort_dir / f"{cancer}_patients.csv")
        screen_eligible = set(cohort[cohort["step1_eligible"] & cohort["has_rna"]]["patient"])
        log2 = pd.read_csv(COMPANION / f"{project}_panel50_log2tpm1.csv").rename(columns={"case": "patient"})
        tpm = pd.read_csv(COMPANION / f"{project}_panel50_tpm.csv").rename(columns={"case": "patient"})
        log2 = log2[log2["patient"].isin(screen_eligible)]
        tpm = tpm[tpm["patient"].isin(screen_eligible)]
        for g in panels[cancer]:
            rows.append(
                {
                    "cancer": cancer,
                    "gene": g,
                    "n": len(log2),
                    "mean_log2tpm1": log2[g].mean(),
                    "sd_log2tpm1": log2[g].std(ddof=1),
                    "zero_share": float((tpm[g] == 0).mean()),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "gene_properties.csv", index=False)
    log.info("wrote gene_properties.csv (%d rows)", len(df))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="configs/analysis_plan.json")
    ap.add_argument("--out", default="results/tcga_final")
    args = ap.parse_args()
    out_root = Path(args.out)
    out = out_root / "features"
    out.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(out_root / "logs" / "52_tcga_features.log"),
            logging.StreamHandler(),
        ],
    )
    plan = json.loads(Path(args.plan).read_text())
    panels = {c: plan["panels"][cfg["head"]] for c, cfg in plan["cohorts"].items()}
    cohort_dir = out_root / "cohort"

    for project, cancer in PROJECTS.items():
        cases = eligible_cases(cohort_dir, cancer)["patient"].tolist()
        gene_features(project, cancer, cases, out_root, out, panels[cancer])
        embedding_features(project, cases, out, cancer)

    gene_matching(out, panels)
    bulk_rna(out, panels)
    gene_properties(cohort_dir, out, panels)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
