"""Discovers the maximal TCGA-BRCA/KIRC/COAD cohort and pulls its clinical fields.
Run as `src/scripts/24_tcga_validation.py cases`; first stage of the TCGA external validation.

A case qualifies with an FFPE diagnostic slide, STAR-Counts RNA-seq and cBioPortal clinical data.
Metadata only: small CSVs go to `tcga.paths.persist_dir`; slides and patches are scratch-only
(see the `tcga` section of configs/base.yaml)."""

from __future__ import annotations

import argparse
from pathlib import Path

from epathst.tcga.gdc import build_tcga_cohort, fetch_cbioportal_clinical
from epathst.utils import RunEnvironment, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["tcga"]

    persist_dir = Path(tcfg["paths"]["persist_dir"])
    manifests_dir = persist_dir / tcfg["paths"]["manifests"]
    clinical_dir = persist_dir / tcfg["paths"]["clinical"]
    logs_dir = persist_dir / tcfg["paths"]["logs"]
    manifests_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    with RunEnvironment("24_tcga_case_discovery", logs_dir / "24_tcga_case_discovery_env.json"):
        for project in tcfg["projects"]:
            study = tcfg["cbioportal_study_by_project"][project]
            print(f"[tcga.cases] {project}: querying GDC (slides + RNA-seq)...")
            cohort = build_tcga_cohort(
                gdc_api=tcfg["gdc_api"],
                cbioportal_api=tcfg["cbioportal_api"],
                project=project,
                cbioportal_study=study,
                slide_type=tcfg["slide_type"],
                slide_format=tcfg["slide_format"],
                rnaseq_workflow=tcfg["rnaseq_workflow"],
            )
            out_path = manifests_dir / f"tcga_cohort_{project}.csv"
            cohort.to_csv(out_path, index=False)
            print(
                f"[tcga.cases] {project}: {len(cohort)} cases "
                f"(slide + RNA-seq + clinical) -> {out_path}"
            )

            if cohort.empty:
                continue
            print(
                f"[tcga.cases] {project}: fetching clinical fields " f"{tcfg['clinical_fields']}..."
            )
            clin = fetch_cbioportal_clinical(
                cbioportal_api=tcfg["cbioportal_api"],
                study=study,
                patient_ids=cohort["case_submitter_id"].tolist(),
                fields=tcfg["clinical_fields"],
            )
            proj_clinical_dir = clinical_dir / project
            proj_clinical_dir.mkdir(parents=True, exist_ok=True)
            clin_path = proj_clinical_dir / "clinical.csv"
            clin.to_csv(clin_path, index=False)
            print(f"[tcga.cases] {project}: clinical for {len(clin)} patients " f"-> {clin_path}")

    print("[tcga.cases] DONE")
