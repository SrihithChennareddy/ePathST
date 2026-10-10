"""GDC and cBioPortal access for the TCGA cohort: file discovery, clinical pull, file download.
Held-out external cohort with no train/test overlap with HEST-1k/CPTAC or the patient_cv
experiment; used to test whether ePathST predictions track bulk RNA-seq and clinical outcomes."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

GDC_RETRY_STATUS = {429, 500, 502, 503, 504}


def _gdc_post(url: str, json_body: dict, max_retries: int = 5) -> dict:
    """POST to the GDC API with exponential backoff; bulk queries intermittently get 429."""
    for attempt in range(max_retries):
        r = requests.post(url, json=json_body, timeout=60)
        if r.status_code not in GDC_RETRY_STATUS:
            r.raise_for_status()
            return r.json()
        time.sleep(2**attempt)
    r.raise_for_status()
    return r.json()


def gdc_files_for_project(
    gdc_api: str,
    project: str,
    experimental_strategy: str,
    data_format: str | None = None,
    workflow_type: str | None = None,
    page_size: int = 2000,
) -> pd.DataFrame:
    """Paginated GDC `files` query, one row per file, with the case `submitter_id`
    (the join key to cBioPortal's `patientId`)."""
    content = [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
        {
            "op": "in",
            "content": {"field": "experimental_strategy", "value": [experimental_strategy]},
        },
    ]
    if data_format:
        content.append({"op": "in", "content": {"field": "data_format", "value": [data_format]}})
    if workflow_type:
        content.append(
            {"op": "in", "content": {"field": "analysis.workflow_type", "value": [workflow_type]}}
        )
    filters = {"op": "and", "content": content}

    rows, frm = [], 0
    while True:
        body = {
            "filters": filters,
            "fields": "file_id,file_name,file_size,md5sum,data_format,cases.submitter_id,cases.case_id",
            "size": page_size,
            "from": frm,
        }
        data = _gdc_post(f"{gdc_api}/files", body)["data"]
        for hit in data["hits"]:
            cases = hit.get("cases", [{}])
            rows.append(
                {
                    "file_id": hit["file_id"],
                    "file_name": hit["file_name"],
                    "file_size": hit.get("file_size"),
                    "md5sum": hit.get("md5sum"),
                    "case_submitter_id": cases[0].get("submitter_id") if cases else None,
                    "case_id": cases[0].get("case_id") if cases else None,
                }
            )
        total = data["pagination"]["total"]
        frm += page_size
        if frm >= total:
            break
    return pd.DataFrame(rows)


def build_tcga_cohort(
    gdc_api: str,
    cbioportal_api: str,
    project: str,
    cbioportal_study: str,
    slide_type: str,
    slide_format: str,
    rnaseq_workflow: str,
) -> pd.DataFrame:
    """One row per case that has an FFPE diagnostic slide, STAR-Counts RNA-seq and cBioPortal
    clinical data (the maximal cohort, `cohort_size: max`). Takes the first slide and first
    RNA-seq file per case; one-slide-per-patient selection by QC score happens downstream,
    as in `cptac_select.select_one_wsi_per_patient`."""
    slides = gdc_files_for_project(gdc_api, project, slide_type, data_format=slide_format)
    rnaseq = gdc_files_for_project(gdc_api, project, "RNA-Seq", workflow_type=rnaseq_workflow)

    r = requests.get(
        f"{cbioportal_api}/studies/{cbioportal_study}/patients",
        params={"projection": "ID", "pageSize": 5000},
        timeout=60,
    )
    r.raise_for_status()
    clinical_patients = {p["patientId"] for p in r.json()}

    slide_by_case = slides.drop_duplicates("case_submitter_id").set_index("case_submitter_id")
    rna_by_case = rnaseq.drop_duplicates("case_submitter_id").set_index("case_submitter_id")

    case_ids = set(slide_by_case.index) & set(rna_by_case.index) & clinical_patients
    # COAD and READ share one cBioPortal study (coadread_tcga_pan_can_atlas_2018); keep only
    # cases GDC tags under this project_id so READ cases do not enter the COAD cohort.
    case_ids &= set(slides["case_submitter_id"]) | set(rnaseq["case_submitter_id"])

    rows = []
    for cid in sorted(case_ids):
        s, rna = slide_by_case.loc[cid], rna_by_case.loc[cid]
        rows.append(
            {
                "case_submitter_id": cid,
                "project": project,
                "slide_file_id": s["file_id"],
                "slide_file_name": s["file_name"],
                "slide_file_size": s["file_size"],
                "rnaseq_file_id": rna["file_id"],
                "rnaseq_file_name": rna["file_name"],
            }
        )
    return pd.DataFrame(rows)


def fetch_cbioportal_clinical(
    cbioportal_api: str,
    study: str,
    patient_ids: list[str],
    fields: list[str],
) -> pd.DataFrame:
    """Per-patient clinical data (vital status, survival time, stage, age) for `patient_ids`
    only, in one fetch call. Returns one wide row per patient."""
    body = {"ids": patient_ids, "attributeIds": fields}
    r = requests.post(
        f"{cbioportal_api}/studies/{study}/clinical-data/fetch",
        params={"clinicalDataType": "PATIENT", "projection": "SUMMARY"},
        json=body,
        timeout=120,
    )
    r.raise_for_status()
    long_df = pd.DataFrame(r.json())
    if long_df.empty:
        return pd.DataFrame(columns=["patientId"] + fields)
    wide = long_df.pivot_table(
        index="patientId",
        columns="clinicalAttributeId",
        values="value",
        aggfunc="first",
    ).reset_index()
    keep = ["patientId"] + [f for f in fields if f in wide.columns]
    missing = [f for f in fields if f not in wide.columns]
    if missing:
        print(f"[fetch_cbioportal_clinical] {study}: fields not returned by cBioPortal: {missing}")
    return wide[keep]


def download_gdc_file(
    gdc_data_url: str, file_id: str, dest: Path, expected_size: int | None = None
) -> dict:
    """Download one GDC file, skipping it if `dest` already has `expected_size` bytes.
    The skip is a size check, not md5: hashing 1-2 GB SVS files on every resume would dominate
    runtime at ~1900 files. Open-access files need no token.
    Returns {"status": "skipped"|"downloaded"|"failed", "bytes", "elapsed_s", "error"}."""
    import time as _time

    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        return {"status": "skipped", "bytes": expected_size, "elapsed_s": 0.0, "error": None}

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = _time.time()
    try:
        with requests.get(f"{gdc_data_url}/{file_id}", stream=True, timeout=300) as r:
            r.raise_for_status()
            n = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    n += len(chunk)
        tmp.rename(dest)
        return {"status": "downloaded", "bytes": n, "elapsed_s": _time.time() - t0, "error": None}
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return {"status": "failed", "bytes": 0, "elapsed_s": _time.time() - t0, "error": str(e)}
