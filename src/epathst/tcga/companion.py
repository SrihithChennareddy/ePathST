"""Downloads and organizes TCGA companion data (RNA-seq, clinical, purity) for the BRCA / COAD /
KIRC zero-shot validation cohort (1,962 patients with an FFPE diagnostic slide).
Run as `src/scripts/24_tcga_validation.py companion`; resumable, every step is idempotent.

Raw data go to scratch, small derived tables are mirrored to project space.

  scratch  /lustre/nvwulf/scratch/schennareddy/TCGA/companion/
    rnaseq/raw/<project>/      one md5-verified GDC STAR-Counts file per patient (primary tumour)
    rnaseq/processed/          all-gene TPM/count matrices (.npz), 50-gene panel tables
                               (log2(TPM+1), TPM)
    clinical/tcga_cdr/         TCGA-CDR (Liu et al., Cell 2018) xlsx + csv of the sheets used
    clinical/cbioportal/<study>/  patient / sample clinical data + attribute list
                               (PanCanAtlas + provisional studies)
    clinical/gdc/              GDC case-level clinical fields (cross-check)
    purity/aran2015_cpe/       Aran et al., Nat Commun 2015 Supplementary Data 1
                               (ESTIMATE/ABSOLUTE/LUMP/IHC/CPE)
    manifests/                 gdc_rnaseq_all_candidates.csv, rnaseq_selection.csv,
                               DOWNLOAD_MANIFEST.csv, COVERAGE.*
  project  .../srihith/tcga/companion_processed/   mirror of everything small

Panel genes are matched by exact symbol; the only exception is a documented HGNC rename (H3F3A ->
H3-3A), recorded per gene in `rnaseq/processed/<project>_panel_gene_mapping.csv`. Missing genes stop
the run.

RNA selection: primary tumour only (GDC sample_type == 'Primary Tumor'); the old cohort manifest
took the first file per patient without checking sample type. With several candidates, prefer vial
'A', then the lowest aliquot barcode; all candidates are kept in rnaseq_selection.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

GDC = "https://api.gdc.cancer.gov"
CBIO = "https://www.cbioportal.org/api"
ROOT = Path("/lustre/nvwulf/scratch/schennareddy/TCGA/companion")
MIRROR = Path("/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/tcga/companion_processed")
COHORT = Path("/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/tcga/manifests")
BENCH = Path("/lustre/nvwulf/scratch/schennareddy/HEST-1k/hest_bench")
PROJECTS = ["TCGA-BRCA", "TCGA-COAD", "TCGA-KIRC"]
PANEL_OF = {"TCGA-BRCA": "IDC", "TCGA-COAD": "COAD", "TCGA-KIRC": "CCRCC"}
CBIO_STUDIES = {
    "TCGA-BRCA": ["brca_tcga_pan_can_atlas_2018", "brca_tcga"],
    "TCGA-COAD": ["coadread_tcga_pan_can_atlas_2018", "coadread_tcga"],
    "TCGA-KIRC": ["kirc_tcga_pan_can_atlas_2018", "kirc_tcga"],
}
CDR_URL = f"{GDC}/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81"
CPE_URL = (
    "https://static-content.springer.com/esm/art%3A10.1038%2Fncomms9971/MediaObjects/"
    "41467_2015_BFncomms9971_MOESM1236_ESM.xlsx"
)
# HGNC renames between HEST panel symbols and GENCODE v36 (GDC STAR pipeline). Tried only after an
# exact-symbol match fails; every match is written to panel_gene_mapping.csv.
ALIASES = {"H3F3A": "H3-3A"}
SESSION = requests.Session()
LOG: list[dict] = []


def log(msg):
    print(f"[companion {datetime.now():%H:%M:%S}] {msg}", flush=True)


def get(url, **kw):
    for attempt in range(6):
        try:
            r = SESSION.get(url, timeout=120, **kw)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(2**attempt)


def post(url, **kw):
    for attempt in range(6):
        try:
            r = SESSION.post(url, timeout=180, **kw)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(2**attempt)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def download(
    url: str, dest: Path, expect_md5: str | None = None, expect_size: int | None = None
) -> dict:
    """Atomic, resumable (skips a file that already verifies), md5-checked against the GDC when
    given.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expect_size is None or dest.stat().st_size == expect_size):
        return {"path": str(dest), "bytes": dest.stat().st_size, "status": "present"}
    tmp = dest.with_name(f".tmp.{uuid.uuid4().hex[:8]}.{dest.name}")
    md5 = hashlib.md5()
    for attempt in range(5):
        try:
            with SESSION.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        md5.update(chunk)
            if expect_md5 and md5.hexdigest() != expect_md5:
                raise IOError(f"md5 mismatch {md5.hexdigest()} != {expect_md5}")
            os.replace(tmp, dest)
            return {"path": str(dest), "bytes": dest.stat().st_size, "status": "downloaded"}
        except Exception as e:
            md5 = hashlib.md5()
            if tmp.exists():
                tmp.unlink()
            if attempt == 4:
                return {
                    "path": str(dest),
                    "bytes": 0,
                    "status": f"FAILED {type(e).__name__}: {e}"[:200],
                }
            time.sleep(2**attempt)


def cohort_cases() -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_csv(COHORT / f"tcga_cohort_{p}.csv")[
                ["case_submitter_id", "project", "rnaseq_file_id"]
            ]
            for p in PROJECTS
        ],
        ignore_index=True,
    )


# RNA-seq
def step_rnaseq_select():
    rows = []
    for proj in PROJECTS:
        f = {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.project.project_id", "value": proj}},
                {
                    "op": "=",
                    "content": {"field": "data_type", "value": "Gene Expression Quantification"},
                },
                {
                    "op": "=",
                    "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"},
                },
                {"op": "=", "content": {"field": "access", "value": "open"}},
            ],
        }
        r = get(
            f"{GDC}/files",
            params={
                "filters": json.dumps(f),
                "size": 5000,
                "format": "JSON",
                "fields": "file_id,file_name,md5sum,file_size,cases.submitter_id,"
                "cases.samples.sample_type,cases.samples.submitter_id,"
                "cases.samples.portions.analytes.aliquots.submitter_id",
            },
        ).json()
        for h in r["data"]["hits"]:
            c = h["cases"][0]
            s = c["samples"][0]
            aliq = [
                a["submitter_id"]
                for p in s.get("portions", [])
                for an in p.get("analytes", [])
                for a in an.get("aliquots", [])
            ]
            rows.append(
                {
                    "project": proj,
                    "case": c["submitter_id"],
                    "file_id": h["file_id"],
                    "file_name": h["file_name"],
                    "md5": h["md5sum"],
                    "size": h["file_size"],
                    "sample": s["submitter_id"],
                    "sample_type": s["sample_type"],
                    "aliquot": aliq[0] if aliq else "",
                }
            )
    cand = pd.DataFrame(rows)
    (ROOT / "manifests").mkdir(parents=True, exist_ok=True)
    cand.to_csv(ROOT / "manifests" / "gdc_rnaseq_all_candidates.csv", index=False)
    log(f"GDC STAR-Counts candidates: {len(cand)} files")
    coh = cohort_cases()
    sel = []
    for _, r in coh.iterrows():
        c = cand[
            (cand["case"] == r["case_submitter_id"]) & (cand["sample_type"] == "Primary Tumor")
        ].copy()
        if c.empty:
            sel.append(
                {
                    "case": r["case_submitter_id"],
                    "project": r["project"],
                    "status": "no_primary_tumor_rnaseq",
                }
            )
            continue
        c["vial_A"] = c["sample"].str[-1] == "A"
        c = c.sort_values(["vial_A", "aliquot"], ascending=[False, True])
        ch = c.iloc[0]
        sel.append(
            {
                "case": r["case_submitter_id"],
                "project": r["project"],
                "status": "ok",
                "file_id": ch["file_id"],
                "file_name": ch["file_name"],
                "md5": ch["md5"],
                "size": int(ch["size"]),
                "sample": ch["sample"],
                "n_primary_candidates": len(c),
                "candidates": ";".join(c["sample"]),
                "old_manifest_file_id": r["rnaseq_file_id"],
                "same_as_old_manifest": ch["file_id"] == r["rnaseq_file_id"],
            }
        )
    sel = pd.DataFrame(sel)
    sel.to_csv(ROOT / "manifests" / "rnaseq_selection.csv", index=False)
    log(
        f"selected: {int((sel.status == 'ok').sum())}/{len(sel)} patients; multi-candidate "
        "patients: "
        f"{int((sel.get('n_primary_candidates', pd.Series(dtype=float)) > 1).sum())}; same file "
        "as old manifest: "
        f"{int(sel.get('same_as_old_manifest', pd.Series(dtype=bool)).fillna(False).sum())}"
    )
    return sel


def step_rnaseq_download(sel: pd.DataFrame, workers: int):
    todo = sel[sel.status == "ok"]
    log(
        f"downloading {len(todo)} STAR-Counts files ({todo['size'].sum() / 1e9:.1f} GB) with "
        f"{workers} threads"
    )
    t0, done = time.time(), [0]

    def one(r):
        res = download(
            f"{GDC}/data/{r['file_id']}",
            ROOT / "rnaseq" / "raw" / r["project"] / r["file_name"],
            expect_md5=r["md5"],
            expect_size=int(r["size"]),
        )
        res.update(
            {
                "kind": "rnaseq_star_counts",
                "case": r["case"],
                "source": f"{GDC}/data/{r['file_id']}",
                "md5_gdc": r["md5"],
            }
        )
        done[0] += 1
        if done[0] % 100 == 0:
            log(f"  {done[0]}/{len(todo)} files, {time.time() - t0:.0f}s")
        return res

    with ThreadPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(one, [r for _, r in todo.iterrows()]))
    LOG.extend(res)
    bad = [r for r in res if r["status"].startswith("FAILED")]
    log(f"RNA download done: {len(res) - len(bad)} ok, {len(bad)} failed")
    return bad


def read_star(path: Path) -> pd.DataFrame:
    d = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        usecols=["gene_id", "gene_name", "gene_type", "unstranded", "tpm_unstranded"],
    )
    return d[~d["gene_id"].str.startswith("N_")].reset_index(drop=True)


def step_rnaseq_process(sel: pd.DataFrame):
    out = ROOT / "rnaseq" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    for proj in PROJECTS:
        s = sel[(sel.project == proj) & (sel.status == "ok")].reset_index(drop=True)
        tpm, cnt, ref = [], [], None
        for _, r in s.iterrows():
            d = read_star(ROOT / "rnaseq" / "raw" / proj / r["file_name"])
            if ref is None:
                ref = d[["gene_id", "gene_name", "gene_type"]]
            assert (
                d["gene_id"].values == ref["gene_id"].values
            ).all(), f"gene order differs in {r['file_name']}"
            tpm.append(d["tpm_unstranded"].values.astype(np.float32))
            cnt.append(d["unstranded"].values.astype(np.int64))
        tpm, cnt = np.stack(tpm), np.stack(cnt)
        np.savez_compressed(
            out / f"{proj}_star_all_genes.npz",
            cases=s["case"].values,
            samples=s["sample"].values,
            gene_ids=ref["gene_id"].values,
            gene_names=ref["gene_name"].values,
            gene_types=ref["gene_type"].values,
            tpm_unstranded=tpm,
            counts_unstranded=cnt,
        )
        # one column per panel symbol; duplicated symbols resolve to the protein-coding row
        panel = json.loads((BENCH / PANEL_OF[proj] / "var_50genes.json").read_text())["genes"]
        cols, missing, mapping = {}, [], []
        for g in panel:
            rule, idx = "exact", np.where(ref["gene_name"].values == g)[0]
            if len(idx) == 0 and g in ALIASES:
                rule, idx = (
                    f"alias {g}->{ALIASES[g]}",
                    np.where(ref["gene_name"].values == ALIASES[g])[0],
                )
            if len(idx) == 0:
                missing.append(g)
                mapping.append(
                    {"panel_gene": g, "matched_gene_name": "", "gene_id": "", "rule": "MISSING"}
                )
                continue
            if len(idx) > 1:
                pc = [i for i in idx if ref["gene_type"].values[i] == "protein_coding"]
                idx = pc if pc else idx
                rule += f" (duplicate symbol, {len(idx)} candidates, protein_coding preferred)"
            cols[g] = int(idx[0])
            mapping.append(
                {
                    "panel_gene": g,
                    "matched_gene_name": ref["gene_name"].values[idx[0]],
                    "gene_id": ref["gene_id"].values[idx[0]],
                    "rule": rule,
                }
            )
        pd.DataFrame(mapping).to_csv(out / f"{proj}_panel_gene_mapping.csv", index=False)
        assert (
            not missing
        ), f"{proj}: official panel genes missing from the expression data: {missing}"
        t = pd.DataFrame({g: tpm[:, i] for g, i in cols.items()}, index=s["case"].values)
        t.index.name = "case"
        t.to_csv(out / f"{proj}_panel50_tpm.csv")
        np.log2(t + 1).to_csv(out / f"{proj}_panel50_log2tpm1.csv")
        log(
            f"{proj}: {len(s)} patients x {tpm.shape[1]} genes; panel genes found "
            f"{len(cols)}/{len(panel)}" + (f"; MISSING {missing}" if missing else "")
        )


# clinical / purity
def step_cdr_cpe():
    for url, dest in (
        (CDR_URL, ROOT / "clinical/tcga_cdr/TCGA-CDR-SupplementalTableS1.xlsx"),
        (CPE_URL, ROOT / "purity/aran2015_cpe/ncomms9971-s2_CPE.xlsx"),
    ):
        res = download(url, dest)
        res.update({"kind": "supplement", "source": url})
        LOG.append(res)
        log(f"{dest.name}: {res['status']}, {res['bytes'] / 1e6:.2f} MB")
    x = pd.ExcelFile(ROOT / "clinical/tcga_cdr/TCGA-CDR-SupplementalTableS1.xlsx")
    cdr = x.parse("TCGA-CDR")
    cdr = cdr.loc[:, ~cdr.columns.astype(str).str.startswith("Unnamed")]
    cdr.to_csv(ROOT / "clinical/tcga_cdr/TCGA-CDR.csv", index=False)
    ee = x.parse("ExtraEndpoints")
    ee.loc[:, ~ee.columns.astype(str).str.startswith("Unnamed")].to_csv(
        ROOT / "clinical/tcga_cdr/ExtraEndpoints.csv", index=False
    )
    cpe = pd.read_excel(ROOT / "purity/aran2015_cpe/ncomms9971-s2_CPE.xlsx", header=3)
    cpe["patient"] = cpe["Sample ID"].str[:12]
    cpe["sample_code"] = cpe["Sample ID"].str[13:15]
    cpe["is_primary_tumour"] = (
        cpe["sample_code"] == "01"
    )  # read sample_code with dtype=str downstream ("01" becomes 1 otherwise)
    cpe.to_csv(ROOT / "purity/aran2015_cpe/cpe_aran2015.csv", index=False)
    log(f"CDR rows {len(cdr)}, CPE rows {len(cpe)}")


def cbio_pages(path, params):
    out, page = [], 0
    while True:
        r = get(f"{CBIO}{path}", params={**params, "pageSize": 10000, "pageNumber": page}).json()
        out += r
        if len(r) < 10000:
            return out
        page += 1


def step_cbioportal():
    for proj, studies in CBIO_STUDIES.items():
        for study in studies:
            d = ROOT / "clinical/cbioportal" / study
            if (d / "DONE").exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            try:
                attrs = get(
                    f"{CBIO}/studies/{study}/clinical-attributes", params={"projection": "DETAILED"}
                ).json()
            except Exception as e:
                log(f"cBioPortal study {study}: unavailable ({type(e).__name__}); skipped")
                continue
            pd.DataFrame(attrs).to_csv(d / "attributes.csv", index=False)
            for kind in ("PATIENT", "SAMPLE"):
                rows = cbio_pages(
                    f"/studies/{study}/clinical-data",
                    {"clinicalDataType": kind, "projection": "DETAILED"},
                )
                df = pd.DataFrame(rows)
                key = "patientId" if kind == "PATIENT" else "sampleId"
                keep = [key] + (["patientId"] if kind == "SAMPLE" else [])
                wide = df.pivot_table(
                    index=keep, columns="clinicalAttributeId", values="value", aggfunc="first"
                ).reset_index()
                wide.to_csv(d / f"{kind.lower()}_clinical.csv", index=False)
                log(
                    f"cBioPortal {study} {kind}: {len(wide)} rows x {wide.shape[1] - len(keep)} "
                    "attributes"
                )
            (d / "DONE").write_text(datetime.now().isoformat())
            LOG.append(
                {
                    "kind": "cbioportal",
                    "path": str(d),
                    "source": f"{CBIO}/studies/{study}",
                    "status": "downloaded",
                    "bytes": 0,
                }
            )


def step_gdc_clinical():
    ids = cohort_cases()["case_submitter_id"].tolist()
    rows = []
    for i in range(0, len(ids), 500):
        f = {"op": "in", "content": {"field": "submitter_id", "value": ids[i : i + 500]}}
        r = post(
            f"{GDC}/cases",
            json={
                "filters": f,
                "size": 500,
                "format": "JSON",
                "fields": "submitter_id,project.project_id,demographic.gender,demographic.race,"
                "demographic.vital_status,demographic.days_to_death,diagnoses.age_at_diagnosis,"
                "diagnoses.ajcc_pathologic_stage,diagnoses.days_to_last_follow_up,"
                "diagnoses.primary_diagnosis,diagnoses.tumor_grade",
            },
        ).json()
        rows += r["data"]["hits"]
    d = ROOT / "clinical/gdc"
    d.mkdir(parents=True, exist_ok=True)
    (d / "gdc_cases_clinical.json").write_text(json.dumps(rows))
    flat = []
    for h in rows:
        dg = (h.get("diagnoses") or [{}])[0]
        dm = h.get("demographic") or {}
        flat.append(
            {
                "case": h["submitter_id"],
                "project": h["project"]["project_id"],
                "gender": dm.get("gender"),
                "race": dm.get("race"),
                "vital_status": dm.get("vital_status"),
                "days_to_death": dm.get("days_to_death"),
                "age_at_diagnosis_days": dg.get("age_at_diagnosis"),
                "ajcc_pathologic_stage": dg.get("ajcc_pathologic_stage"),
                "days_to_last_follow_up": dg.get("days_to_last_follow_up"),
                "primary_diagnosis": dg.get("primary_diagnosis"),
            }
        )
    pd.DataFrame(flat).to_csv(d / "gdc_cases_clinical.csv", index=False)
    log(f"GDC clinical: {len(flat)} cases")


# coverage / docs
def step_coverage(sel: pd.DataFrame):
    coh = cohort_cases().rename(columns={"case_submitter_id": "case"})
    cdr = pd.read_csv(ROOT / "clinical/tcga_cdr/TCGA-CDR.csv")
    cpe = pd.read_csv(ROOT / "purity/aran2015_cpe/cpe_aran2015.csv", dtype={"sample_code": str})
    cpe_tum = cpe[cpe["is_primary_tumour"]].drop_duplicates("patient")
    rows = []
    for proj in PROJECTS:
        c = coh[coh.project == proj]
        cases = set(c["case"])
        rna = set(sel[(sel.project == proj) & (sel.status == "ok")]["case"])
        cd = cdr[cdr["bcr_patient_barcode"].isin(cases)]
        row = {
            "project": proj,
            "cohort_patients": len(cases),
            "rnaseq_primary_tumour": len(rna & cases),
            "in_TCGA_CDR": len(cd),
            "OS_and_OS.time": int((cd["OS"].notna() & cd["OS.time"].notna()).sum()),
            "PFI_and_PFI.time": int((cd["PFI"].notna() & cd["PFI.time"].notna()).sum()),
            "stage_known": int(
                cd["ajcc_pathologic_tumor_stage"].astype(str).str.contains("Stage", na=False).sum()
            ),
            "age_known": int(
                pd.to_numeric(cd["age_at_initial_pathologic_diagnosis"], errors="coerce")
                .notna()
                .sum()
            ),
            "CPE_purity": int(
                cpe_tum[cpe_tum["patient"].isin(cases) & cpe_tum["CPE"].notna()].shape[0]
            ),
            "OS_events": int((cd["OS"] == 1).sum()),
            "PFI_events": int((cd["PFI"] == 1).sum()),
        }
        for study in CBIO_STUDIES[proj]:
            f = ROOT / "clinical/cbioportal" / study / "patient_clinical.csv"
            if f.exists():
                row[f"in_cBioPortal:{study}"] = int(pd.read_csv(f)["patientId"].isin(cases).sum())
        if proj == "TCGA-BRCA":
            f = ROOT / "clinical/cbioportal/brca_tcga/patient_clinical.csv"
            if f.exists():
                b = pd.read_csv(f)
                b = b[b["patientId"].isin(cases)]
                row["ER_IHC_positive"] = int((b["ER_STATUS_BY_IHC"] == "Positive").sum())
                row["ER_IHC_negative"] = int((b["ER_STATUS_BY_IHC"] == "Negative").sum())

                # positive = IHC or FISH positive; negative = IHC/FISH negative; else excluded
                def her2(r):
                    if r.get("IHC_HER2") == "Positive" or r.get("HER2_FISH_STATUS") == "Positive":
                        return "pos"
                    if r.get("IHC_HER2") == "Negative" or r.get("HER2_FISH_STATUS") == "Negative":
                        return "neg"
                    return "excluded"

                h = b.apply(her2, axis=1).value_counts()
                row["HER2_positive"], row["HER2_negative"], row["HER2_excluded"] = (
                    int(h.get("pos", 0)),
                    int(h.get("neg", 0)),
                    int(h.get("excluded", 0)),
                )
        rows.append(row)
    cov = pd.DataFrame(rows)
    cov.to_csv(ROOT / "manifests" / "COVERAGE.csv", index=False)
    # BRCA receptor attribute names in cBioPortal (values are inspected in the analysis)
    rec = []
    for study in CBIO_STUDIES["TCGA-BRCA"]:
        f = ROOT / "clinical/cbioportal" / study / "attributes.csv"
        if f.exists():
            a = pd.read_csv(f)
            m = a[
                a["clinicalAttributeId"].str.contains(
                    r"ER|HER2|PR_|PGR|IHC|FISH|ESTROGEN|PROGESTERONE", case=True, regex=True
                )
            ]
            rec += [
                {
                    "study": study,
                    "attribute": r["clinicalAttributeId"],
                    "name": r["displayName"],
                    "level": "patient" if r["patientAttribute"] else "sample",
                }
                for _, r in m.iterrows()
            ]
    pd.DataFrame(rec).to_csv(
        ROOT / "manifests" / "brca_receptor_attributes_cbioportal.csv", index=False
    )
    log("coverage:\n" + cov.T.to_string())
    return cov


def step_manifest_and_mirror(cov: pd.DataFrame):
    man = pd.DataFrame(LOG)
    old = ROOT / "manifests" / "DOWNLOAD_MANIFEST.csv"
    if old.exists():
        prev = pd.read_csv(old)
        man = (
            pd.concat([prev, man], ignore_index=True).drop_duplicates(subset=["path"], keep="last")
            if len(man)
            else prev
        )
    man["recorded"] = datetime.now().isoformat(timespec="seconds")
    man.to_csv(old, index=False)
    readme = f"""# TCGA companion data (BRCA / COAD / KIRC, 1,962-patient cohort)
Built {datetime.now():%Y-%m-%d %H:%M} by `src/scripts/24_tcga_validation.py companion` (rerunnable; skips what already verifies).
Raw data live in scratch (30-day purge); everything small is mirrored in project space (`companion_processed/`).

| Folder | Content | Source |
|---|---|---|
| `rnaseq/raw/<project>/` | one GDC STAR-Counts TSV per patient, primary tumour only, md5-verified | GDC `data` API |
| `rnaseq/processed/<project>_star_all_genes.npz` | all-gene `tpm_unstranded` and `counts_unstranded` (patients x genes), gene ids/names/types | derived |
| `rnaseq/processed/<project>_panel50_{{tpm,log2tpm1}}.csv` | the cancer's official 50-gene panel (BRCA->IDC, COAD->COAD, KIRC->CCRCC), per patient | derived |
| `clinical/tcga_cdr/` | TCGA-CDR Supplemental Table S1 (xlsx) + `TCGA-CDR.csv`, `ExtraEndpoints.csv` (OS, PFI, stage, age ...) | Liu et al., Cell 2018 (GDC publication file) |
| `purity/aran2015_cpe/` | `cpe_aran2015.csv`: ESTIMATE, ABSOLUTE, LUMP, IHC, CPE per sample (`sample_code` 01 = primary tumour) | Aran et al., Nat Commun 2015, Supp. Data 1 |
| `clinical/cbioportal/<study>/` | patient + sample clinical data and attribute lists (PanCanAtlas 2018 and provisional studies, for ER/HER2) | cBioPortal API |
| `clinical/gdc/` | GDC case-level clinical fields for the cohort (cross-check) | GDC `cases` API |
| `manifests/` | `rnaseq_selection.csv` (chosen file + all candidates per patient), `gdc_rnaseq_all_candidates.csv`, `DOWNLOAD_MANIFEST.csv`, `COVERAGE.csv`, `brca_receptor_attributes_cbioportal.csv` | |

RNA selection: GDC `sample_type == Primary Tumor` only; with several, vial A first then lowest aliquot barcode. The old cohort manifest
(`tcga/manifests/tcga_cohort_*.csv`) is not used for RNA: 992 of its file ids no longer exist at the GDC and 60 were not primary tumour.
Transform for the analysis plan: log2(TPM + 1) of `tpm_unstranded`, panel genes only. Patient barcode = first 12 characters. CPE: read `sample_code` with dtype=str (or use `is_primary_tumour`); BRCA ER/HER2 come from the provisional
cBioPortal study `brca_tcga` (ER_STATUS_BY_IHC, IHC_HER2, HER2_FISH_STATUS, HER2_IHC_SCORE); the PanCanAtlas study has no receptor fields.

## Coverage of the 1,962 cohort patients
{cov.T.to_string()}
"""
    (ROOT / "README.md").write_text(readme)
    MIRROR.mkdir(parents=True, exist_ok=True)
    for src, dst in [
        (ROOT / "README.md", MIRROR / "README.md"),
        (ROOT / "manifests", MIRROR / "manifests"),
        (ROOT / "rnaseq/processed", MIRROR / "rnaseq_processed"),
        (ROOT / "clinical/tcga_cdr", MIRROR / "clinical_tcga_cdr"),
        (ROOT / "purity", MIRROR / "purity"),
        (ROOT / "clinical/gdc", MIRROR / "clinical_gdc"),
        (ROOT / "clinical/cbioportal", MIRROR / "clinical_cbioportal"),
    ]:
        if dst.exists() and dst.is_dir():
            shutil.rmtree(dst)
        if src.is_dir():
            shutil.copytree(
                src, dst, ignore=shutil.ignore_patterns("*_all_genes.npz", "*.xlsx", "DONE")
            )
        else:
            shutil.copy(src, dst)
    log(
        f"mirror written to {MIRROR} "
        f"({sum(f.stat().st_size for f in MIRROR.rglob('*') if f.is_file()) / 1e6:.0f} MB)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-rnaseq-download", action="store_true")
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="only regenerate CDR/CPE csv, coverage, manifest, README, mirror",
    )
    args = ap.parse_args()
    for sub in (
        "rnaseq/raw",
        "rnaseq/processed",
        "clinical/tcga_cdr",
        "clinical/cbioportal",
        "clinical/gdc",
        "purity/aran2015_cpe",
        "manifests",
        "logs",
    ):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)
    if args.report_only:
        step_cdr_cpe()
        step_manifest_and_mirror(
            step_coverage(pd.read_csv(ROOT / "manifests" / "rnaseq_selection.csv"))
        )
        log("REPORT REGENERATED")
        return
    sel = step_rnaseq_select()
    if not args.skip_rnaseq_download:
        bad = step_rnaseq_download(sel, args.workers)
        if bad:
            log(f"{len(bad)} RNA downloads failed; rerun this script to retry them")
            pd.DataFrame(bad).to_csv(
                ROOT / "manifests" / "rnaseq_download_failures.csv", index=False
            )
            sys.exit(2)
    step_rnaseq_process(sel)
    step_cdr_cpe()
    step_cbioportal()
    step_gdc_clinical()
    cov = step_coverage(sel)
    step_manifest_and_mirror(cov)
    log("ALL DONE")
