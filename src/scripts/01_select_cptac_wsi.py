#!/usr/bin/env python
"""Computes per-WSI QC for CPTAC WSIs in a resumable process pool, applies hard filters + Q-score
ranking to select one WSI per patient, and writes manifests/cptac_selected_wsi.csv."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathology_st.cptac_select import compute_wsi_qc, select_one_wsi_per_patient, to_manifest_rows
from pathology_st.utils import RunEnvironment, load_config, resolve_run_paths, save_json
from pathology_st.wsi_patches import open_slide

# Cancer-dir casing on disk (BRCA/COAD/ccRCC) differs from cfg.cancers.cptac's (BRCA/COAD/CCRCC).
CANCER_DIR_ALIASES = {"BRCA": "BRCA", "COAD": "COAD", "CCRCC": "ccRCC"}


def discover_candidates(
    cptac_root: Path, cptac_metadata: Path, cancers: list[str]
) -> pd.DataFrame:
    # Missing patients are expected here (incremental scratch download), not an error.
    rows = []
    n_not_yet_downloaded = 0
    for cancer in cancers:
        dir_name = CANCER_DIR_ALIASES.get(cancer, cancer)
        meta_path = cptac_metadata / f"{dir_name}_selected_WSI_per_patient.txt"
        if not meta_path.exists():
            print(f"[01_select_cptac_wsi] no metadata file for {cancer} at {meta_path}",
                  file=sys.stderr)
            continue
        meta = pd.read_csv(meta_path, sep="\t")
        # ccRCC uses `wsi_name` not `wsi_series_uid`; QC is recomputed for all three anyway so Q-score ranking stays consistent.
        id_col = "wsi_series_uid" if "wsi_series_uid" in meta.columns else "wsi_name"
        cancer_dir = cptac_root / dir_name
        for _, r in meta.iterrows():
            patient_dir = cancer_dir / str(r["patient_id"])
            dcm_files = list(patient_dir.glob("*.dcm")) if patient_dir.exists() else []
            svs_files = list(patient_dir.glob("*.svs")) if patient_dir.exists() else []
            if not dcm_files and not svs_files:
                n_not_yet_downloaded += 1
                continue
            wsi_path = str(patient_dir) if dcm_files else str(svs_files[0])
            label = str(r.get("specimen_label", ""))
            rows.append({
                "patient_id": str(r["patient_id"]), "cancer_type": cancer,
                "wsi_id": str(r[id_col]), "wsi_path": wsi_path,
                "specimen_type": label, "is_human": True, "is_he": "HE" in label.upper(),
                "is_tumor_specimen": "tumor" in label.lower(), "readable": True,
            })
    print(f"[01_select_cptac_wsi] {n_not_yet_downloaded} patient(s) listed in metadata but not yet "
          f"downloaded to CPTAC_ROOT -- skipped (rerun once more of the scratch download lands)")
    return pd.DataFrame(rows)


def _qc_one(rec: dict, cfg: dict) -> dict:
    # A read failure is captured as readable=False instead of crashing the pool.
    try:
        reader = open_slide(rec["wsi_path"])
        qc = compute_wsi_qc(rec["wsi_path"], cfg, reader)
        rec.update(qc)
        rec["readable"] = True
    except Exception as e:
        rec["readable"] = False
        rec["mpp"] = None
        rec["tissue_fraction"] = 0.0
        rec["blurry_tissue_fraction"] = 1.0
        rec["artifact_fraction"] = 1.0
        rec["qc_error"] = f"{type(e).__name__}: {e}"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                     help="process-pool size for parallel WSI QC (default: all cores minus 2)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    cptac_root, cptac_meta = Path(cfg["paths"]["cptac_root"]), Path(cfg["paths"]["cptac_metadata"])

    # bail out early if the scratch download hasn't landed yet
    if not cptac_root.exists():
        print(f"[01_select_cptac_wsi] CPTAC_ROOT does not exist yet: {cptac_root}\n"
              f"This is expected while the scratch download is in progress -- rerun this script once data lands.",
              file=sys.stderr)
        sys.exit(2)

    with RunEnvironment("01_select_cptac_wsi", paths["logs"] / "01_select_cptac_wsi_env.json"):
        # directory layout is discovered at runtime, not assumed, since the scratch download's structure can vary
        candidates = discover_candidates(cptac_root, cptac_meta, cfg["cancers"]["cptac"])
        print(f"[01_select_cptac_wsi] discovered {len(candidates)} candidate WSI rows across "
              f"{candidates['patient_id'].nunique() if len(candidates) else 0} patients")

        # load QC cache, resume
        cache_path = paths["qc"] / "wsi_qc_cache.jsonl"
        cached: dict[str, dict] = {}
        if cache_path.exists():
            with open(cache_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    cached[rec["wsi_id"]] = rec
            print(f"[01_select_cptac_wsi] resuming: {len(cached)} WSIs already QC'd in a previous run")

        todo = [row.to_dict() for _, row in candidates.iterrows() if row["wsi_id"] not in cached]
        print(f"[01_select_cptac_wsi] {len(todo)} WSIs left to QC, using {args.workers} parallel workers")

        # each completed record is appended immediately so a killed/time-limited SLURM job can resume from cache
        if todo:
            with open(cache_path, "a") as cache_f, mp.get_context("spawn").Pool(args.workers) as pool:
                results_iter = pool.imap_unordered(_qc_one_star, [(rec, cfg) for rec in todo], chunksize=1)
                n_done = 0
                for rec in results_iter:
                    cache_f.write(json.dumps(rec, default=str) + "\n")
                    cache_f.flush()
                    cached[rec["wsi_id"]] = rec
                    n_done += 1
                    if n_done % 25 == 0 or n_done == len(todo):
                        print(f"[01_select_cptac_wsi] QC progress: {n_done}/{len(todo)}")

        # select + write manifest
        records = list(cached.values())
        selected, report = select_one_wsi_per_patient(records, cfg)
        manifest_rows = to_manifest_rows(selected)
        out_csv = paths["manifests"] / "cptac_selected_wsi.csv"
        pd.DataFrame(manifest_rows).to_csv(out_csv, index=False)
        save_json(report, paths["qc"] / "cptac_wsi_selection_report.json")

        print(f"[01_select_cptac_wsi] {report['n_initial_wsis']} initial WSIs / "
              f"{report['n_initial_patients']} patients -> "
              f"{report['n_patients_passed_qc_before_target_cap']} passed QC -> "
              f"{report['n_final_wsis']} selected (target ~{cfg['wsi_selection']['target_n_patients']})")
        print(f"[01_select_cptac_wsi] wrote {out_csv}")


def _qc_one_star(args):
    return _qc_one(*args)


if __name__ == "__main__":
    main()
