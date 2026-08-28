#!/usr/bin/env python
"""Steps 10-12 (Sections 13-15): audit HEST metadata for IDC/COAD/CCRCC,
run mandatory H&E<->ST alignment QC on a random spot sample per sample, and
extract ST-spot-centered 224x224 patches. Writes manifests/hest_samples.csv
and one HDF5 patch shard per sample under runs/.../hest_patches/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts.hest_prep import (HEST_SAMPLE_COLUMNS, alignment_qc_sample, audit_hest_samples,
                            enforce_patient_section_grouping, extract_hest_patches_for_sample)
from sts.utils import RunEnvironment, load_config, resolve_run_paths, save_json
from sts.wsi_patches import open_slide, write_patch_shard

# Real, on-disk layout confirmed against
# /lustre/nvwulf/projects/YurovskyGroup-nvwulf/simons_2026 (2026-08-28,
# colon cohort -- the same 23-slide CRC cohort PoC_5 used, see PoC_5.txt
# "23 slides = 23 unique patients"):
#   HEST_ROOT/copy_log_{organ_dirname}_cancer.csv  (id, metadata_organ,
#     destination_organ, disease_state, oncotree_code, st_technology,
#     image_filename, dest_h5ad, dest_wsi, ... -- NOT the generic
#     sample_id/patient_id/cancer/species/mpp schema originally assumed)
#   HEST_ROOT/{organ_dirname}/wsis/{id}.tif   (pyramidal generic-TIFF,
#     openslide-readable via the `openslide-bin` wheel -- confirmed
#     openslide.mpp-x matches 55um / scalefactors.spot_diameter_fullres
#     computed independently from the paired .h5ad, a strong cross-check)
#   HEST_ROOT/{organ_dirname}/st/{id}.h5ad    (standard scanpy/Visium-style
#     AnnData: obs has array_row/col + pxl_row/col_in_fullres, obsm['spatial']
#     = (x_px, y_px) = (pxl_col_in_fullres, pxl_row_in_fullres) already
#     filtered to in_tissue==1, X = raw counts, uns['spatial']['ST']
#     ['scalefactors']['spot_diameter_fullres'] gives the Visium 55um spot's
#     pixel diameter)
# There is no explicit patient_id column or species column: each `id` is
# one section from one patient (id doubles as patient_id, matching PoC_5's
# confirmed 23-slides=23-patients count), and every present sample is human.
ORGAN_DIRNAME = {"COAD": "colon", "IDC": "breast", "CCRCC": "kidney"}
# HEST/oncotree codes that count as this project's "COAD/colon" cohort
# (Section 2's HEST target is "COAD / colon"; the real colon copy_log
# contains COAD, COADREAD, and READ -- all colorectal, so all three map to
# the single Final_Project.txt cancer bucket "COAD" rather than being
# silently dropped or invented as extra cancer types).
ONCOTREE_TO_CANCER = {"COAD": "COAD", "COADREAD": "COAD", "READ": "COAD"}


def load_hest_raw_metadata(hest_root: Path, cancers: list[str]) -> pd.DataFrame:
    """Reads each cancer's `copy_log_{organ}_cancer.csv` (only `colon`
    exists as of 2026-08-28; missing ones are skipped and reported, exactly
    like the analogous CPTAC incremental-download handling in
    scripts/01_select_cptac_wsi.py) and normalizes it to
    sts.hest_prep.HEST_SAMPLE_COLUMNS."""
    rows = []
    for cancer in cancers:
        organ_dir = ORGAN_DIRNAME.get(cancer, cancer.lower())
        log_path = hest_root / f"copy_log_{organ_dir}_cancer.csv"
        if not log_path.exists():
            print(f"[08_prepare_hest_metadata] no copy_log for {cancer} ({organ_dir}) at {log_path} -- "
                  f"skipping (not yet available)", file=sys.stderr)
            continue
        log = pd.read_csv(log_path)
        for _, r in log.iterrows():
            mapped_cancer = ONCOTREE_TO_CANCER.get(str(r["oncotree_code"]))
            if mapped_cancer != cancer:
                continue  # e.g. this project's COAD bucket only, even though the colon log also has READ/COADREAD
            if str(r.get("disease_state", "")).strip().lower() != "cancer":
                continue
            he_path = hest_root / r["destination_organ"] / "wsis" / r["image_filename"]
            expr_path = hest_root / r["destination_organ"] / "st" / f"{r['id']}.h5ad"
            if not (he_path.exists() and expr_path.exists()):
                continue
            rows.append({
                "sample_id": str(r["id"]), "patient_id": str(r["id"]), "study": "simons_2026_yurovsky",
                "cancer": mapped_cancer, "organ": r["destination_organ"], "species": "human",
                "st_technology": r["st_technology"], "he_path": str(he_path), "expression_path": str(expr_path),
                "mpp": None,
            })
    return pd.DataFrame(rows, columns=HEST_SAMPLE_COLUMNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    hest_root = Path(cfg["paths"]["hest_root"])

    if not hest_root.exists():
        print(f"[08_prepare_hest_metadata] HEST_ROOT does not exist yet: {hest_root}\n"
              f"Expected while scratch download is in progress -- rerun once data lands.", file=sys.stderr)
        sys.exit(2)

    with RunEnvironment("08_prepare_hest_metadata", paths["logs"] / "08_prepare_hest_metadata_env.json"):
        raw_meta = load_hest_raw_metadata(hest_root, cfg["cancers"]["hest"])
        clean, report = audit_hest_samples(raw_meta, cfg)
        clean = enforce_patient_section_grouping(clean)
        save_json(report, paths["qc"] / "hest_metadata_audit.json")
        print(f"[08_prepare_hest_metadata] {report['n_initial_samples']} -> {report['n_final_samples']} samples; "
              f"by cancer: {report['by_cancer']}")

        patch_shard_dir = paths["work_dir"] / "hest_patches"
        patch_shard_dir.mkdir(parents=True, exist_ok=True)
        alignment_reports = {}
        spot_manifest_rows = []

        for _, sample in tqdm(list(clean.iterrows()), desc="HEST samples"):
            reader = open_slide(sample["he_path"])
            import anndata as ad
            adata = ad.read_h5ad(sample["expression_path"])
            coords_px = adata.obsm["spatial"]
            spot_ids = adata.obs_names.tolist()
            meta_mpp = sample.get("mpp")
            mpp = meta_mpp if (meta_mpp is not None and meta_mpp == meta_mpp) else reader.mpp  # `x == x` is False for NaN

            qc = alignment_qc_sample(reader, coords_px, spot_ids, mpp, cfg, seed=0)
            alignment_reports[sample["sample_id"]] = qc
            if qc["likely_misaligned"]:
                print(f"[08_prepare_hest_metadata] WARNING sample {sample['sample_id']} looks misaligned "
                      f"(pct_spots_with_tissue={qc['pct_spots_with_tissue']:.2f}) -- excluding", file=sys.stderr)
                continue

            patches = extract_hest_patches_for_sample(reader, spot_ids, coords_px, mpp, cfg)
            write_patch_shard(patch_shard_dir / f"{sample['sample_id']}.h5", spot_ids, patches,
                               {"x_um": coords_px[:, 0] * mpp, "y_um": coords_px[:, 1] * mpp})

            for i, sid in enumerate(spot_ids):
                spot_manifest_rows.append({"sample_id": sample["sample_id"], "patient_id": sample["patient_id"],
                                            "cancer": sample["cancer"], "spot_id": sid,
                                            "x_um": coords_px[i, 0] * mpp, "y_um": coords_px[i, 1] * mpp})

        save_json(alignment_reports, paths["qc"] / "hest_alignment_qc.json")
        clean.to_csv(paths["manifests"] / "hest_samples.csv", index=False)
        pd.DataFrame(spot_manifest_rows).to_csv(paths["manifests"] / "hest_spot_manifest.csv", index=False)
        print(f"[08_prepare_hest_metadata] wrote {paths['manifests'] / 'hest_samples.csv'} and hest_spot_manifest.csv")


if __name__ == "__main__":
    main()
