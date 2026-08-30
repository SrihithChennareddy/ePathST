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

# Real, on-disk layout confirmed 2026-08-29 for all three cancers, via
# `configs/base.yaml`'s `paths.hest_by_cancer` (two different provenances:
# breast's WSI/ST are used in place at the YurovskyGroup project space,
# never copied locally; colon/kidney were downloaded from HuggingFace
# MahmoodLab/hest into scratch -- see docs/INPUT_DATA.md):
#   hest_by_cancer[cancer].metadata_csv   standard HEST-1k per-organ metadata
#     (id, organ, disease_state, oncotree_code, st_technology, species
#     [breast's CSV omits `species` -- every present breast sample is human,
#     same as colon/kidney's explicit `Homo sapiens` rows], ...)
#   hest_by_cancer[cancer].wsi_dir/{id}.tif    pyramidal generic-TIFF,
#     openslide-readable via the `openslide-bin` wheel
#   hest_by_cancer[cancer].st_dir/{id}.h5ad    standard scanpy/Visium-style
#     AnnData: obs has array_row/col + pxl_row/col_in_fullres, obsm['spatial']
#     = (x_px, y_px) = (pxl_col_in_fullres, pxl_row_in_fullres) already
#     filtered to in_tissue==1, X = raw counts, uns['spatial']['ST']
#     ['scalefactors']['spot_diameter_fullres'] gives the Visium 55um spot's
#     pixel diameter
# There is no explicit patient_id column: each `id` is one section from one
# patient (id doubles as patient_id, matching PoC_5's confirmed
# 23-slides=23-patients convention for this same colon cohort family).
# HEST/oncotree codes that map onto this project's three cancer buckets.
# COAD/COADREAD/READ are all colorectal (Section 2's HEST target is
# "COAD / colon"), so all three map to the single "COAD" bucket rather than
# being silently dropped or invented as extra cancer types. SCCRCC (24/25 of
# the real kidney cohort) is a clear-cell RCC variant and maps to "CCRCC";
# PRCC (papillary RCC, 1/25 rows) is a biologically distinct RCC subtype and
# is deliberately left unmapped (excluded) rather than folded in. ILC
# (invasive lobular carcinoma, 4/125 breast rows) is likewise left unmapped
# -- this project's breast bucket is specifically IDC (invasive ductal).
ONCOTREE_TO_CANCER = {"IDC": "IDC", "COAD": "COAD", "COADREAD": "COAD", "READ": "COAD",
                       "CCRCC": "CCRCC", "SCCRCC": "CCRCC"}


def load_hest_raw_metadata(cfg: dict) -> pd.DataFrame:
    """Reads each cancer's real per-cancer HEST metadata CSV + WSI/ST
    directories from `cfg["paths"]["hest_by_cancer"]` (missing/not-yet-
    available cancers are skipped and reported, exactly like the analogous
    CPTAC incremental-download handling in scripts/01_select_cptac_wsi.py)
    and normalizes each to sts.hest_prep.HEST_SAMPLE_COLUMNS."""
    sources = cfg["paths"].get("hest_by_cancer", {})
    rows = []
    for cancer in cfg["cancers"]["hest"]:
        src = sources.get(cancer)
        if src is None or not Path(src["metadata_csv"]).exists():
            print(f"[08_prepare_hest_metadata] no HEST metadata source configured/found for {cancer} -- "
                  f"skipping (not yet available)", file=sys.stderr)
            continue
        wsi_dir, st_dir = Path(src["wsi_dir"]), Path(src["st_dir"])
        meta = pd.read_csv(src["metadata_csv"])
        for _, r in meta.iterrows():
            mapped_cancer = ONCOTREE_TO_CANCER.get(str(r["oncotree_code"]))
            if mapped_cancer != cancer:
                continue  # e.g. drops the 4 ILC / 1 PRCC rows described above
            if str(r.get("disease_state", "")).strip().lower() != "cancer":
                continue
            sample_id = str(r["id"])
            he_path = wsi_dir / f"{sample_id}.tif"
            expr_path = st_dir / f"{sample_id}.h5ad"
            if not (he_path.exists() and expr_path.exists()):
                continue
            rows.append({
                "sample_id": sample_id, "patient_id": sample_id, "study": src.get("study", cancer.lower()),
                "cancer": mapped_cancer, "organ": str(r.get("organ", cancer)), "species": str(r.get("species", "human")),
                "st_technology": r.get("st_technology"), "he_path": str(he_path), "expression_path": str(expr_path),
                "mpp": None,
            })
    return pd.DataFrame(rows, columns=HEST_SAMPLE_COLUMNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    sources = cfg["paths"].get("hest_by_cancer", {})

    missing = [c for c in cfg["cancers"]["hest"] if not Path(sources.get(c, {}).get("wsi_dir", "")).exists()]
    if len(missing) == len(cfg["cancers"]["hest"]):
        print(f"[08_prepare_hest_metadata] no configured hest_by_cancer WSI directory exists yet for any of "
              f"{cfg['cancers']['hest']} -- expected while scratch download is in progress, rerun once data lands.",
              file=sys.stderr)
        sys.exit(2)
    if missing:
        print(f"[08_prepare_hest_metadata] WSI directory not found for {missing} -- will be skipped, "
              f"reported by load_hest_raw_metadata below", file=sys.stderr)

    with RunEnvironment("08_prepare_hest_metadata", paths["logs"] / "08_prepare_hest_metadata_env.json"):
        raw_meta = load_hest_raw_metadata(cfg)
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
                # `spot_id` (bare) is only unique WITHIN one sample's own patch shard/feature-cache file
                # (both are per-sample HDF5s, so a bare lookup is always correctly scoped there). HEST
                # spot ids are Visium array-grid coordinates like "010x016", not global barcodes -- they
                # collide heavily ACROSS different samples/slides (confirmed live, 2026-08-29: two real
                # breast samples SPA142/SPA137 shared 144/176 spot_ids). Any code that pools multiple
                # samples' spots together (scripts/10_run_patient_cv.py's per-fold expression/feature
                # tables, which always span multiple patients) must join on `spot_uid`, never bare
                # `spot_id`, or it silently misattributes coordinates/features/targets across patients.
                spot_manifest_rows.append({"sample_id": sample["sample_id"], "patient_id": sample["patient_id"],
                                            "cancer": sample["cancer"], "spot_id": sid,
                                            "spot_uid": f"{sample['sample_id']}::{sid}",
                                            "x_um": coords_px[i, 0] * mpp, "y_um": coords_px[i, 1] * mpp})

        save_json(alignment_reports, paths["qc"] / "hest_alignment_qc.json")
        clean.to_csv(paths["manifests"] / "hest_samples.csv", index=False)
        pd.DataFrame(spot_manifest_rows).to_csv(paths["manifests"] / "hest_spot_manifest.csv", index=False)
        print(f"[08_prepare_hest_metadata] wrote {paths['manifests'] / 'hest_samples.csv'} and hest_spot_manifest.csv")


if __name__ == "__main__":
    main()
