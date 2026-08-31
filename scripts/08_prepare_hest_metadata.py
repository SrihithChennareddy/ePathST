#!/usr/bin/env python
"""Steps 10-12: audit HEST metadata for IDC/COAD/CCRCC, run mandatory
H&E<->ST alignment QC on a random spot sample per sample, and extract
ST-spot-centered 224x224 patches. Writes manifests/hest_samples.csv and
one HDF5 patch shard per sample under runs/.../hest_patches/."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts.hest_prep import (HEST_SAMPLE_COLUMNS, alignment_qc_sample, audit_hest_samples,
                            enforce_patient_section_grouping, extract_hest_patches_for_sample)
from sts.utils import RunEnvironment, load_config, resolve_run_paths, save_json
from sts.wsi_patches import open_slide, write_patch_shard

# Per-cancer HEST layout (configs/base.yaml `paths.hest_by_cancer`; breast is used
# in place at the YurovskyGroup project space, colon/kidney came from HuggingFace
# MahmoodLab/hest -- see docs/INPUT_DATA.md):
#   metadata_csv  HEST-1k per-organ metadata (id, organ, disease_state, oncotree_code, ...)
#   wsi_dir/{id}.tif   pyramidal TIFF, openslide-readable
#   st_dir/{id}.h5ad   Visium AnnData: obsm['spatial'] = (x_px, y_px), already filtered to
#     in_tissue==1, X = raw counts, uns['spatial']['ST']['scalefactors']['spot_diameter_fullres']
# There is no explicit patient_id column -- each `id` is one section from one patient, so
# `id` doubles as `patient_id`.
#
# Oncotree -> cancer-bucket mapping: COAD/COADREAD/READ are all colorectal and collapse to
# "COAD". SCCRCC (clear-cell RCC) maps to "CCRCC"; PRCC (papillary RCC, a distinct subtype)
# and ILC (invasive lobular, distinct from this project's IDC target) are deliberately left
# unmapped and excluded.
ONCOTREE_TO_CANCER = {"IDC": "IDC", "COAD": "COAD", "COADREAD": "COAD", "READ": "COAD",
                       "CCRCC": "CCRCC", "SCCRCC": "CCRCC"}


def load_hest_raw_metadata(cfg: dict) -> pd.DataFrame:
    """Reads each cancer's HEST metadata CSV + WSI/ST directories from
    `cfg["paths"]["hest_by_cancer"]`, normalized to
    sts.hest_prep.HEST_SAMPLE_COLUMNS. Missing/not-yet-downloaded cancers
    are skipped and reported, not treated as an error."""
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
                continue  # drops ILC/PRCC and any other unmapped oncotree code
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


def _process_one_sample(sample: dict, cfg: dict, patch_shard_dir: str, n_threads: int, seed: int = 0) -> dict:
    """Runs in a worker process: alignment QC + patch extraction + shard
    write for exactly ONE HEST sample. Self-contained (opens its own
    WSI/expression handles), writes only its own `{sample_id}.h5` shard,
    and returns manifest rows only -- never the patch array, which is
    already on disk and would otherwise get pickled back needlessly."""
    import anndata as ad

    reader = open_slide(sample["he_path"])
    adata = ad.read_h5ad(sample["expression_path"])
    coords_px = adata.obsm["spatial"]
    spot_ids = adata.obs_names.tolist()
    meta_mpp = sample.get("mpp")
    mpp = meta_mpp if (meta_mpp is not None and meta_mpp == meta_mpp) else reader.mpp  # `x == x` is False for NaN

    qc = alignment_qc_sample(reader, coords_px, spot_ids, mpp, cfg, seed=seed)
    if qc["likely_misaligned"]:
        return {"sample_id": sample["sample_id"], "qc": qc, "misaligned": True, "manifest_rows": []}

    patches = extract_hest_patches_for_sample(reader, spot_ids, coords_px, mpp, cfg, n_threads=n_threads)
    write_patch_shard(Path(patch_shard_dir) / f"{sample['sample_id']}.h5", spot_ids, patches,
                       {"x_um": coords_px[:, 0] * mpp, "y_um": coords_px[:, 1] * mpp})

    manifest_rows = [{"sample_id": sample["sample_id"], "patient_id": sample["patient_id"],
                       "cancer": sample["cancer"], "spot_id": sid,
                       "spot_uid": f"{sample['sample_id']}::{sid}",
                       "x_um": coords_px[i, 0] * mpp, "y_um": coords_px[i, 1] * mpp}
                      for i, sid in enumerate(spot_ids)]
    return {"sample_id": sample["sample_id"], "qc": qc, "misaligned": False, "manifest_rows": manifest_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--sample-workers", type=int, default=4,
                     help="HEST samples (WSIs) to process concurrently, each in its own process "
                          "(default 4, sized against a ~44G mem budget: worst-case ~26k-spot sample's "
                          "in-flight patch array is ~3.8GB)")
    ap.add_argument("--threads-per-sample", type=int, default=6,
                     help="per-sample thread-pool size for concurrent per-spot WSI reads (default 6; "
                          "sample-workers x threads-per-sample should match --cpus-per-task)")
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
        qc_path = paths["qc"] / "hest_alignment_qc.json"
        manifest_path = paths["manifests"] / "hest_spot_manifest.csv"
        samples_path = paths["manifests"] / "hest_samples.csv"
        alignment_reports = json.loads(qc_path.read_text()) if qc_path.exists() else {}
        spot_manifest_rows = pd.read_csv(manifest_path).to_dict("records") if manifest_path.exists() else []
        done_samples = {r["sample_id"] for r in spot_manifest_rows}
        if done_samples:
            print(f"[08_prepare_hest_metadata] resuming: {len(done_samples)} samples already have a patch shard "
                  f"+ manifest rows from a prior run -- skipping those", file=sys.stderr)

        def checkpoint(included_samples):
            save_json(alignment_reports, qc_path)
            included_samples.to_csv(samples_path, index=False)
            pd.DataFrame(spot_manifest_rows).to_csv(manifest_path, index=False)

        # `spot_id` alone is only unique within one sample's own shard/feature-cache file -- HEST
        # spot ids are Visium array-grid coordinates (e.g. "010x016"), not global barcodes, and
        # collide heavily across samples. Anything that pools multiple samples' spots (e.g. script
        # 10's per-fold tables) must join on `spot_uid` (built the same way in `_process_one_sample`
        # above), never bare `spot_id`, or it silently misattributes data across patients.
        pending = [row.to_dict() for _, row in clean.iterrows() if row["sample_id"] not in done_samples]
        n_since_checkpoint = 0
        with ProcessPoolExecutor(max_workers=args.sample_workers) as executor:
            futures = [executor.submit(_process_one_sample, s, cfg, str(patch_shard_dir),
                                        args.threads_per_sample) for s in pending]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="HEST samples"):
                result = fut.result()
                alignment_reports[result["sample_id"]] = result["qc"]
                if result["misaligned"]:
                    print(f"[08_prepare_hest_metadata] WARNING sample {result['sample_id']} looks misaligned "
                          f"(pct_spots_with_tissue={result['qc']['pct_spots_with_tissue']:.2f}) -- excluding",
                          file=sys.stderr)
                else:
                    spot_manifest_rows.extend(result["manifest_rows"])
                    done_samples.add(result["sample_id"])
                n_since_checkpoint += 1
                if n_since_checkpoint >= 5:
                    checkpoint(clean)  # periodic checkpoint so a wall-clock cutoff mid-run loses at most a few samples
                    n_since_checkpoint = 0

        checkpoint(clean)
        print(f"[08_prepare_hest_metadata] wrote {samples_path} and {manifest_path} "
              f"({len(done_samples)}/{len(clean)} samples processed)")


if __name__ == "__main__":
    main()
