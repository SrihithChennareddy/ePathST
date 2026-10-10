#!/usr/bin/env python
"""HEST metadata audit, H&E<->ST alignment QC and spot-centered patch extraction (IDC/COAD/CCRCC).

Stage 08 of the ePathST pipeline: writes manifests/hest_samples.csv,
manifests/hest_spot_manifest.csv and one HDF5 patch shard per sample, which feed feature
extraction (09) and the patient-level CV (10). Alignment QC is mandatory; samples that look
misaligned are excluded.

Resumable: samples already in the spot manifest are skipped, and QC reports and manifests are
checkpointed every few samples.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.data.hest_prep import (
    HEST_SAMPLE_COLUMNS,
    alignment_qc_sample,
    audit_hest_samples,
    enforce_patient_section_grouping,
    extract_hest_patches_for_sample,
)
from epathst.utils import RunEnvironment, load_config, resolve_run_paths, save_json
from epathst.data.wsi_patches import open_slide, write_patch_shard

# HEST layout: docs/reports/INPUT_DATA.md. `id` doubles as `patient_id` (no separate column).
# PRCC/ILC oncotree codes are intentionally unmapped, so those samples are dropped.
ONCOTREE_TO_CANCER = {
    "IDC": "IDC",
    "COAD": "COAD",
    "COADREAD": "COAD",
    "READ": "COAD",
    "CCRCC": "CCRCC",
    "SCCRCC": "CCRCC",
}


def load_hest_raw_metadata(cfg: dict) -> pd.DataFrame:
    """Collect per-sample HEST metadata for the configured cancers.

    Keeps samples with disease_state "cancer" whose oncotree code maps to the cancer (see
    ``ONCOTREE_TO_CANCER``) and whose H&E and expression files exist. Cancers with no configured or
    downloaded metadata are skipped with a stderr message, not treated as errors.
    """
    sources = cfg["paths"].get("hest_by_cancer", {})
    rows = []
    for cancer in cfg["cancers"]["hest"]:
        src = sources.get(cancer)
        if src is None or not Path(src["metadata_csv"]).exists():
            print(
                f"[08_prepare_hest_metadata] no HEST metadata source configured/found for "
                f"{cancer} -- skipping (not yet available)",
                file=sys.stderr,
            )
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
            rows.append(
                {
                    "sample_id": sample_id,
                    "patient_id": sample_id,
                    "study": src.get("study", cancer.lower()),
                    "cancer": mapped_cancer,
                    "organ": str(r.get("organ", cancer)),
                    "species": str(r.get("species", "human")),
                    "st_technology": r.get("st_technology"),
                    "he_path": str(he_path),
                    "expression_path": str(expr_path),
                    "mpp": None,
                }
            )
    return pd.DataFrame(rows, columns=HEST_SAMPLE_COLUMNS)


def _process_one_sample(
    sample: dict, cfg: dict, patch_shard_dir: str, n_threads: int, seed: int = 0
) -> dict:
    """Run alignment QC on one HEST sample and, if it passes, write its spot patches.

    Writes ``{sample_id}.h5`` itself and returns only manifest rows, so the patch array stays on
    disk instead of being pickled back to the parent process.

    Args:
        sample: Row of the cleaned HEST sample table.
        cfg: Loaded config.
        patch_shard_dir: Directory for the per-sample patch shards.
        n_threads: Thread-pool size for concurrent per-spot WSI reads.
        seed: Seed passed to the alignment QC.

    Returns:
        Dict with ``sample_id``, ``qc``, ``misaligned`` and ``manifest_rows`` ([] if misaligned).
    """
    import anndata as ad

    reader = open_slide(sample["he_path"])
    adata = ad.read_h5ad(sample["expression_path"])
    coords_px = adata.obsm["spatial"]
    spot_ids = adata.obs_names.tolist()
    meta_mpp = sample.get("mpp")
    # Prefer the metadata MPP; fall back to the slide's when it is missing or NaN (NaN != NaN).
    mpp = meta_mpp if (meta_mpp is not None and meta_mpp == meta_mpp) else reader.mpp

    qc = alignment_qc_sample(reader, coords_px, spot_ids, mpp, cfg, seed=seed)
    if qc["likely_misaligned"]:
        return {"sample_id": sample["sample_id"], "qc": qc, "misaligned": True, "manifest_rows": []}

    patches = extract_hest_patches_for_sample(
        reader, spot_ids, coords_px, mpp, cfg, n_threads=n_threads
    )
    write_patch_shard(
        Path(patch_shard_dir) / f"{sample['sample_id']}.h5",
        spot_ids,
        patches,
        {"x_um": coords_px[:, 0] * mpp, "y_um": coords_px[:, 1] * mpp},
    )

    # spot_uid is the cross-sample key: HEST spot_ids are grid coordinates, not global barcodes,
    # so they collide across samples and joining on bare spot_id misattributes patients.
    manifest_rows = [
        {
            "sample_id": sample["sample_id"],
            "patient_id": sample["patient_id"],
            "cancer": sample["cancer"],
            "spot_id": sid,
            "spot_uid": f"{sample['sample_id']}::{sid}",
            "x_um": coords_px[i, 0] * mpp,
            "y_um": coords_px[i, 1] * mpp,
        }
        for i, sid in enumerate(spot_ids)
    ]
    return {
        "sample_id": sample["sample_id"],
        "qc": qc,
        "misaligned": False,
        "manifest_rows": manifest_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument(
        "--sample-workers",
        type=int,
        default=4,
        help="HEST samples (WSIs) to process concurrently, each in its own process "
        "(default 4, sized against a ~44G mem budget: worst-case ~26k-spot sample's "
        "in-flight patch array is ~3.8GB)",
    )
    ap.add_argument(
        "--threads-per-sample",
        type=int,
        default=6,
        help="per-sample thread-pool size for concurrent per-spot WSI reads (default 6; "
        "sample-workers x threads-per-sample should match --cpus-per-task)",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    sources = cfg["paths"].get("hest_by_cancer", {})

    missing = [
        c
        for c in cfg["cancers"]["hest"]
        if not Path(sources.get(c, {}).get("wsi_dir", "")).exists()
    ]
    if len(missing) == len(cfg["cancers"]["hest"]):
        print(
            f"[08_prepare_hest_metadata] no configured hest_by_cancer WSI directory exists "
            f"yet for any of {cfg['cancers']['hest']} -- expected while scratch download is "
            f"in progress, rerun once data lands.",
            file=sys.stderr,
        )
        sys.exit(2)
    if missing:
        print(
            f"[08_prepare_hest_metadata] WSI directory not found for {missing} -- will be skipped, "
            f"reported by load_hest_raw_metadata below",
            file=sys.stderr,
        )

    with RunEnvironment(
        "08_prepare_hest_metadata", paths["logs"] / "08_prepare_hest_metadata_env.json"
    ):
        raw_meta = load_hest_raw_metadata(cfg)
        clean, report = audit_hest_samples(raw_meta, cfg)
        clean = enforce_patient_section_grouping(clean)
        save_json(report, paths["qc"] / "hest_metadata_audit.json")
        print(
            f"[08_prepare_hest_metadata] {report['n_initial_samples']} -> "
            f"{report['n_final_samples']} samples; "
            f"by cancer: {report['by_cancer']}"
        )

        patch_shard_dir = paths["work_dir"] / "hest_patches"
        patch_shard_dir.mkdir(parents=True, exist_ok=True)
        qc_path = paths["qc"] / "hest_alignment_qc.json"
        manifest_path = paths["manifests"] / "hest_spot_manifest.csv"
        samples_path = paths["manifests"] / "hest_samples.csv"
        alignment_reports = json.loads(qc_path.read_text()) if qc_path.exists() else {}
        spot_manifest_rows = (
            pd.read_csv(manifest_path).to_dict("records") if manifest_path.exists() else []
        )
        done_samples = {r["sample_id"] for r in spot_manifest_rows}
        if done_samples:
            print(
                f"[08_prepare_hest_metadata] resuming: {len(done_samples)} samples already have a "
                "patch shard "
                f"+ manifest rows from a prior run -- skipping those",
                file=sys.stderr,
            )

        def checkpoint(included_samples):
            save_json(alignment_reports, qc_path)
            included_samples.to_csv(samples_path, index=False)
            pd.DataFrame(spot_manifest_rows).to_csv(manifest_path, index=False)

        pending = [
            row.to_dict() for _, row in clean.iterrows() if row["sample_id"] not in done_samples
        ]
        n_since_checkpoint = 0
        with ProcessPoolExecutor(max_workers=args.sample_workers) as executor:
            futures = [
                executor.submit(
                    _process_one_sample, s, cfg, str(patch_shard_dir), args.threads_per_sample
                )
                for s in pending
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="HEST samples"):
                result = fut.result()
                alignment_reports[result["sample_id"]] = result["qc"]
                if result["misaligned"]:
                    print(
                        f"[08_prepare_hest_metadata] WARNING sample {result['sample_id']} looks "
                        "misaligned "
                        f"(pct_spots_with_tissue={result['qc']['pct_spots_with_tissue']:.2f}) -- "
                        "excluding",
                        file=sys.stderr,
                    )
                else:
                    spot_manifest_rows.extend(result["manifest_rows"])
                    done_samples.add(result["sample_id"])
                n_since_checkpoint += 1
                if n_since_checkpoint >= 5:
                    # Periodic checkpoint: a wall-clock cutoff loses at most a few samples.
                    checkpoint(clean)
                    n_since_checkpoint = 0

        checkpoint(clean)
        print(
            f"[08_prepare_hest_metadata] wrote {samples_path} and {manifest_path} "
            f"({len(done_samples)}/{len(clean)} samples processed)"
        )


if __name__ == "__main__":
    main()
