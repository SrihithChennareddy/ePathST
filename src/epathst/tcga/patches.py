"""Extracts tissue-masked 224x224 patches (~0.5 um/px, as in CPTAC/HEST) from TCGA slides.
Run as `src/scripts/24_tcga_validation.py patches`; one HDF5 shard per slide, CPU only.

Only slides already on disk are processed, so it can run alongside the `download` stage and be
rerun as more slides land. Patches and their manifest are pixel-derived, hence scratch only
(see the `tcga` section of configs/base.yaml)."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import pandas as pd

from epathst.data.cptac_select import tissue_mask_from_thumbnail
from epathst.data.wsi_patches import (
    open_slide,
    patch_tissue_fraction,
    sample_patch_grid,
    write_patch_shard,
)
from epathst.utils import RunEnvironment, load_config


def _extract_one(slide: dict, cfg: dict, shard_dir: str) -> list[dict]:
    shard_path = Path(shard_dir) / f"{slide['case_submitter_id']}.h5"
    try:
        reader = open_slide(slide["slide_path"])
        thumb = reader.get_thumbnail(cfg["wsi_selection"]["thumbnail_max_dim"])
        tissue_mask = tissue_mask_from_thumbnail(thumb)

        coords = sample_patch_grid(
            reader, tissue_mask, cfg, rng_seed=hash(slide["case_submitter_id"]) % (2**31)
        )
        patch_px, target_mpp = cfg["patching"]["patch_px"], cfg["patching"]["target_mpp"]

        patches, patch_ids, kept_coords = [], [], []
        for x, y in coords:
            patch = reader.read_region_at_mpp(x, y, patch_px * target_mpp, patch_px)
            tf = patch_tissue_fraction(patch)
            if tf < cfg["patching"]["min_tissue_fraction"]:
                continue
            patch_ids.append(f"{slide['case_submitter_id']}_{x}_{y}")
            patches.append(patch)
            kept_coords.append((x, y, tf))

        if hasattr(reader, "close"):
            reader.close()
        if not patches:
            return []
        import numpy as np

        patches = np.stack(patches)
        write_patch_shard(
            shard_path,
            patch_ids,
            patches,
            {"x": [c[0] for c in kept_coords], "y": [c[1] for c in kept_coords]},
        )
        return [
            {
                "patch_id": pid,
                "case_submitter_id": slide["case_submitter_id"],
                "project": slide["project"],
                "x": x,
                "y": y,
                "mpp": target_mpp,
                "tissue_fraction": tf,
            }
            for pid, (x, y, tf) in zip(patch_ids, kept_coords)
        ]
    except Exception as e:
        print(
            f"[tcga.patches] FAILED {slide['case_submitter_id']}: " f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return []


def _extract_one_star(args):
    return _extract_one(*args)


def load_available_slides(tcfg: dict, projects: list[str]) -> pd.DataFrame:
    """Slides already on disk at full size; safe while the download stage is still running."""
    persist_dir = Path(tcfg["paths"]["persist_dir"])
    scratch_dir = Path(tcfg["paths"]["scratch_dir"])
    manifests_dir = persist_dir / tcfg["paths"]["manifests"]

    rows = []
    for project in projects:
        df = pd.read_csv(manifests_dir / f"tcga_cohort_{project}.csv")
        slide_dir = scratch_dir / tcfg["paths"]["slides"] / project
        for _, r in df.iterrows():
            p = slide_dir / r["slide_file_name"]
            if p.exists() and p.stat().st_size == int(r["slide_file_size"]):
                rows.append(
                    {
                        "case_submitter_id": r["case_submitter_id"],
                        "project": project,
                        "slide_path": str(p),
                    }
                )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--projects", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument(
        "--probe-n",
        type=int,
        default=0,
        help="if >0, process only this many slides total then stop (for a timing probe)",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["tcga"]
    projects = args.projects or tcfg["projects"]

    scratch_dir = Path(tcfg["paths"]["scratch_dir"])
    meta_dir = scratch_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = scratch_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    with RunEnvironment("28_tcga_extract_patches", logs_dir / "28_tcga_extract_patches_env.json"):
        available = load_available_slides(tcfg, projects)
        print(
            f"[tcga.patches] {len(available)} slide(s) currently on disk across "
            f"{projects} (more may land as the slide download continues)"
        )
        if available.empty:
            print("[tcga.patches] nothing available yet")
            return

        for project in projects:
            shard_dir = scratch_dir / tcfg["paths"]["patches"] / project
            shard_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = meta_dir / f"{project}_patches.csv"

            proj_slides = available[available["project"] == project]
            if proj_slides.empty:
                continue

            done_ids = set()
            existing_rows = []
            if manifest_path.exists():
                existing = pd.read_csv(manifest_path)
                existing["shard_exists"] = existing["case_submitter_id"].apply(
                    lambda c: (shard_dir / f"{c}.h5").exists()
                )
                existing_rows = (
                    existing[existing["shard_exists"]]
                    .drop(columns=["shard_exists"])
                    .to_dict("records")
                )
                done_ids = {r["case_submitter_id"] for r in existing_rows}

            todo = [
                row.to_dict()
                for _, row in proj_slides.iterrows()
                if row["case_submitter_id"] not in done_ids
            ]
            if args.probe_n:
                todo = todo[: args.probe_n]
            print(
                f"[tcga.patches] {project}: {len(todo)} slide(s) to extract "
                f"({len(done_ids)} already done), {args.workers} workers"
            )
            if not todo:
                continue

            manifest_rows = list(existing_rows)
            import time

            t0 = time.time()
            with mp.get_context("spawn").Pool(args.workers) as pool:
                results_iter = pool.imap_unordered(
                    _extract_one_star, [(s, cfg, str(shard_dir)) for s in todo], chunksize=1
                )
                n_done = 0
                for rows in results_iter:
                    manifest_rows.extend(rows)
                    n_done += 1
                    if n_done % 10 == 0 or n_done == len(todo):
                        elapsed = time.time() - t0
                        rate = n_done / elapsed if elapsed > 0 else 0
                        remaining = (len(todo) - n_done) / rate / 60 if rate > 0 else float("inf")
                        print(
                            f"[tcga.patches] {project}: {n_done}/{len(todo)} slides "
                            f"({rate*60:.1f} slides/min, ~{remaining:.0f} min remaining this batch)"
                        )
                        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

            out_manifest = pd.DataFrame(manifest_rows)
            out_manifest.to_csv(manifest_path, index=False)
            n_patches = len(out_manifest)
            n_slides_done = out_manifest["case_submitter_id"].nunique() if n_patches else 0
            print(
                f"[tcga.patches] {project}: {n_patches} total patches across "
                f"{n_slides_done} slides -> {manifest_path}"
            )

    print("[tcga.patches] DONE")
