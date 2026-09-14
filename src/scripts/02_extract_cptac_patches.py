#!/usr/bin/env python
"""Extracts tissue-masked 224x224 patches (~0.5um/px) per CPTAC WSI into one HDF5 shard each plus a
global manifest; resumable like script 01 (existing shards skipped on rerun)."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathology_st.utils import RunEnvironment, load_config, resolve_run_paths
from pathology_st.wsi_patches import open_slide, patch_tissue_fraction, sample_patch_grid, write_patch_shard


def _extract_one(wsi: dict, cfg: dict, shard_dir: str) -> list[dict]:
    # returns [] on failure -- logged, not fatal to the whole pool
    shard_path = Path(shard_dir) / f"{wsi['wsi_id']}.h5"
    try:
        reader = open_slide(wsi["wsi_path"])
        thumb = reader.get_thumbnail(cfg["wsi_selection"]["thumbnail_max_dim"])
        from pathology_st.cptac_select import tissue_mask_from_thumbnail
        tissue_mask = tissue_mask_from_thumbnail(thumb)

        coords = sample_patch_grid(reader, tissue_mask, cfg, rng_seed=hash(wsi["wsi_id"]) % (2 ** 31))
        patch_px, target_mpp = cfg["patching"]["patch_px"], cfg["patching"]["target_mpp"]

        patches, patch_ids, kept_coords = [], [], []
        for (x, y) in coords:
            patch = reader.read_region_at_mpp(x, y, patch_px * target_mpp, patch_px)
            tf = patch_tissue_fraction(patch)
            if tf < cfg["patching"]["min_tissue_fraction"]:
                continue
            patch_ids.append(f"{wsi['wsi_id']}_{x}_{y}")
            patches.append(patch)
            kept_coords.append((x, y, tf))

        if not patches:
            return []
        patches = np.stack(patches)
        write_patch_shard(shard_path, patch_ids, patches,
                           {"x": [c[0] for c in kept_coords], "y": [c[1] for c in kept_coords]})
        return [
            {"patch_id": pid, "patient_id": wsi["patient_id"], "cancer_type": wsi["cancer_type"],
             "wsi_id": wsi["wsi_id"], "x": x, "y": y, "mpp": target_mpp, "tissue_fraction": tf}
            for pid, (x, y, tf) in zip(patch_ids, kept_coords)
        ]
    except Exception as e:
        print(f"[02_extract_cptac_patches] FAILED {wsi['wsi_id']} ({wsi['patient_id']}): "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return []


def _extract_one_star(args):
    return _extract_one(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    selected = pd.read_csv(paths["manifests"] / "cptac_selected_wsi.csv")
    shard_dir = paths["work_dir"] / "cptac_patches"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = paths["manifests"] / "cptac_patch_manifest.csv"

    with RunEnvironment("02_extract_cptac_patches", paths["logs"] / "02_extract_cptac_patches_env.json"):
        # resume from existing manifest
        done_wsi_ids = set()
        existing_rows = []
        if manifest_path.exists():
            existing = pd.read_csv(manifest_path)
            # only trust rows whose shard file still exists on disk
            existing["shard_exists"] = existing["wsi_id"].apply(lambda w: (shard_dir / f"{w}.h5").exists())
            existing_rows = existing[existing["shard_exists"]].drop(columns=["shard_exists"]).to_dict("records")
            done_wsi_ids = {r["wsi_id"] for r in existing_rows}
            print(f"[02_extract_cptac_patches] resuming: {len(done_wsi_ids)} WSIs already extracted")

        todo = [row.to_dict() for _, row in selected.iterrows() if row["wsi_id"] not in done_wsi_ids]
        print(f"[02_extract_cptac_patches] {len(todo)} WSIs left to extract, "
              f"using {args.workers} parallel workers")

        # run extraction in parallel
        manifest_rows = list(existing_rows)
        if todo:
            with mp.get_context("spawn").Pool(args.workers) as pool:
                results_iter = pool.imap_unordered(
                    _extract_one_star, [(wsi, cfg, str(shard_dir)) for wsi in todo], chunksize=1)
                n_done = 0
                for rows in results_iter:
                    manifest_rows.extend(rows)
                    n_done += 1
                    if n_done % 25 == 0 or n_done == len(todo):
                        print(f"[02_extract_cptac_patches] extraction progress: {n_done}/{len(todo)}")
                        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)  # incremental checkpoint

        # write final manifest + summary stats
        out_manifest = pd.DataFrame(manifest_rows)
        out_manifest.to_csv(manifest_path, index=False)

        counts = out_manifest.groupby("wsi_id").size()
        print(f"[02_extract_cptac_patches] {len(out_manifest)} total patches across "
              f"{out_manifest['wsi_id'].nunique()} WSIs")
        print(f"[02_extract_cptac_patches] per-WSI patch count: min={counts.min()} "
              f"median={counts.median()} max={counts.max()}")
        print(f"[02_extract_cptac_patches] wrote {manifest_path}")


if __name__ == "__main__":
    main()
