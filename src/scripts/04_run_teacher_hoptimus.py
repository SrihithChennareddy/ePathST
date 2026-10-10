#!/usr/bin/env python
"""Frozen H-optimus-0 teacher embeddings for every CPTAC patch, cached once as FP16.

Stage 04 of the ePathST pipeline: the cache (cptac_teacher_embeddings.h5) holds the distillation
targets read by 05_train_student_distill.py, so the teacher never runs during student training.

Resumable (patch_ids already cached are skipped). Multi-GPU runs split WSIs across --num-shards
processes, each writing its own temp file; --merge-shards then folds them into the canonical cache
and verifies it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.models.teacher import HOptimusTeacher, cache_teacher_embeddings, verify_teacher_cache
from epathst.utils import RunEnvironment, load_config, resolve_run_paths, save_json


def _already_cached_patch_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with h5py.File(path, "r") as f:
        return set(f["patch_id"][:].astype(str))


def merge_shards(canonical_path: Path, shard_paths: list[Path]) -> None:
    """Append embeddings from the shard files to the canonical cache.

    Patch ids already in the canonical cache are skipped, so reruns are idempotent.
    """
    have = _already_cached_patch_ids(canonical_path)
    for sp in shard_paths:
        if not sp.exists():
            continue
        with h5py.File(sp, "r") as f:
            pids = f["patch_id"][:].astype(str)
            emb = f["teacher_embedding"][:]
        mask = np.array([p not in have for p in pids])
        if mask.any():
            cache_teacher_embeddings(canonical_path, pids[mask].tolist(), emb[mask])
            have.update(pids[mask].tolist())
        print(
            f"[04_run_teacher_hoptimus] merged {mask.sum()}/{len(pids)} new embeddings from "
            f"{sp.name}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--shard-index", type=int, default=0, help="0-based shard index for multi-GPU sharding"
    )
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="total number of parallel shards (e.g. 4 for H200x4)",
    )
    ap.add_argument(
        "--merge-shards",
        action="store_true",
        help="skip embedding; just merge every shard file into the canonical cache and verify",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    # The teacher is never re-run per distillation epoch; this run-once cache replaces it.
    canonical_path = paths["teacher_embeddings"] / "cptac_teacher_embeddings.h5"

    if args.merge_shards:
        shard_paths = [
            paths["teacher_embeddings"] / f"_shard{i}_of_{args.num_shards}.h5"
            for i in range(args.num_shards)
        ]
        merge_shards(canonical_path, shard_paths)
        report = verify_teacher_cache(canonical_path, cfg["teacher"]["embedding_dim"])
        save_json(report, paths["qc"] / "teacher_embedding_verification.json")
        print(f"[04_run_teacher_hoptimus] post-merge verification: {report}")
        assert report["passed"], "merged teacher embedding cache failed integrity checks"
        for sp in shard_paths:
            sp.unlink(missing_ok=True)
        print(
            f"[04_run_teacher_hoptimus] merged {args.num_shards} shard(s) into {canonical_path}, "
            f"cleaned up shard files"
        )
        return

    patch_manifest = pd.read_csv(paths["manifests"] / "cptac_patch_manifest.csv")
    shard_dir = paths["work_dir"] / "cptac_patches"

    # Each shard writes its own file so the canonical cache never has concurrent writers.
    out_path = (
        paths["teacher_embeddings"] / f"_shard{args.shard_index}_of_{args.num_shards}.h5"
        if args.num_shards > 1
        else canonical_path
    )

    # Never re-embed: skip patches in the canonical cache or in this shard's partial output.
    already_done = _already_cached_patch_ids(canonical_path) | _already_cached_patch_ids(out_path)

    wsi_ids = sorted(patch_manifest["wsi_id"].unique())
    # Shards take disjoint sets of whole WSIs, so one WSI's patches never span two shard files.
    my_wsi_ids = set(wsi_ids[args.shard_index :: args.num_shards])
    todo = patch_manifest[
        patch_manifest["wsi_id"].isin(my_wsi_ids) & ~patch_manifest["patch_id"].isin(already_done)
    ]
    print(
        f"[04_run_teacher_hoptimus] shard {args.shard_index}/{args.num_shards}: "
        f"{len(my_wsi_ids)} WSIs assigned, {len(todo)} patches left to embed "
        f"({len(already_done)} already cached)"
    )

    if len(todo) == 0:
        print("[04_run_teacher_hoptimus] nothing to do for this shard")
        return

    stage = f"04_run_teacher_hoptimus_shard{args.shard_index}"
    with RunEnvironment(stage, paths["logs"] / f"{stage}_env.json"):
        teacher = HOptimusTeacher(cfg, device=args.device)
        print(f"[04_run_teacher_hoptimus] H-optimus-0 loaded: {teacher.n_parameters:,} parameters")

        bs = cfg["teacher"]["batch_size"]
        n_embedded = 0
        for wsi_id, group in tqdm(todo.groupby("wsi_id"), desc=f"WSIs (shard {args.shard_index})"):
            with h5py.File(shard_dir / f"{wsi_id}.h5", "r") as f:
                shard_pids = f["patch_id"][:].astype(str)
                shard_patches = f["patches"][:]
            pid_to_idx = {p: i for i, p in enumerate(shard_pids)}
            group = group.sort_values("patch_id")
            idxs = [pid_to_idx[p] for p in group["patch_id"]]

            for start in range(0, len(idxs), bs):
                batch_idx = idxs[start : start + bs]
                batch_patches = shard_patches[batch_idx]
                batch_ids = [group["patch_id"].iloc[start + i] for i in range(len(batch_idx))]
                emb = teacher.embed_batch(batch_patches)
                cache_teacher_embeddings(out_path, batch_ids, emb)
                n_embedded += len(batch_ids)

        print(
            f"[04_run_teacher_hoptimus] shard {args.shard_index} embedded {n_embedded} new "
            f"patches -> {out_path}"
        )

        # Single-shard runs verify here; multi-shard runs verify after --merge-shards.
        if args.num_shards == 1:
            report = verify_teacher_cache(out_path, cfg["teacher"]["embedding_dim"])
            save_json(report, paths["qc"] / "teacher_embedding_verification.json")
            print(f"[04_run_teacher_hoptimus] verification: {report}")
            assert report["passed"], "teacher embedding cache failed integrity checks"


if __name__ == "__main__":
    main()
