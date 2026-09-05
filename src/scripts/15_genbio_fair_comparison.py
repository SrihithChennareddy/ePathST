#!/usr/bin/env python
"""Extracts GenBio-PathFM (teacher) features on the same HEST patches used
for the student, cached in the same format so they can be dropped into
src/scripts/10_run_patient_cv.py's ablation loop for a like-for-like
teacher-vs-student comparison (same patients/genes/folds/normalization/
graph/metric). Superseded by the train-fold-only PCA256 comparison in
src/pathology_st/genbio_pca.py for the locked final experiment; kept for reference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathology_st.feature_cache import cache_features
from pathology_st.teacher_genbio import GenBioPathFMTeacher
from pathology_st.utils import RunEnvironment, load_config, resolve_run_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    spot_manifest = pd.read_csv(paths["manifests"] / "hest_spot_manifest.csv")  # load manifest
    patch_shard_dir = paths["work_dir"] / "hest_patches"
    out_dir = paths["student_embeddings"] / "genbio_teacher"
    out_dir.mkdir(parents=True, exist_ok=True)

    with RunEnvironment("15_genbio_fair_comparison_extract", paths["logs"] / "15_genbio_fair_comparison_env.json"):
        teacher = GenBioPathFMTeacher(cfg, device=args.device)  # load teacher
        bs = cfg["teacher"]["batch_size"]
        for sample_id, group in tqdm(spot_manifest.groupby("sample_id"), desc="GenBio features on HEST"):
            with h5py.File(patch_shard_dir / f"{sample_id}.h5", "r") as f:  # load patch shard
                shard_pids = f["patch_id"][:].astype(str)
                shard_patches = f["patches"][:]
            group = group.set_index("spot_id").loc[shard_pids].reset_index()

            feats = []
            for start in range(0, len(shard_pids), bs):
                feats.append(teacher.embed_batch(shard_patches[start:start + bs]))  # run inference
            feats = np.concatenate(feats, axis=0)
            coords = group[["x_um", "y_um"]].values
            cache_features(out_dir / f"{sample_id}.h5", shard_pids.tolist(), feats, coords)  # save features

    print(f"[15_genbio_fair_comparison] cached GenBio-teacher (4608-D) HEST features under {out_dir}")
    print("[15_genbio_fair_comparison] to compare against the student, rerun the patch-only/graph "
          "ablations with feature_source='genbio_teacher' in place of the distilled-student "
          "features, keeping every fold/target/graph identical.")


if __name__ == "__main__":
    main()
