#!/usr/bin/env python
"""Step 24 (Section 28): A8/A9 -- run GenBio-PathFM itself on the SAME HEST
patches, same patients/genes/folds/normalization/graph/metric as the
student, for a like-for-like comparison of:
  1. GenBio patch-only   2. distilled-student patch-only
  3. GenBio + graph      4. distilled-student + graph
Never compares this custom split against the published HEST leaderboard."""
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
from sts.feature_cache import cache_features
from sts.teacher_genbio import GenBioPathFMTeacher
from sts.utils import RunEnvironment, load_config, resolve_run_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    spot_manifest = pd.read_csv(paths["manifests"] / "hest_spot_manifest.csv")
    patch_shard_dir = paths["work_dir"] / "hest_patches"
    out_dir = paths["student_embeddings"] / "genbio_teacher"
    out_dir.mkdir(parents=True, exist_ok=True)

    with RunEnvironment("15_genbio_fair_comparison_extract", paths["logs"] / "15_genbio_fair_comparison_env.json"):
        teacher = GenBioPathFMTeacher(cfg, device=args.device)
        bs = cfg["teacher"]["batch_size"]
        for sample_id, group in tqdm(spot_manifest.groupby("sample_id"), desc="GenBio features on HEST"):
            with h5py.File(patch_shard_dir / f"{sample_id}.h5", "r") as f:
                shard_pids = f["patch_id"][:].astype(str)
                shard_patches = f["patches"][:]
            group = group.set_index("spot_id").loc[shard_pids].reset_index()

            feats = []
            for start in range(0, len(shard_pids), bs):
                feats.append(teacher.embed_batch(shard_patches[start:start + bs]))
            feats = np.concatenate(feats, axis=0)
            coords = group[["x_um", "y_um"]].values
            cache_features(out_dir / f"{sample_id}.h5", shard_pids.tolist(), feats, coords)

    print(f"[15_genbio_fair_comparison] cached GenBio-teacher (4608-D) HEST features under {out_dir}")
    print("[15_genbio_fair_comparison] rerun scripts/10_run_patient_cv.py's ablation loop with "
          "feature_source='genbio_teacher' (A8/A9 in sts.ablations.ABLATION_GRID) using this cache "
          "in place of the distilled-student features, keeping every fold/target/graph identical, "
          "then compare against the A2/A7 rows already in results/cv_ablation_metrics.csv.")


if __name__ == "__main__":
    main()
