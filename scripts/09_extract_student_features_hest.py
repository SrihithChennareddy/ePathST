#!/usr/bin/env python
"""Step 13 (Section 16): run the frozen, locked GenBio-distilled ConvNeXt
V2-Atto (per seed) on every HEST patch and cache 320-D node features with
full spot_id<->patch<->coordinate<->embedding integrity checks. Also
extracts ImageNet-only (A1) and GenBio-teacher (A8/A9, Step 24) features on
the identical patches for fair ablation/baseline comparisons."""
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
from sts.feature_cache import assert_full_alignment, cache_features, extract_student_features
from sts.student_model import ConvNeXtV2AttoStudent, load_locked_encoder
from sts.utils import RunEnvironment, load_config, resolve_run_paths


def load_expression_index(sample_id: str, hest_root: Path, hest_samples: pd.DataFrame):
    import anndata as ad
    row = hest_samples[hest_samples["sample_id"] == sample_id].iloc[0]
    adata = ad.read_h5ad(row["expression_path"])
    return adata.obs_names.astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="which locked-encoder seed to use as the primary distilled features")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    spot_manifest = pd.read_csv(paths["manifests"] / "hest_spot_manifest.csv")
    hest_samples = pd.read_csv(paths["manifests"] / "hest_samples.csv")
    patch_shard_dir = paths["work_dir"] / "hest_patches"

    encoders = {
        "distilled": load_locked_encoder(
            torch.load(paths["student_checkpoints"] / f"locked_encoder_seed{args.seed}.pt",
                       map_location=args.device)["encoder_state_dict"], args.device),
        "imagenet": ConvNeXtV2AttoStudent(pretrained="imagenet").to(args.device).eval(),
    }

    with RunEnvironment("09_extract_student_features_hest", paths["logs"] / "09_extract_student_features_hest_env.json"):
        for feat_name, encoder in encoders.items():
            out_dir = paths["student_embeddings"] / feat_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for sample_id, group in tqdm(spot_manifest.groupby("sample_id"), desc=f"HEST features ({feat_name})"):
                with h5py.File(patch_shard_dir / f"{sample_id}.h5", "r") as f:
                    shard_pids = f["patch_id"][:].astype(str)
                    shard_patches = f["patches"][:]
                group = group.set_index("spot_id").loc[shard_pids].reset_index()

                feats = extract_student_features(encoder, shard_patches, args.device)
                coords = group[["x_um", "y_um"]].values
                out_path = out_dir / f"{sample_id}.h5"
                cache_features(out_path, shard_pids.tolist(), feats, coords)

                expr_idx = load_expression_index(sample_id, Path(cfg["paths"]["hest_root"]), hest_samples)
                import pandas as _pd
                assert_full_alignment(shard_pids.tolist(), feats, coords, _pd.DataFrame(index=expr_idx))

        print(f"[09_extract_student_features_hest] wrote cached features under {paths['student_embeddings']}")


if __name__ == "__main__":
    main()
