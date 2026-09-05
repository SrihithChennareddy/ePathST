#!/usr/bin/env python
"""Step 13: run the frozen, locked GenBio-distilled ConvNeXt V2-Atto (per
seed) and the off-the-shelf ImageNet-pretrained encoder (A1 baseline) on
every HEST patch, caching 320-D node features with full
spot_id<->patch<->coordinate<->embedding integrity checks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathology_st.feature_cache import assert_full_alignment, cache_features, extract_student_features
from pathology_st.student_model import ConvNeXtV2AttoStudent, load_locked_encoder
from pathology_st.utils import RunEnvironment, load_config, resolve_run_paths


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

    distilled_checkpoint = paths["student_checkpoints"] / f"locked_encoder_seed{args.seed}.pt"
    encoders = {
        "distilled": load_locked_encoder(
            torch.load(distilled_checkpoint, map_location=args.device)["encoder_state_dict"], args.device),
        "imagenet": ConvNeXtV2AttoStudent(pretrained="imagenet").to(args.device).eval(),
    }
    checkpoint_sources = {
        "distilled": str(distilled_checkpoint),
        "imagenet": "timm:convnextv2_atto:imagenet-pretrained",
    }

    with RunEnvironment("09_extract_student_features_hest", paths["logs"] / "09_extract_student_features_hest_env.json"):
        for feat_name, encoder in encoders.items():
            out_dir = paths["student_embeddings"] / feat_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for sample_id, group in tqdm(spot_manifest.groupby("sample_id"), desc=f"HEST features ({feat_name})"):
                out_path = out_dir / f"{sample_id}.h5"
                if out_path.exists():
                    continue  # resumability: a wall-clock cutoff mid-run should not redo already-cached samples
                with h5py.File(patch_shard_dir / f"{sample_id}.h5", "r") as f:
                    shard_pids = f["patch_id"][:].astype(str)
                    shard_patches = f["patches"][:]
                group = group.set_index("spot_id").loc[shard_pids].reset_index()

                feats = extract_student_features(encoder, shard_patches, args.device)
                coords = group[["x_um", "y_um"]].values
                cache_features(out_path, shard_pids.tolist(), feats, coords, encoder_name=feat_name,
                                checkpoint_source=checkpoint_sources[feat_name], sample_id=sample_id)

                expr_idx = load_expression_index(sample_id, Path(cfg["paths"]["hest_root"]), hest_samples)
                assert_full_alignment(shard_pids.tolist(), feats, coords, pd.DataFrame(index=expr_idx))

        print(f"[09_extract_student_features_hest] wrote cached features under {paths['student_embeddings']}")


if __name__ == "__main__":
    main()
