#!/usr/bin/env python
"""Step 9: lock the student. Given hyperparameters chosen from train/val
only, either (Option A) keep the validated S1 checkpoint per seed, or
(Option B) retrain on ALL CPTAC patients with the locked config. Either
way: drop the 4608-D teacher projection head and save only the 320-D
ConvNeXt V2-Atto encoder as the final pathology encoder."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathology_st.distill_train import CPTACDistillDataset, train_student
from pathology_st.student_model import DistillationStudent
from pathology_st.utils import RunEnvironment, count_parameters, load_config, resolve_run_paths, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=["A", "B"], default="A",
                     help="A: keep validated per-seed checkpoint. B: retrain on all patients with locked hparams.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    patch_manifest = pd.read_csv(paths["manifests"] / "cptac_patch_manifest.csv")
    teacher_h5 = paths["teacher_embeddings"] / "cptac_teacher_embeddings.h5"
    shard_dir = paths["work_dir"] / "cptac_patches"

    for seed in cfg["seeds"]:
        if args.mode == "A":
            ckpt_path = paths["student_checkpoints"] / f"S1_seed{seed}_best.pt"
            if not ckpt_path.exists():
                print(f"[07_lock_student] seed{seed}: no S1 checkpoint yet at {ckpt_path} -- "
                      f"skipping (Step 5 training hasn't produced this seed's best checkpoint yet)")
                continue
        with RunEnvironment(f"07_lock_student_seed{seed}", paths["logs"] / f"07_lock_student_seed{seed}_env.json"):
            if args.mode == "A":
                state = torch.load(ckpt_path, map_location=args.device)
                model = DistillationStudent(cfg, pretrained="imagenet")
                model.load_state_dict(state["model_state_dict"])
            else:
                train_ds = CPTACDistillDataset(patch_manifest, teacher_h5, shard_dir, train=True, cfg=cfg)
                val_pm = patch_manifest.sample(frac=0.05, random_state=seed)  # small holdout purely for checkpoint selection
                val_ds = CPTACDistillDataset(val_pm, teacher_h5, shard_dir, train=False, cfg=cfg)
                res = train_student("S1", train_ds, val_ds, cfg, paths["student_checkpoints"], seed, args.device)
                state = torch.load(res["checkpoint"], map_location=args.device)
                model = DistillationStudent(cfg, pretrained="imagenet")
                model.load_state_dict(state["model_state_dict"])

            encoder_state = model.strip_to_encoder()
            out_path = paths["student_checkpoints"] / f"locked_encoder_seed{seed}.pt"
            torch.save({"encoder_state_dict": encoder_state, "seed": seed, "mode": args.mode,
                        "backbone": cfg["student"]["backbone"]}, out_path)

            params = count_parameters(model.encoder)
            save_json(params, paths["results"] / f"student_encoder_param_count_seed{seed}.json")
            print(f"[07_lock_student] seed{seed}: locked encoder ({params['total_parameters']:,} params) -> {out_path}")


if __name__ == "__main__":
    main()
