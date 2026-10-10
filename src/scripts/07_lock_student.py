#!/usr/bin/env python
"""Locks the student encoder that stays frozen for all HEST downstream experiments.

Stage 07 of the ePathST pipeline: saves only the ConvNeXt V2-Atto encoder per seed as
locked_encoder_seed{seed}.pt, read by 09_extract_student_features_hest.py. Hyperparameters were
chosen on train/val only.

Mode A keeps the validated per-seed S1 checkpoint, so its CPTAC test metrics (06) stay valid.
Mode B retrains S1 on all CPTAC patients with the locked hyperparameters, leaving no CPTAC test
set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.distillation.training import CPTACDistillDataset, train_student
from epathst.models.student import DistillationStudent
from epathst.utils import (
    RunEnvironment,
    count_parameters,
    load_config,
    resolve_run_paths,
    save_json,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--mode",
        choices=["A", "B"],
        default="A",
        help="A: keep validated per-seed checkpoint. "
        "B: retrain on all patients with locked hparams.",
    )
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
                print(
                    f"[07_lock_student] seed{seed}: no S1 checkpoint yet at {ckpt_path} -- "
                    f"skipping (Step 5 training hasn't produced this seed's best checkpoint yet)"
                )
                continue
        with RunEnvironment(
            f"07_lock_student_seed{seed}", paths["logs"] / f"07_lock_student_seed{seed}_env.json"
        ):
            if args.mode == "A":
                state = torch.load(ckpt_path, map_location=args.device)
                model = DistillationStudent(cfg, pretrained="imagenet")
                model.load_state_dict(state["model_state_dict"])
            else:
                train_ds = CPTACDistillDataset(
                    patch_manifest, teacher_h5, shard_dir, train=True, cfg=cfg
                )
                # Random 5% of patches, used only to pick the best epoch. It is drawn from the
                # training set, so it is not a held-out estimate.
                val_pm = patch_manifest.sample(frac=0.05, random_state=seed)
                val_ds = CPTACDistillDataset(val_pm, teacher_h5, shard_dir, train=False, cfg=cfg)
                res = train_student(
                    "S1", train_ds, val_ds, cfg, paths["student_checkpoints"], seed, args.device
                )
                state = torch.load(res["checkpoint"], map_location=args.device)
                model = DistillationStudent(cfg, pretrained="imagenet")
                model.load_state_dict(state["model_state_dict"])

            # Ship the encoder only; the distillation projection head is not used downstream.
            encoder_state = model.strip_to_encoder()
            out_path = paths["student_checkpoints"] / f"locked_encoder_seed{seed}.pt"
            torch.save(
                {
                    "encoder_state_dict": encoder_state,
                    "seed": seed,
                    "mode": args.mode,
                    "backbone": cfg["student"]["backbone"],
                },
                out_path,
            )

            params = count_parameters(model.encoder)
            save_json(params, paths["results"] / f"student_encoder_param_count_seed{seed}.json")
            print(
                f"[07_lock_student] seed{seed}: locked encoder "
                f"({params['total_parameters']:,} params) -> {out_path}"
            )


if __name__ == "__main__":
    main()
