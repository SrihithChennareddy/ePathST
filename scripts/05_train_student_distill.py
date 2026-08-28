#!/usr/bin/env python
"""Steps 5-7 (Sections 8-10): train the S1 (GenBio-distilled) and, if
enabled, S2 (random-init distilled) students for every seed. S0
(ImageNet, no distillation) needs no training loop -- it is just the
off-the-shelf ImageNet-pretrained ConvNeXt V2-Atto encoder, handled
directly by scripts/06_eval_distillation.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts.distill_train import CPTACDistillDataset, train_student
from sts.utils import RunEnvironment, load_config, resolve_run_paths, save_json


def main():
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override cfg.seeds (priority order)")
    ap.add_argument("--baselines", nargs="+", default=None, choices=["S1", "S2"],
                     help="override which baselines to train (priority order); default both, S1 first")
    ap.add_argument("--time-budget-hours", type=float, default=None,
                     help="total wall-clock budget across every (baseline, seed) run below; the LAST "
                          "run in priority order gets whatever remains, so put the most important "
                          "baseline/seed first -- S1 seed0 is the main Section 10 comparison target")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    patch_manifest = pd.read_csv(paths["manifests"] / "cptac_patch_manifest.csv")
    teacher_h5 = paths["teacher_embeddings"] / "cptac_teacher_embeddings.h5"
    shard_dir = paths["work_dir"] / "cptac_patches"

    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    baseline_order = args.baselines if args.baselines is not None else ["S1", "S2"]
    enabled_map = {"S1": cfg["baselines"]["S1_genbio_distilled"], "S2": cfg["baselines"]["S2_random_init_distilled"]}
    runs = [(b, s) for s in seeds for b in baseline_order if enabled_map.get(b)]

    deadline = time.time() + args.time_budget_hours * 3600 if args.time_budget_hours else None
    results = {}
    for baseline, seed in runs:
        if deadline is not None and time.time() >= deadline:
            print(f"[05_train_student_distill] time budget exhausted before starting {baseline} seed{seed} -- skipping")
            continue
        remaining = (deadline - time.time()) if deadline is not None else None

        split_df = pd.read_csv(paths["manifests"] / f"cptac_split_seed{seed}.csv")
        merged = patch_manifest.merge(split_df[["patient_id", "split"]], on="patient_id", how="inner")
        train_pm = merged[merged["split"] == "train"]
        val_pm = merged[merged["split"] == "val"]
        train_ds = CPTACDistillDataset(train_pm, teacher_h5, shard_dir, train=True, cfg=cfg)
        val_ds = CPTACDistillDataset(val_pm, teacher_h5, shard_dir, train=False, cfg=cfg)

        stage = f"05_train_student_{baseline}_seed{seed}"
        with RunEnvironment(stage, paths["logs"] / f"{stage}_env.json"):
            res = train_student(baseline, train_ds, val_ds, cfg, paths["student_checkpoints"], seed, args.device,
                                 time_budget_seconds=remaining)
        results[f"{baseline}_seed{seed}"] = res
        print(f"[05_train_student_distill] {baseline} seed{seed}: best_epoch={res['best_epoch']} "
              f"val_cos={res['best_val_cosine_sim']:.4f} stopped_early_for_time={res['stopped_early_for_time']}")

    save_json(results, paths["results"] / "distillation_training_summary.json")


if __name__ == "__main__":
    main()
