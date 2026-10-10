#!/usr/bin/env python
"""Evaluation of how closely each student reproduces the teacher embedding on unseen patients.

Stage 06 of the ePathST pipeline: scores S0 (ImageNet, no distillation), S1 and S2 (when their
checkpoints exist) on the test patients of each seed's patient split, none of which were seen in
training (05).

Metrics, overall and per cancer type with bootstrap CIs: cosine-similarity distribution, retrieval
overlap@K and a linear-probe sanity check. UMAP coordinates are saved for the first seed only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.distillation.evaluation import (
    evaluate_distillation,
    linear_probe_sanity,
    umap_embedding,
)
from epathst.distillation.training import CPTACDistillDataset
from epathst.models.student import DistillationStudent
from epathst.utils import load_config, resolve_run_paths, save_json


@torch.no_grad()
def project_all(model, dataset, device, batch_size=256):
    """Run the student over a dataset in order (no shuffling) and return numpy arrays.

    Returns:
        Student projected embeddings, teacher embeddings and patch ids, aligned row for row.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    model.eval()
    projs, teachers, pids = [], [], []
    for x, t, pid in loader:
        out = model(x.to(device))
        projs.append(out["projected"].cpu().numpy())
        teachers.append(t.numpy())
        pids += list(pid)
    return np.concatenate(projs), np.concatenate(teachers), np.array(pids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    patch_manifest = pd.read_csv(paths["manifests"] / "cptac_patch_manifest.csv")
    teacher_h5 = paths["teacher_embeddings"] / "cptac_teacher_embeddings.h5"
    shard_dir = paths["work_dir"] / "cptac_patches"

    all_results = {}
    for seed in cfg["seeds"]:
        # Held-out test patients of this seed's patient-level split.
        split_df = pd.read_csv(paths["manifests"] / f"cptac_split_seed{seed}.csv")
        merged = patch_manifest.merge(
            split_df[["patient_id", "split", "cancer_type"]],
            on="patient_id",
            suffixes=("", "_split"),
        )
        test_pm = merged[merged["split"] == "test"]
        test_ds = CPTACDistillDataset(test_pm, teacher_h5, shard_dir, train=False, cfg=cfg)

        for baseline in ["S0", "S1", "S2"]:
            ckpt = paths["student_checkpoints"] / f"{baseline}_seed{seed}_best.pt"
            # S1/S2 are skipped for seeds whose training has not produced a best checkpoint yet.
            if baseline != "S0" and not ckpt.exists():
                continue

            # S1/S2 load trained weights over this init; S0 stays at the ImageNet weights.
            pretrained = "imagenet" if baseline in ("S0", "S1") else None
            model = DistillationStudent(cfg, pretrained=pretrained).to(args.device)
            if baseline != "S0":
                state = torch.load(ckpt, map_location=args.device)
                model.load_state_dict(state["model_state_dict"])

            proj, teacher, pids = project_all(model, test_ds, args.device)
            cancer_types = test_pm.set_index("patch_id").loc[pids, "cancer_type"].values

            report = evaluate_distillation(proj, teacher, cancer_types, cfg)
            probe = linear_probe_sanity(proj, cancer_types)
            report["linear_probe_cancer_type"] = probe
            all_results[f"{baseline}_seed{seed}"] = report
            print(
                f"[06_eval_distillation] {baseline} seed{seed}: overall mean cosine = "
                f"{report['overall']['cosine_similarity']['mean']:.4f}"
            )

            # UMAP is saved for the first seed only (qualitative figure, not a metric).
            if seed == cfg["seeds"][0]:
                umap_xy = umap_embedding(proj, seed=seed)
                np.savez(
                    paths["figures"] / f"umap_{baseline}_seed{seed}.npz",
                    xy=umap_xy,
                    cancer=cancer_types,
                )

    save_json(all_results, paths["results"] / "distillation_evaluation.json")
    print(f"[06_eval_distillation] wrote {paths['results'] / 'distillation_evaluation.json'}")


if __name__ == "__main__":
    main()
