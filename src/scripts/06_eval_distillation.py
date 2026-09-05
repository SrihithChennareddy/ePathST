#!/usr/bin/env python
"""Step 8: evaluate S0/S1/(S2) on unseen CPTAC test patients -- cosine
similarity distribution, retrieval overlap@K, linear-probe sanity, and
UMAP visualization -- overall and per cancer type, with bootstrap CIs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathology_st.distill_eval import evaluate_distillation, linear_probe_sanity, umap_embedding
from pathology_st.distill_train import CPTACDistillDataset
from pathology_st.student_model import DistillationStudent
from pathology_st.utils import load_config, resolve_run_paths, save_json


@torch.no_grad()
def project_all(model, dataset, device, batch_size=256):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
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
        split_df = pd.read_csv(paths["manifests"] / f"cptac_split_seed{seed}.csv")
        merged = patch_manifest.merge(split_df[["patient_id", "split", "cancer_type"]], on="patient_id",
                                       suffixes=("", "_split"))
        test_pm = merged[merged["split"] == "test"]
        test_ds = CPTACDistillDataset(test_pm, teacher_h5, shard_dir, train=False, cfg=cfg)

        for baseline in ["S0", "S1", "S2"]:
            ckpt = paths["student_checkpoints"] / f"{baseline}_seed{seed}_best.pt"
            if baseline != "S0" and not ckpt.exists():
                continue
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
            print(f"[06_eval_distillation] {baseline} seed{seed}: overall mean cosine = "
                  f"{report['overall']['cosine_similarity']['mean']:.4f}")

            if seed == cfg["seeds"][0]:
                umap_xy = umap_embedding(proj, seed=seed)
                np.savez(paths["figures"] / f"umap_{baseline}_seed{seed}.npz", xy=umap_xy, cancer=cancer_types)

    save_json(all_results, paths["results"] / "distillation_evaluation.json")
    print(f"[06_eval_distillation] wrote {paths['results'] / 'distillation_evaluation.json'}")


if __name__ == "__main__":
    main()
