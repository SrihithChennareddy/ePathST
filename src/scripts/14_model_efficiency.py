#!/usr/bin/env python
"""Model efficiency report: parameter counts, compression ratio, FLOPs/MACs,
stored embedding size. Teacher and student costs are reported separately --
teacher inference is a one-time distillation cost, not a per-run cost."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathology_st.efficiency import (compression_ratio, count_model_params, measure_flops_macs,
                             measure_inference_throughput, stored_embedding_size_bytes)
from pathology_st.student_model import ConvNeXtV2AttoStudent, load_locked_encoder
from pathology_st.teacher_genbio import GenBioPathFMTeacher
from pathology_st.utils import RunEnvironment, load_config, resolve_run_paths, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    report = {}
    with RunEnvironment("14_model_efficiency", paths["logs"] / "14_model_efficiency_env.json"):
        student = load_locked_encoder(
            torch.load(paths["student_checkpoints"] / f"locked_encoder_seed{args.seed}.pt",
                       map_location=args.device)["encoder_state_dict"], args.device)
        student_params = count_model_params(student)
        report["student"] = {
            **student_params,
            "flops": measure_flops_macs(student, (3, 224, 224), args.device),
            "stored_embedding_bytes_per_patch": stored_embedding_size_bytes(1, cfg["student"]["backbone_dim"]),
        }

        try:
            teacher = GenBioPathFMTeacher(cfg, device=args.device)
            teacher_params = {"total_parameters": teacher.n_parameters, "trainable_parameters": 0}
        except Exception as e:
            print(f"[14_model_efficiency] could not load GenBio teacher for a live param count ({e}); "
                  f"reporting teacher params as unmeasured rather than guessing.")
            teacher_params = {"total_parameters": None, "trainable_parameters": 0, "note": "not measured this run"}

        report["teacher"] = teacher_params
        if teacher_params["total_parameters"]:
            report["compression"] = compression_ratio(teacher_params, student_params)
            report["compression"]["note"] = "teacher-label generation cost (Step 4) is a ONE-TIME cost, reported separately from repeated student inference cost below"

    save_json(report, paths["results"] / "model_efficiency.json")
    print(f"[14_model_efficiency] student: {student_params}")
    if report["teacher"]["total_parameters"]:
        print(f"[14_model_efficiency] compression ratio: {report['compression']['compression_ratio_x']:.1f}x")
    print(f"[14_model_efficiency] wrote {paths['results'] / 'model_efficiency.json'}")


if __name__ == "__main__":
    main()
