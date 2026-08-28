#!/usr/bin/env python
"""Step 28 (Sections 32-33): final scientific interpretation. Answers each
of the 11 required questions strictly from files already written by
scripts 06/12/13/14/17, classifies SUPPORTED / PARTIALLY_SUPPORTED /
NOT_SUPPORTED, and writes results/final_scientific_interpretation.md."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts.report import (Conclusion, build_final_report, classify_compression_retention,
                         classify_distillation_improvement, classify_morans_correlation,
                         classify_multiscale_vs_local, classify_real_vs_shuffled,
                         classify_representation_fidelity)
from sts.utils import load_config, resolve_run_paths, save_json


def load_json_if_exists(path: Path):
    return json.load(open(path)) if path.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    distill_eval = load_json_if_exists(paths["results"] / "distillation_evaluation.json") or {}
    primary = load_json_if_exists(paths["results"] / "primary_hypothesis_test.json")
    secondary = load_json_if_exists(paths["results"] / "secondary_hypothesis_tests.json") or {}
    efficiency = load_json_if_exists(paths["results"] / "model_efficiency.json") or {}
    final_table = load_json_if_exists(paths["results"] / "final_results_table.csv")

    conclusions = []

    s0 = distill_eval.get(f"S0_seed{cfg['seeds'][0]}", {}).get("overall", {}).get("cosine_similarity", {}).get("mean")
    s1 = distill_eval.get(f"S1_seed{cfg['seeds'][0]}", {}).get("overall", {}).get("cosine_similarity", {}).get("mean")
    if s0 is not None and s1 is not None:
        conclusions.append(classify_distillation_improvement(s0, s1, s1 - s0))
    else:
        conclusions.append(Conclusion("Did GenBio distillation improve the 3.7M CNN over ImageNet init?",
                                       "distillation_evaluation.json missing S0/S1 results -- rerun scripts 05-06",
                                       "NOT_SUPPORTED", {}))

    if s1 is not None:
        conclusions.append(classify_representation_fidelity(s1))
    else:
        conclusions.append(Conclusion("How closely did the student reproduce GenBio representations?",
                                       "not yet measured", "NOT_SUPPORTED", {}))

    if "compression" in efficiency:
        conclusions.append(Conclusion("What was the exact compute difference?",
                                       f"{efficiency['compression']['compression_ratio_x']:.1f}x parameter compression "
                                       f"({efficiency['teacher']['total_parameters']:,} -> "
                                       f"{efficiency['student']['total_parameters']:,})",
                                       "SUPPORTED", efficiency["compression"]))
    else:
        conclusions.append(Conclusion("What was the exact compute difference?", "efficiency accounting not yet run",
                                       "NOT_SUPPORTED", {}))

    if primary is not None:
        from sts.report import classify_real_vs_shuffled as _c
        conclusions.append(_c(primary))
    else:
        conclusions.append(Conclusion("Did the real graph outperform the degree-preserving shuffled graph?",
                                       "primary_hypothesis_test.json missing -- rerun script 12", "NOT_SUPPORTED", {}))

    if "multiscale_vs_local" in secondary:
        conclusions.append(classify_multiscale_vs_local(secondary["multiscale_vs_local"]))
    else:
        conclusions.append(Conclusion("Did multiscale context outperform purely local context?",
                                       "secondary_hypothesis_tests.json missing -- rerun script 12", "NOT_SUPPORTED", {}))

    report_md = build_final_report(conclusions)
    out_path = paths["results"] / "final_scientific_interpretation.md"
    out_path.write_text(report_md)
    save_json([{"question": c.question, "verdict": c.verdict, "answer": c.answer} for c in conclusions],
              paths["results"] / "final_scientific_interpretation.json")
    print(report_md)
    print(f"[18_final_interpretation] wrote {out_path}")


if __name__ == "__main__":
    main()
