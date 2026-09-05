#!/usr/bin/env python
"""Capture a point-in-time snapshot of the current run's results/ into
experimental_runs/<run_id>/ for later cross-run comparison (see
experimental_runs/README.md). Never fabricates numbers -- only copies
whatever the run has actually produced so far, and records which files
were present/absent so a partial snapshot (e.g. taken mid-training) is
never mistaken for a complete one.

Usage (typically called from a SLURM script right after a stage
completes, or by hand):
    python3 src/scripts/snapshot_experiment.py --config configs/base.yaml \\
        --job-id 83723 --partition b40x4 --gpu "1x RTX PRO 6000 Blackwell" \\
        --label cptac_distill_b40x4 --stage "step1-2 complete"
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathology_st.utils import load_config, resolve_run_paths


def _headline_metric(results_dir: Path) -> str:
    """Best-effort single number to show in INDEX.csv at a glance --
    prefers the primary hypothesis test, then the final results table's
    multiscale-PNA row, then distillation cosine similarity, in that
    priority order (most-downstream evidence first). Returns 'N/A' if none
    of these exist yet (e.g. a snapshot taken during CPTAC distillation,
    before any HEST results exist)."""
    primary = results_dir / "primary_hypothesis_test.json"
    if primary.exists():
        d = json.loads(primary.read_text())
        # script 12 nests results as {"pooled": {"<gate_name>": {"bootstrap": {...}}, ...},
        # "per_cancer": {...}} -- there is no top-level "bootstrap" key. Pick gate2 (spatial
        # context, the flagship comparison) as the headline, falling back to whichever gate exists.
        pooled = d.get("pooled", {})
        gate_name = next((g for g in ("gate2_spatial_context", "gate3_true_topology", "gate1_distillation")
                           if g in pooled), next(iter(pooled), None))
        b = pooled.get(gate_name, {}).get("bootstrap", {}) if gate_name else {}
        if b.get("mean_delta") is not None:
            return (f"{gate_name}_delta_R={b['mean_delta']:.4f} "
                    f"CI=[{b.get('ci_low', float('nan')):.4f},{b.get('ci_high', float('nan')):.4f}]")

    table = results_dir / "final_results_table.csv"
    if table.exists():
        import pandas as pd
        df = pd.read_csv(table)
        row = df[df["model"] == "A7_multiscale_pna"]
        if len(row):
            return f"A7_multiscale_pna R={row['hvg50_r_mean'].iloc[0]:.4f}"

    distill_eval = results_dir / "distillation_evaluation.json"
    if distill_eval.exists():
        d = json.loads(distill_eval.read_text())
        for key in d:
            if key.startswith("S1_"):
                cos = d[key].get("overall", {}).get("cosine_similarity", {}).get("mean")
                if cos is not None:
                    return f"S1 val cosine_sim={cos:.4f}"

    train_summary = results_dir / "distillation_training_summary.json"
    if train_summary.exists():
        d = json.loads(train_summary.read_text())
        for key in d:
            if key.startswith("S1_"):
                v = d[key].get("best_val_cosine_sim")
                if v is not None:
                    return f"S1 training best_val_cosine_sim={v:.4f} (epoch {d[key].get('best_epoch')})"

    return "N/A"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--job-id", default="")
    ap.add_argument("--partition", default="")
    ap.add_argument("--gpu", default="")
    ap.add_argument("--label", default="run", help="short slug for the run_id, e.g. cptac_distill_b40x4")
    ap.add_argument("--stage", default="", help="free-text: what stage/milestone this snapshot represents")
    ap.add_argument("--run-id", default=None, help="override the auto-generated run_id")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    today = date.today().isoformat()
    run_id = args.run_id or f"{today}_job{args.job_id or 'NA'}_{args.label}"

    out_dir = Path(__file__).resolve().parents[1] / "experimental_runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = {}
    for src_name, dst_name in [("results", "results"), ("figures", "figures")]:
        src = paths[src_name]
        if src.exists() and any(src.iterdir()):
            dst = out_dir / dst_name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            copied[dst_name] = sorted(p.name for p in dst.iterdir())
        else:
            copied[dst_name] = []

    config_snapshot_path = out_dir / "config_used.json"
    config_json = Path(args.config).with_suffix(".json")
    if config_json.exists():
        shutil.copy(config_json, config_snapshot_path)

    manifest = {
        "run_id": run_id,
        "captured_at": today,
        "job_id": args.job_id or None,
        "partition": args.partition or None,
        "gpu": args.gpu or None,
        "stage": args.stage or None,
        "source_work_dir": str(paths["work_dir"]),
        "source_persist_dir": str(paths["work_dir"].parent),  # not exact, informational only
        "files_captured": copied,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    notes_path = out_dir / "notes.md"
    if not notes_path.exists():
        notes_path.write_text(f"# {run_id}\n\nStage: {args.stage or '(unspecified)'}\n\n"
                               f"(add free-text observations about this snapshot here)\n")

    headline = _headline_metric(out_dir / "results") if (out_dir / "results").exists() else "N/A"

    index_path = Path(__file__).resolve().parents[1] / "experimental_runs" / "INDEX.csv"
    is_new = not index_path.exists()
    with open(index_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["run_id", "date", "job_id", "partition", "gpu", "stage_reached",
                              "headline_metric", "notes"])
        writer.writerow([run_id, today, args.job_id, args.partition, args.gpu, args.stage, headline, ""])

    print(f"[snapshot_experiment] captured {run_id} -> {out_dir}")
    print(f"[snapshot_experiment] results files: {copied['results']}")
    print(f"[snapshot_experiment] figures files: {copied['figures']}")
    print(f"[snapshot_experiment] headline metric: {headline}")
    print(f"[snapshot_experiment] appended row to {index_path}")


if __name__ == "__main__":
    main()
