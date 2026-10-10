#!/usr/bin/env python
"""Step 9 of FINAL_GUIDE.md: every HEST number traced to a file, one metric definition, honest
sample sizes. No training; reading outputs only. Output in results/tcga_final/hest/."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SRIHITH = Path("/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith")
WORK = SRIHITH / "cptac_hoptimus0_distill_hest_pna_work"
HEST_RUNS = {0: WORK / "results" / "leiden_official_hvg50", 1: WORK / "results" / "leiden_official_hvg50_seed1"}
HEAD_OF = {"BRCA": "IDC", "COAD": "COAD", "KIRC": "CCRCC"}
SYSTEMS = {"E": "ePathST-E", "P": "ePathST", "T": "H-Optimus-0"}
COLLAPSED = {("IDC", 2, "E", 0), ("CCRCC", 4, "T", 1)}
OLD_PAPER_VALUES = {
    "ePathST": (0.5384, 0.3302, 0.2375), "patch-only (ePathST-E)": (0.4890, 0.3151, 0.2275),
    "local": (0.5161, 0.3238, 0.2316), "meso": (0.5237, 0.3261, 0.2328),
    "macro": (0.5194, 0.3249, 0.2341), "multiscale": (0.5272, 0.3256, 0.2337),
}
BENCH_SUMMARY = WORK / "results" / "official_hest_benchmark" / "official_hest_benchmark_summary.csv"
BENCH_MANIFEST = Path("experimental_runs/2026-08-31_job84871_final_5fold_3cancer_seed0/manifest.json")

log = logging.getLogger("57_hest_tables")


def provenance_search() -> dict:
    """9.1: search for the old paper numbers in any logged run; separately verify whether
    ePathST-E's own linear-probe number (0.3144) traces to a real job."""
    hits = {}
    for name, vals in OLD_PAPER_VALUES.items():
        found = []
        for v in vals:
            s = f"{v:.4f}"
            try:
                out = subprocess.run(
                    ["grep", "-rl", s, "logs", "results", "experimental_runs"],
                    cwd=Path.cwd(), capture_output=True, text=True, timeout=60,
                )
                files = [f for f in out.stdout.splitlines() if "docs/" not in f and "gene_level_outcomes" not in f
                         and "program_tests" not in f and "coefficients" not in f and "cohort" not in f
                         and "screen_full" not in f and "A5_scale" not in f and "A10_A14" not in f
                         and "gene_context" not in f]
                found += files
            except Exception as e:
                log.warning("grep failed for %s: %s", s, e)
        hits[name] = sorted(set(found))

    linear_probe_trace = {"traced": False}
    if BENCH_SUMMARY.exists():
        df = pd.read_csv(BENCH_SUMMARY)
        d = df[df.encoder == "distilled_convnextv2_atto"]
        if len(d) == 3:
            mean_pcc = d["pearson_mean"].mean()
            manifest = json.loads(BENCH_MANIFEST.read_text()) if BENCH_MANIFEST.exists() else {}
            linear_probe_trace = {
                "traced": True, "file": str(BENCH_SUMMARY), "per_cancer": d.set_index("cancer")["pearson_mean"].to_dict(),
                "mean": float(mean_pcc), "matches_repo_0.3144": abs(mean_pcc - 0.3144) < 0.001,
                "job_id": manifest.get("job_id"), "source_work_dir": manifest.get("source_work_dir"),
            }
    return {
        "old_ablation_values_found_in_logs": hits,
        "ePathST-E_linear_probe_0.3144": linear_probe_trace,
        "teacher_provenance_note": (
            "The job-84871 manifest's source_work_dir is "
            "'cptac_genbio_distill_hest_pna_work' (GenBio-PathFM), while the current TCGA "
            "prediction pipeline's work dir is 'cptac_hoptimus0_distill_hest_pna_work' "
            "(H-Optimus-0) -- additional evidence for the already-open teacher-identity "
            "question (Step 0/round-2-refactor), not resolved here."
        ),
    }


def hest_main_table(panels: dict) -> pd.DataFrame:
    """9.2: per cancer and 3-cancer mean, E/P/T for seed 0, seed 1 and the two-seed mean; a second
    version excludes the two Step 1 collapsed folds."""
    rows = []
    for cancer, genes in panels.items():
        head = HEAD_OF[cancer]
        for sys_code, sys_name in SYSTEMS.items():
            for seed, run in HEST_RUNS.items():
                files = sorted((run / "preds").glob(f"{head}_fold*_{sys_name}.npz"))
                fold_means_all, fold_means_excl = [], []
                for f in files:
                    fold = int(f.stem.split("_fold")[1].split("_")[0])
                    z = np.load(f)
                    y_true, y_pred = z["y_true"], z["y_pred"]
                    pcc = np.array(
                        [stats.pearsonr(y_true[:, gi], y_pred[:, gi])[0] for gi in range(len(genes))]
                    )
                    m = float(np.mean(pcc))
                    fold_means_all.append(m)
                    if (head, fold, sys_code, seed) not in COLLAPSED:
                        fold_means_excl.append(m)
                rows.append(
                    {"cancer": cancer, "system": sys_code, "seed": seed, "n_folds": len(fold_means_all),
                     "pcc_mean_all_folds": float(np.mean(fold_means_all)),
                     "n_folds_excl_collapsed": len(fold_means_excl),
                     "pcc_mean_excl_collapsed": float(np.mean(fold_means_excl)) if fold_means_excl else np.nan}
                )
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["cancer", "system"], columns="seed", values=["pcc_mean_all_folds", "pcc_mean_excl_collapsed"])
    wide.columns = [f"{a}_seed{b}" for a, b in wide.columns]
    wide["pcc_mean_all_folds_2seed_mean"] = wide[["pcc_mean_all_folds_seed0", "pcc_mean_all_folds_seed1"]].mean(axis=1)
    wide["pcc_mean_excl_collapsed_2seed_mean"] = wide[["pcc_mean_excl_collapsed_seed0", "pcc_mean_excl_collapsed_seed1"]].mean(axis=1)
    wide = wide.reset_index()
    three_cancer_mean = wide.groupby("system")[[c for c in wide.columns if "pcc_mean" in c]].mean()
    three_cancer_mean.insert(0, "cancer", "three_cancer_mean")
    three_cancer_mean = three_cancer_mean.reset_index()
    out = pd.concat([wide, three_cancer_mean], ignore_index=True)
    out["metric"] = (
        "per-gene Pearson correlation on each held-out fold, averaged over genes, then over "
        "folds; cancers weighted equally for the three-cancer mean"
    )
    return out


def sample_sizes() -> pd.DataFrame:
    """9.5, from A9's training-set counts (already on disk from Step 1)."""
    a9 = pd.read_csv("results/tcga_final/audit/A9_hest_training_sets.csv")
    rows = []
    for cancer, head in HEAD_OF.items():
        g = a9[a9.cancer == head]
        rows.append(
            {"cancer": cancer, "head": head, "n_test_folds": g["fold"].nunique(),
             "train_slides_per_fold_min": int(g["n_train"].min()), "train_slides_per_fold_max": int(g["n_train"].max()),
             "n_seeds": a9["seed"].nunique()}
        )
    return pd.DataFrame(rows)


def compute_table(out: Path) -> pd.DataFrame:
    """9.6: per-patch MACs (fvcore) for the student, the H-Optimus-0 teacher, and ResNet-50 (the
    old repo's own comparator, not the teacher) -- one patches-per-WSI assumption, one tool,
    rechecking the old 3.69x ResNet-50-vs-student ratio (4.1 GMACs there is ResNet-50's own
    figure, not H-Optimus-0's)."""
    rows = []
    try:
        import torch
        from fvcore.nn import FlopCountAnalysis

        from epathst.utils import load_config
        from epathst.models.student import load_locked_encoder

        cfg = load_config("configs/base.yaml")
        ckpt_path = None
        for cand in Path(".").glob("**/locked_encoder_seed0.pt"):
            ckpt_path = cand
            break
        if ckpt_path is None:
            raise FileNotFoundError("no locked_encoder_seed0.pt found to load the student for FLOP counting")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        student = load_locked_encoder(ckpt["encoder_state_dict"], "cpu").eval()
        x = torch.randn(1, 3, 224, 224)
        macs_student = FlopCountAnalysis(student, x).total()
        rows.append({"model": "student (distilled ConvNeXt V2-Atto)", "patch_px": 224, "GMACs_per_patch": macs_student / 1e9})
    except Exception as e:
        log.warning("student FLOP count failed: %s", e)

    try:
        import timm
        import torch
        from fvcore.nn import FlopCountAnalysis

        teacher = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False).eval()
        x = torch.randn(1, 3, 224, 224)
        macs_teacher = FlopCountAnalysis(teacher, x).total()
        rows.append({"model": "H-Optimus-0 teacher", "patch_px": 224, "GMACs_per_patch": macs_teacher / 1e9})
    except Exception as e:
        log.warning("teacher FLOP count failed (likely needs network access for the HF checkpoint): %s", e)

    try:
        import torch
        import torchvision.models as tvm
        from fvcore.nn import FlopCountAnalysis

        resnet50 = tvm.resnet50(weights=None).eval()
        x = torch.randn(1, 3, 224, 224)
        macs_resnet = FlopCountAnalysis(resnet50, x).total()
        rows.append({"model": "ResNet-50 (old repo's own comparator)", "patch_px": 224, "GMACs_per_patch": macs_resnet / 1e9})
    except Exception as e:
        log.warning("ResNet-50 FLOP count failed: %s", e)

    df = pd.DataFrame(rows)
    df["tool"] = "fvcore (counts MACs, not FLOPs)"
    by_model = df.set_index("model")["GMACs_per_patch"]
    ratios = {}
    if "H-Optimus-0 teacher" in by_model.index and "student (distilled ConvNeXt V2-Atto)" in by_model.index:
        ratios["teacher_over_student"] = float(by_model["H-Optimus-0 teacher"] / by_model["student (distilled ConvNeXt V2-Atto)"])
        log.info("teacher/student MACs ratio (recomputed): %.2fx", ratios["teacher_over_student"])
    if "ResNet-50 (old repo's own comparator)" in by_model.index and "student (distilled ConvNeXt V2-Atto)" in by_model.index:
        ratios["resnet50_over_student"] = float(by_model["ResNet-50 (old repo's own comparator)"] / by_model["student (distilled ConvNeXt V2-Atto)"])
        log.info(
            "ResNet-50/student MACs ratio (recomputed): %.2fx (old repo Table IV claim: 3.69x; "
            "guide's own 'about 7.5x expected from 4.1 vs 0.55 GMACs' reference)", ratios["resnet50_over_student"]
        )
    df.attrs["ratios"] = ratios
    return df, ratios


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="configs/analysis_plan.json")
    ap.add_argument("--out", default="results/tcga_final")
    args = ap.parse_args()
    out_root = Path(args.out)
    out = out_root / "hest"
    out.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(out_root / "logs" / "57_hest_tables.log"), logging.StreamHandler()],
    )
    plan = json.loads(Path(args.plan).read_text())
    panels = {c: plan["panels"][cfg["head"]] for c, cfg in plan["cohorts"].items()}

    prov = provenance_search()
    (out / "provenance_search.json").write_text(json.dumps(prov, indent=1, default=str))
    log.info("provenance: %s", json.dumps(prov, default=str)[:2000])

    main_table = hest_main_table(panels)
    main_table.to_csv(out / "hest_main.csv", index=False)
    log.info("hest_main.csv:\n%s", main_table.to_string(index=False))

    sizes = sample_sizes()
    sizes.to_csv(out / "sample_sizes.csv", index=False)

    comp, ratios = compute_table(out)
    comp.to_csv(out / "compute.csv", index=False)
    (out / "compute_ratios.json").write_text(json.dumps(ratios, indent=1))

    scale_note = (
        "9.3 scale ablation: the only on-disk local/meso/macro/union/scale-aware ablation grid "
        "(experimental_runs/*_job88050_official_hest_ablation_grid and the earlier full A0-A9 "
        "grid referenced in docs/reports/HEST_OFFICIAL_ABLATION_RESULTS.md) is the same "
        "untraceable-to-the-current-pipeline run as the old paper numbers (trained on the "
        "~200-slide cohort, not the official benchmark folds now on disk). Per the guide's "
        "fallback: dropped from the main text; the GNN is described as one system (P)."
    )
    secondary_note = (
        "9.4 secondary table: the on-disk official_hest_benchmark run (job 84871) only "
        "benchmarked 2 encoders under the HEST linear-probe protocol (PCA-256 + Ridge): "
        "distilled_convnextv2_atto (ePathST-E, mean PCC 0.3144 across IDC/COAD/CCRCC -- matches "
        "the repo's cited value) and genbio_pathfm. The full 9-foundation-model comparison "
        "table is not available from any on-disk artifact without either citing HEST's own "
        "published numbers externally or running a new benchmark, neither of which fits "
        "'no training, reading outputs only' -- not fabricated here."
    )
    (out / "NOTES.md").write_text(f"# Step 9 notes\n\n{scale_note}\n\n{secondary_note}\n")
    log.info(scale_note)
    log.info(secondary_note)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
