#!/bin/bash
#SBATCH --job-name=debug_idc_pipeline
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell:1
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_debug_idc_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_debug_idc_%j.err

# IDC-only sanity rerun (final-experiment diagnostic, 2026-08-30) after fixing:
#   1. A1_imagenet_patch restored with its own real feature cache + patch_head
#      (src/sts/feature_cache.py metadata + src/sts/ablations.py + src/scripts/10).
#   2. Target/prediction misalignment bug for every graph ablation (A4-A9)
#      in src/scripts/10_run_patient_cv.py's run_ablation().
# Reuses ALL existing caches (patches, distilled+imagenet features, teacher
# embeddings) -- only re-runs Step 10 (full A0-A9 grid, single 80/20 IDC
# split, seed 0) + Step 12 (gates) against the fixed code.

set -euo pipefail
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

CONFIG=configs/debug_idc_only.json

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Step 10 (fixed): full A0-A9 grid, single 80/20 IDC split, seed 0 ==="
python3 src/scripts/10_run_patient_cv.py --config "$CONFIG" --device cuda --feature-seed 0 \
  --max-folds-per-cancer 1 --seeds 0

log "=== Step 12: primary hypothesis test (3 gates, IDC only) ==="
python3 src/scripts/12_primary_hypothesis_test.py --config "$CONFIG" --max-folds-per-cancer 1 --seeds 0

log "=== copying results to results/debug_idc_pipeline/ ==="
mkdir -p results/debug_idc_pipeline
python3 - <<'PYEOF'
import json, sys
sys.path.insert(0, ".")
from sts.utils import load_config, resolve_run_paths
cfg = load_config("configs/debug_idc_only.json")
paths = resolve_run_paths(cfg)
import shutil
shutil.copy(paths["results"] / "cv_ablation_metrics.csv", "results/debug_idc_pipeline/cv_ablation_metrics_postfix.csv")
gates_path = paths["results"] / "primary_hypothesis_test.json"
if gates_path.exists():
    shutil.copy(gates_path, "results/debug_idc_pipeline/primary_hypothesis_test_postfix.json")
print("copied results into results/debug_idc_pipeline/")
PYEOF

log "=== DONE ==="
