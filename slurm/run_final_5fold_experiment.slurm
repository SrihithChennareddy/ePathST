#!/bin/bash
#SBATCH --job-name=final_5fold_experiment
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell:1
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_final_5fold_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_final_5fold_%j.err

# THE LOCKED FINAL SCIENTIFIC EXPERIMENT (2026-08-30) -- full V2 Phase 3
# design: 5-fold patient-level CV x 3 cancers (IDC/COAD/CCRCC) x seed 0,
# full A0-A9 ablation grid (A1 restored, target-misalignment bug fixed,
# pathway_loss_weight=0.2), using the validated implementation exactly as
# locked after the diagnostic pass (docs/RUN_STATUS.md + this session's
# commit "Fix A1 feature-mixup, graph-ablation target misalignment,
# pathway_loss_weight=0; add official HEST-Benchmark reproduction").
#
# Per the final-experiment spec: seed 0 is the PRIMARY run (this job).
# Seeds 1/2 for the major models only are a SEPARATE follow-up, run only
# if compute allows after this primary result is in -- not required for
# every ablation.
#
# Reuses all existing caches (patches, distilled+imagenet+genbio features,
# locked encoder checkpoint) -- only Step 10 (full 5-fold grid) + Step 12
# (gates) need to run.

set -euo pipefail
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

CONFIG=configs/base.yaml

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Step 10: full A0-A9 grid, full 5-fold CV, 3 cancers, seed 0 (LOCKED FINAL RUN) ==="
python3 scripts/10_run_patient_cv.py --config "$CONFIG" --device cuda --feature-seed 0 --seeds 0

log "=== Step 12: primary hypothesis test (3 gates, per-cancer + pooled) ==="
python3 scripts/12_primary_hypothesis_test.py --config "$CONFIG" --seeds 0

log "=== snapshotting results to experimental_runs/ ==="
python3 scripts/snapshot_experiment.py --config "$CONFIG" --job-id "${SLURM_JOB_ID}" --partition b40x4-long \
  --gpu "1x RTX PRO 6000 Blackwell" --label final_5fold_3cancer_seed0 \
  --stage "LOCKED final 5-fold x 3-cancer x seed-0 primary experiment, post-diagnostic-fix" || \
  log "snapshot failed -- non-fatal, results/cv_ablation_metrics.csv is still the source of truth"

log "=== DONE ==="
