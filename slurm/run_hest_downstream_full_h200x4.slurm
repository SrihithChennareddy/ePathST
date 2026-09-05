#!/bin/bash
#SBATCH --job-name=hest_downstream_full
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=44G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_hest_full_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_hest_full_%j.err

# H200x4-partition variant of run_hest_downstream_full.slurm, identical
# script logic -- only --partition/--gres changed (RTX PRO 6000 Blackwell
# -> H200) for more compute on later chained resubmits. Submitted with
# --dependency=afterany:<prior job> so it only starts once the prior link
# in the chain (b40x4 or another h200x4 job) has stopped running --
# src/scripts/08-12 write to the same shared manifest/checkpoint/output paths,
# so two links must never run concurrently, only chained.
#
# Everything below is unchanged from run_hest_downstream_full.slurm --
# see that file for full step-by-step commentary.

set -euo pipefail
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

CONFIG=configs/base.yaml
JOB_START=$SECONDS
WALL_CLOCK_SECONDS=$((8 * 3600))
RESERVE_SECONDS=$((15 * 60))

log() { echo "[$(date +%H:%M:%S)] $*"; }
remaining() { echo $((WALL_CLOCK_SECONDS - (SECONDS - JOB_START) - RESERVE_SECONDS)); }

log "=== Step 9 (re-lock): package the LATEST CPTAC-distilled checkpoint as the frozen encoder ==="
log "    (packaging only -- does not retrain; CNN/distillation code and config untouched)"
python3 src/scripts/07_lock_student.py --config "$CONFIG" --device cuda --mode A

if [ "$(remaining)" -le 0 ]; then log "no time left after Step 9 -- exiting for resubmit"; exit 0; fi
log "=== Step 8: HEST metadata audit + alignment QC + patch extraction, ALL 3 cancers (resumable) ==="
python3 src/scripts/08_prepare_hest_metadata.py --config "$CONFIG"

if [ "$(remaining)" -le 0 ]; then log "no time left after Step 8 -- exiting for resubmit"; exit 0; fi
log "=== Step 9b: locked-distilled + ImageNet feature extraction over every HEST patch (resumable) ==="
python3 src/scripts/09_extract_student_features_hest.py --config "$CONFIG" --device cuda --seed 0

if [ "$(remaining)" -le 0 ]; then log "no time left after Step 9b -- exiting for resubmit"; exit 0; fi
log "=== Step 10: full A0-A9 ablation grid (all 3 GNN backends), single 80/20 split, all 3 cancers ==="
python3 src/scripts/10_run_patient_cv.py --config "$CONFIG" --device cuda --feature-seed 0 \
  --max-folds-per-cancer 1 --seeds 0

if [ "$(remaining)" -le 0 ]; then log "no time left after Step 10 -- exiting for resubmit"; exit 0; fi
log "=== Step 12: primary hypothesis test (3 gates, per-cancer + pooled) ==="
python3 src/scripts/12_primary_hypothesis_test.py --config "$CONFIG" --max-folds-per-cancer 1 --seeds 0 || \
  log "Step 12 failed or incomplete -- non-fatal, results/cv_ablation_metrics.csv is still valid, rerun once fixed"

log "=== snapshotting results to experimental_runs/ for later comparison ==="
python3 src/scripts/snapshot_experiment.py --config "$CONFIG" --job-id "${SLURM_JOB_ID}" --partition h200x4 \
  --gpu "1x H200" --label hest_downstream_full_3cancer --stage "end of 8h job" || \
  log "snapshot failed -- non-fatal"

log "=== DONE (total elapsed: $((SECONDS - JOB_START))s) ==="
