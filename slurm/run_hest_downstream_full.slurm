#!/bin/bash
#SBATCH --job-name=hest_downstream_full
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell:1
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_hest_full_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_hest_full_%j.err

# Full-scale HEST Phase-2 downstream run (Final_Project_V2.txt), all 3 real
# cancers (breast/IDC, colon/COAD, kidney/CCRCC -- 202 real samples total,
# configs/base.yaml's paths.hest_by_cancer) end to end: metadata audit,
# alignment QC, patch extraction, locked-encoder + ImageNet feature
# extraction, the full A0-A9 ablation grid (all 3 GNN backends), and the
# primary hypothesis test gates.
#
# --mem=160G (confirmed 2026-08-30 via a real OOM kill at the old 44G):
# Step 10's load_fold_expression pools every train-fold sample's FULL
# (non-HVG) dense expression matrix via pd.concat -- IDC fold0 alone is
# ~92k spots x ~18k genes (~13GB for one copy), and pd.concat plus the
# per-sample DataFrame list held simultaneously roughly doubles that
# peak, well past the old 44G limit sized for the (much lighter) CPTAC
# distillation job. b40x4 nodes have 500G+ RAM with plenty of headroom
# for this.
#
# CV DESIGN NOTE: per explicit user instruction this run uses a SINGLE real
# 80/20 patient-level split per cancer (--max-folds-per-cancer 1 --seeds 0),
# NOT Final_Project_V2.txt Phase 3's full 5-fold x 3-seed design -- an
# 80/20 GroupKFold split is exactly what n_outer_folds=5's fold 0 already
# is (configs/base.yaml's cross_validation.n_outer_folds=5 is unchanged;
# only how many of those 5 folds this particular run executes is trimmed).
# This is a deliberate time-budget tradeoff for one 24h job, not the
# project's primary/final design -- rerun with --max-folds-per-cancer 5
# --seeds 0 1 2 for the real V2 Phase 18 full patient-level experiment once
# there is a multi-job or longer-wall-clock budget for it.
#
# Steps 1-9 (CPTAC/GenBio distillation) are NOT touched or rerun here --
# this job only re-locks the encoder from whatever is currently the best
# CPTAC-distilled checkpoint (a fast repackaging step, not retraining) and
# then runs Phase-2 HEST steps 8-12.
#
# Resumable: src/scripts/08 and src/scripts/09 now skip already-processed samples
# (checkpointed every 5 samples / per-sample respectively) -- rerunning
# this script after a 24h cutoff picks up wherever it left off, same
# pattern as the CPTAC distillation jobs. src/scripts/10/12 are NOT
# checkpointed mid-run; if the job runs out of time inside Step 10, rerun
# from Step 10 (Steps 8/9's resumability means that part is cheap to skip).

set -euo pipefail
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

CONFIG=configs/base.yaml
JOB_START=$SECONDS
WALL_CLOCK_SECONDS=$((24 * 3600))
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
python3 src/scripts/snapshot_experiment.py --config "$CONFIG" --job-id "${SLURM_JOB_ID}" --partition b40x4-long \
  --gpu "1x RTX PRO 6000 Blackwell" --label hest_downstream_full_3cancer --stage "end of 24h job" || \
  log "snapshot failed -- non-fatal"

log "=== DONE (total elapsed: $((SECONDS - JOB_START))s) ==="
