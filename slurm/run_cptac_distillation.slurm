#!/bin/bash
#SBATCH --job-name=cptac_distill
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:h200:4
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_%j.err

# Runs Final_Project.txt Steps 1-9 (CPTAC WSI selection -> GenBio-PathFM
# teacher embedding -> ConvNeXt V2-Atto distillation -> lock) inside a
# single 8-hour H200x4 allocation. Every stage is resumable (see
# scripts/01, 02, 04, 05's docstrings) -- if this job runs out of wall
# clock, resubmit the SAME script and it continues from checkpoints/caches
# rather than restarting.
#
# Time budget within the 8h wall clock (dynamically computed below, not
# hard-coded): Steps 1-2 (CPU-bound WSI QC + patch extraction, parallel
# across --cpus-per-task) and Step 4 (GPU-bound teacher embedding, sharded
# 4-way across the node's 4 H200s) run first since they are prerequisites
# with a bounded cost; whatever wall clock remains (minus a 20-minute
# reserve for Step 9's lock/strip + buffer) goes to Step 5's distillation
# training, prioritizing S1 seed 0 (Section 10's main comparison target)
# -- see scripts/05_train_student_distill.py's --time-budget-hours.

set -euo pipefail
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

CONFIG=configs/base.yaml
JOB_START=$SECONDS
WALL_CLOCK_SECONDS=$((8 * 3600))
RESERVE_SECONDS=$((20 * 60))  # for Step 9 + final bookkeeping

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Step 1: CPTAC WSI selection/QC (parallel, ${SLURM_CPUS_PER_TASK} workers) ==="
python3 scripts/01_select_cptac_wsi.py --config "$CONFIG" --workers "${SLURM_CPUS_PER_TASK}"

log "=== Step 2: CPTAC patch extraction (parallel, ${SLURM_CPUS_PER_TASK} workers) ==="
python3 scripts/02_extract_cptac_patches.py --config "$CONFIG" --workers "${SLURM_CPUS_PER_TASK}"

log "=== Step 3: CPTAC patient split (x3 seeds) ==="
python3 scripts/03_cptac_patient_split.py --config "$CONFIG"

log "=== Step 4: GenBio-PathFM teacher embedding, sharded across 4 GPUs ==="
NUM_SHARDS=4
PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$i python3 scripts/04_run_teacher_genbio.py --config "$CONFIG" --device cuda \
    --shard-index "$i" --num-shards "$NUM_SHARDS" \
    > "logs/step4_shard${i}_${SLURM_JOB_ID}.out" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=1
done
if [ "$FAIL" -ne 0 ]; then
  log "one or more Step 4 shards failed -- see logs/step4_shard*_${SLURM_JOB_ID}.out"
  exit 1
fi
log "merging Step 4 shards into the canonical persistent cache"
python3 scripts/04_run_teacher_genbio.py --config "$CONFIG" --merge-shards --num-shards "$NUM_SHARDS"

ELAPSED=$((SECONDS - JOB_START))
REMAINING=$((WALL_CLOCK_SECONDS - ELAPSED - RESERVE_SECONDS))
REMAINING_HOURS=$(python3 -c "print(max(0.05, $REMAINING / 3600))")
log "=== Step 5: distillation training, budget ${REMAINING_HOURS}h (S1 seed0 first) ==="
python3 scripts/05_train_student_distill.py --config "$CONFIG" --device cuda \
  --seeds 0 1 2 --baselines S1 S2 --time-budget-hours "$REMAINING_HOURS"

log "=== Step 6: distillation evaluation (whatever seeds/baselines finished) ==="
python3 scripts/06_eval_distillation.py --config "$CONFIG" --device cuda || \
  log "Step 6 failed or nothing to evaluate yet -- non-fatal, rerun after more training"

log "=== Step 9: lock student, strip teacher-projection head, save to persist_dir ==="
# skips any seed whose S1 checkpoint doesn't exist yet (time-budget-limited
# Step 5 may not have reached every seed) rather than failing the whole job
python3 scripts/07_lock_student.py --config "$CONFIG" --device cuda --mode A

log "=== DONE (total elapsed: $((SECONDS - JOB_START))s) ==="
