#!/bin/bash
#SBATCH --job-name=cptac_distill_full
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell:1
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_full_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_full_%j.err

# Long-horizon companion to run_cptac_distillation_b40x4.slurm (job 83723,
# left RUNNING -- this job must NEVER touch that one). Goal: reach the
# spec's upper convergence target -- Section 9 says "Train approximately
# 30-100 epochs depending on convergence"; configs/base.yaml's
# distillation.{max_epochs,min_epochs,early_stop_patience} were raised to
# 100/50/15 (from 60/30/10) specifically for this run, with a full 24h
# budget to actually reach that range rather than being cut off early.
#
# WHY b40x4-01 / b40x4-long: h200x8-02 had 2/8 free H200s at first look,
# but h200x8-long's backfill scheduler predicted a ~24h wait for it
# (reason: "reserved for jobs in higher priority partitions") -- -long
# partitions apparently sit behind their short-partition siblings for the
# SAME physical GPUs. b40x4-01 had 1/4 free RTX PRO 6000 Blackwell and
# ~25 free CPUs / ~135GB free RAM; b40x4-long carries the same GRES as
# b40x4 but with a 2-day time limit, so --time=24:00:00 fits.
# --cpus-per-task=20/--mem=100G are sized to that node's actual free
# headroom (NOT the H200 node's -- an earlier submission requesting
# 32 CPUs/400G against this smaller node stayed PENDING("Resources")
# indefinitely since neither fit).
#
# CRITICAL SAFETY REQUIREMENT -- do not remove the wait loop below: job
# 83723 (still running on b40x4-09) is, at submission time, mid-Step-4,
# writing directly to the CANONICAL teacher-embeddings cache
# (persist_dir/teacher_embeddings/cptac_teacher_embeddings.h5) because it
# was launched with --num-shards 1 (unsharded -> writes straight to the
# canonical file, not a temp shard file). h5py/HDF5 does not support
# concurrent multi-process writers to the same file. If THIS job's own
# Step 4 call started while 83723 is still writing, the cache file could
# be corrupted. So this job waits until 83723's log shows it has moved
# past Step 4 (i.e. "=== Step 5" appears, meaning Step 4 finished and
# closed the file) before touching Steps 1-4 at all. Steps 1-3 are safe to
# rerun concurrently with 83723's Step 4/5 (they don't touch the same
# files 83723 is using at that point), and doing so here recovers the
# ~42 WSIs job 83723 lost to the (now-fixed) WsiDicomOutOfBoundsError bug.

set -euo pipefail
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

CONFIG=configs/base.yaml
OTHER_JOB_LOG=logs/slurm_b40_83723.out
OTHER_JOB_ID=83723
JOB_START=$SECONDS
WALL_CLOCK_SECONDS=$((24 * 3600))
RESERVE_SECONDS=$((30 * 60))
MAX_WAIT_SECONDS=$((4 * 3600))  # give up waiting after 4h and proceed anyway (logged loudly)

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== waiting for job ${OTHER_JOB_ID} to clear Step 4 (avoid concurrent teacher-embedding cache writes) ==="
WAITED=0
while true; do
  if grep -q "=== Step 5" "$OTHER_JOB_LOG" 2>/dev/null; then
    log "job ${OTHER_JOB_ID} reached Step 5 -- safe to proceed"
    break
  fi
  if ! squeue -j "${OTHER_JOB_ID}" -h >/dev/null 2>&1 || [ -z "$(squeue -j ${OTHER_JOB_ID} -h 2>/dev/null)" ]; then
    log "job ${OTHER_JOB_ID} is no longer in the queue (finished/failed) -- safe to proceed"
    break
  fi
  if [ "$WAITED" -ge "$MAX_WAIT_SECONDS" ]; then
    log "WARNING: waited ${MAX_WAIT_SECONDS}s for job ${OTHER_JOB_ID} without seeing Step 5 -- "
    log "proceeding anyway. If job ${OTHER_JOB_ID} is still mid-Step-4, this risks a concurrent-write"
    log "conflict on the teacher-embeddings cache. Check ${OTHER_JOB_LOG} manually if this looks wrong."
    break
  fi
  sleep 60
  WAITED=$((WAITED + 60))
done

log "=== Step 1: CPTAC WSI selection/QC (recovers any WSIs job ${OTHER_JOB_ID} lost to the fixed bug) ==="
python3 src/scripts/01_select_cptac_wsi.py --config "$CONFIG" --workers "${SLURM_CPUS_PER_TASK}"

log "=== Step 2: CPTAC patch extraction ==="
python3 src/scripts/02_extract_cptac_patches.py --config "$CONFIG" --workers "${SLURM_CPUS_PER_TASK}"

log "=== Step 3: CPTAC patient split (x3 seeds) ==="
python3 src/scripts/03_cptac_patient_split.py --config "$CONFIG"

log "=== Step 4: H-Optimus-0 teacher embedding (only NEW patches not already cached) ==="
python3 src/scripts/04_run_teacher_hoptimus.py --config "$CONFIG" --device cuda --shard-index 0 --num-shards 1

ELAPSED=$((SECONDS - JOB_START))
REMAINING=$((WALL_CLOCK_SECONDS - ELAPSED - RESERVE_SECONDS))
REMAINING_HOURS=$(python3 -c "print(max(0.05, $REMAINING / 3600))")
log "=== Step 5: distillation training, budget ${REMAINING_HOURS}h, targeting 50-100 epochs (S1 seed0 first) ==="
python3 src/scripts/05_train_student_distill.py --config "$CONFIG" --device cuda \
  --seeds 0 1 2 --baselines S1 S2 --time-budget-hours "$REMAINING_HOURS"

log "=== Step 6: distillation evaluation (whatever seeds/baselines finished) ==="
python3 src/scripts/06_eval_distillation.py --config "$CONFIG" --device cuda || \
  log "Step 6 failed or nothing to evaluate yet -- non-fatal, rerun after more training"

log "=== Step 9: lock student, strip teacher-projection head, save to persist_dir ==="
python3 src/scripts/07_lock_student.py --config "$CONFIG" --device cuda --mode A

log "=== snapshotting results to experimental_runs/ for later comparison ==="
python3 src/scripts/tools/snapshot_experiment.py --config "$CONFIG" --job-id "${SLURM_JOB_ID}" --partition b40x4-long \
  --gpu "1x RTX PRO 6000 Blackwell" --label cptac_distill_full_convergence --stage "end of 24h job (target 50-100 epochs)"

log "=== DONE (total elapsed: $((SECONDS - JOB_START))s) ==="
