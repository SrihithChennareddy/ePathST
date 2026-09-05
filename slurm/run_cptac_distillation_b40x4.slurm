#!/bin/bash
#SBATCH --job-name=cptac_distill_b40
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=44G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell:1
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_b40_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_b40_%j.err

# Single-GPU variant of run_cptac_distillation.slurm, sized for the b40x4
# partition (RTX PRO 6000 Blackwell) instead of h200x4.
#
# WHY THIS EXISTS: as of 2026-08-28, every H200 GPU cluster-wide (both
# h200x4 and h200x8 partitions, which share the same 4+4 physical nodes) is
# fully allocated -- `scontrol show node` reports gres/gpu fully consumed
# on every h200x4-0[1-4] and h200x8-0[1-4] node -- and Slurm's own backfill
# scheduler predicted a ~24h wait for the h200x4 job (see job 83715,
# submitted first and left queued as a fallback/backup). b40x4-09 had 1
# idle RTX PRO 6000 Blackwell GPU (out of 4) and 30 idle CPUs (out of 62)
# at submission time, so this job targets that partition instead to start
# with ~zero queue wait. The RTX PRO 6000 Blackwell is the same GPU model
# this project's very first validation session already confirmed the full
# pipeline runs correctly on (97GB VRAM, plenty for these tiny models).
#
# Same resumable design as the H200 script (see its comments and
# src/scripts/01,02,04,05's docstrings) -- work_dir/persist_dir are shared, so
# whichever job (this one, or 83715 if/when it eventually starts) runs
# first does the real work and the other one's reruns just pick up
# whatever remains. Only ONE Step-1/2 run should be in flight at a time
# (their resumable caches are append-only, not concurrency-safe across
# simultaneous writers) -- 83715 was left PENDING specifically so it can't
# collide with this job's Steps 1-2; if it starts before this job reaches
# Step 5, cancel one of the two before Step 1/2 overlaps.
#
# Only 1 GPU here (vs. the H200 script's 4-way Step-4 shard), so Step 4
# runs unsharded (--num-shards 1) -- teacher embedding takes longer but
# still benefits fully from every already-cached embedding.
#
# --mem/--cpus-per-task must be sized to what's ACTUALLY free, checked via
# `scontrol show node <n> | grep -E "RealMemory|AllocMem"` (RealMemory -
# AllocMem = what Slurm's scheduler will actually let a new job claim --
# the node's live `FreeMem` OS metric is NOT what the scheduler uses and
# can look far more generous than what's really allocatable). Two
# lessons learned the hard way on this line:
#   1. job 83721 omitted --mem entirely, got Slurm's small default
#      allocation, and was OOM-killed ~1 minute into Step 1 (parallel
#      workers each holding a multi-hundred-MB DICOM WSI pyramid +
#      decompressed tile buffers spike well past a default few-GB limit).
#   2. job 83815 (resubmitted after the truncation-bug fix, see
#      docs/RUN_STATUS.md) requested --mem=220G (sized for a node that
#      had ~468GB truly free at 83721/83723's fix time) but got pinned via
#      --nodelist to b40x4-11, which only had ~47GB actually allocatable
#      at THAT submission time -- sat PENDING(Resources) indefinitely
#      despite showing a free GPU and "generous" live FreeMem, because
#      220G > 47GB by Slurm's own accounting. Reduced to 12 CPUs / 44GB
#      (fits comfortably under the tightest free-GPU node's ~47GB budget
#      observed across this cluster's b40x4 nodes) and dropped the
#      --nodelist pin so the scheduler can pick ANY node that currently
#      fits, rather than forcing one specific node that might not.
# Always re-check current headroom before resubmitting -- these numbers
# describe conditions at write time, not a permanent cluster state.

set -euo pipefail
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

CONFIG=configs/base.yaml
JOB_START=$SECONDS
WALL_CLOCK_SECONDS=$((8 * 3600))
RESERVE_SECONDS=$((20 * 60))

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Step 1: CPTAC WSI selection/QC (parallel, ${SLURM_CPUS_PER_TASK} workers) ==="
python3 src/scripts/01_select_cptac_wsi.py --config "$CONFIG" --workers "${SLURM_CPUS_PER_TASK}"

log "=== Step 2: CPTAC patch extraction (parallel, ${SLURM_CPUS_PER_TASK} workers) ==="
python3 src/scripts/02_extract_cptac_patches.py --config "$CONFIG" --workers "${SLURM_CPUS_PER_TASK}"

log "=== Step 3: CPTAC patient split (x3 seeds) ==="
python3 src/scripts/03_cptac_patient_split.py --config "$CONFIG"

log "=== Step 4: GenBio-PathFM teacher embedding (single GPU) ==="
python3 src/scripts/04_run_teacher_genbio.py --config "$CONFIG" --device cuda --shard-index 0 --num-shards 1

ELAPSED=$((SECONDS - JOB_START))
REMAINING=$((WALL_CLOCK_SECONDS - ELAPSED - RESERVE_SECONDS))
REMAINING_HOURS=$(python3 -c "print(max(0.05, $REMAINING / 3600))")
log "=== Step 5: distillation training, budget ${REMAINING_HOURS}h (S1 seed0 first) ==="
python3 src/scripts/05_train_student_distill.py --config "$CONFIG" --device cuda \
  --seeds 0 1 2 --baselines S1 S2 --time-budget-hours "$REMAINING_HOURS"

log "=== Step 6: distillation evaluation (whatever seeds/baselines finished) ==="
python3 src/scripts/06_eval_distillation.py --config "$CONFIG" --device cuda || \
  log "Step 6 failed or nothing to evaluate yet -- non-fatal, rerun after more training"

log "=== Step 9: lock student, strip teacher-projection head, save to persist_dir ==="
python3 src/scripts/07_lock_student.py --config "$CONFIG" --device cuda --mode A

log "=== snapshotting results to experimental_runs/ for later comparison ==="
python3 src/scripts/snapshot_experiment.py --config "$CONFIG" --job-id "${SLURM_JOB_ID}" --partition b40x4 \
  --gpu "1x RTX PRO 6000 Blackwell" --label cptac_distill_b40x4 --stage "end of 8h job"

log "=== DONE (total elapsed: $((SECONDS - JOB_START))s) ==="
