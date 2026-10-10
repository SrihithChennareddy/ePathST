#!/bin/bash
#SBATCH --job-name=tcga_worker
#SBATCH --account=pn_alyu012826n
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/tcga_worker_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/tcga_worker_%j.err

# One self-scheduling worker (1 GPU) of the TCGA hex-grid run; see src/scripts/24_tcga_validation.py worker.
# Submit with a partition and matching time limit (or use slurm/launch_tcga_workers.sh):
#   sbatch --partition=h200x4 --time=08:00:00 slurm/run_tcga_worker.slurm
#
# Keep-alive: as soon as this job STARTS it queues its successor with --dependency=afterany:<this job>, so the next session
# starts when this one ends however it ends (finished, crash, OOM, node failure, hard timeout). The successor is only queued
# while unfinished slides remain. Guards: CHAIN caps the chain length; a successor refuses to run if its predecessor failed
# within 10 minutes of starting (crash-loop breaker). Nothing here depends on any login session staying open.
set -uo pipefail
export HF_HOME=/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/hf_cache/huggingface
export OMP_NUM_THREADS=8
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

if [ -n "${PREV_JOB:-}" ]; then
  read -r PSTATE PELAPSED < <(sacct -j "$PREV_JOB" -n -X -P -o State,ElapsedRaw 2>/dev/null | head -1 | tr '|' ' ')
  case "${PSTATE:-}" in
    FAILED*|OUT_OF_ME*|NODE_FAIL*|BOOT_FAIL*)
      if [ "${PELAPSED:-99999}" -lt 600 ]; then
        echo "[run_tcga_worker] predecessor $PREV_JOB ${PSTATE} after ${PELAPSED}s: crash-loop breaker, stopping this chain"; exit 1
      fi;;
  esac
fi

REM=$(python3 src/scripts/24_tcga_validation.py worker --check-queue "$@" 2>/dev/null | tail -1)
echo "[run_tcga_worker] job $SLURM_JOB_ID partition $SLURM_JOB_PARTITION chain ${CHAIN:-0}: ${REM:-?} slides unfinished"
if [ "${REM:-0}" -gt 0 ] && [ "${CHAIN:-0}" -lt 200 ]; then
  TL=$(squeue -h -j "$SLURM_JOB_ID" -o %l)
  NEW=$(sbatch --parsable --partition="$SLURM_JOB_PARTITION" --time="$TL" --dependency=afterany:"$SLURM_JOB_ID" --begin=now+120 \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-6}" --mem="${SLURM_MEM_PER_NODE:-32768}M" \
        --export=ALL,CHAIN=$(( ${CHAIN:-0} + 1 )),PREV_JOB="$SLURM_JOB_ID" slurm/run_tcga_worker.slurm "$@")
  if [ -z "$NEW" ]; then echo "[run_tcga_worker] SUCCESSOR SUBMIT FAILED"; else echo "[run_tcga_worker] successor queued: job $NEW"; fi
fi

NT=${SLURM_CPUS_PER_TASK:-6}
python3 src/scripts/24_tcga_validation.py worker --config configs/base.yaml --device cuda --io-threads $(( NT > 2 ? NT - 1 : 1 )) "$@"
rc=$?
echo "[run_tcga_worker] worker exit code $rc"
# rc 0 = queue empty (or --limit reached), 10 = time limit near with slides left (successor already queued)
[ "$rc" -eq 10 ] && exit 0
exit $rc
