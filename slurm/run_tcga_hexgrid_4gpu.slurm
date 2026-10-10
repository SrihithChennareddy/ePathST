#!/bin/bash
#SBATCH --job-name=tcga_hex4gpu
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --gres=gpu:h200:4
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/tcga_hex4gpu_%A_%a.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/tcga_hex4gpu_%A_%a.err

# One job = one 4-GPU node = 4 shards (one per GPU), so the per-user job limit is not the bottleneck.
# Submit with --array=0-2 for 12 shards total: array task T runs shards 4T..4T+3 of --n-shards 12.
# Extra args pass through to 40_tcga_predict_hexgrid.py (e.g. --models-dir, --tag, --projects).
# Resumable: finished slides are skipped, so a timed-out job can simply be resubmitted.
set -uo pipefail
export HF_HOME=/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/hf_cache/huggingface
export OMP_NUM_THREADS=8
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

N_SHARDS=12
TASK=${SLURM_ARRAY_TASK_ID:-0}
pids=()
for g in 0 1 2 3; do
  shard=$((TASK * 4 + g))
  CUDA_VISIBLE_DEVICES=$g python3 src/scripts/24_tcga_validation.py predict --config configs/base.yaml --device cuda \
    --shard "$shard" --n-shards $N_SHARDS --io-threads 10 "$@" \
    > "logs/tcga_hex4gpu_${SLURM_ARRAY_JOB_ID:-0}_shard${shard}.out" 2> "logs/tcga_hex4gpu_${SLURM_ARRAY_JOB_ID:-0}_shard${shard}.err" &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[run_tcga_hexgrid_4gpu] array task $TASK finished, rc=$rc"
exit $rc
