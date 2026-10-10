#!/bin/bash
#SBATCH --job-name=tcga_predict_hvg50
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4,h200x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/tcga_predict_hvg50_%A_%a.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/tcga_predict_hvg50_%A_%a.err

# Frozen ePathST-E / ePathST / H-Optimus-0 predictions of each cancer's official 50-gene panel on the extracted
# TCGA patches. Extra args pass through (e.g. --max-slides 5 --tag smoke --projects TCGA-COAD).
# Use --array=0-N%K with --n-shards N+1 for a full run; shard index = SLURM_ARRAY_TASK_ID.
set -euo pipefail
export HF_HOME=/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/hf_cache/huggingface
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1
python3 src/scripts/24_tcga_validation.py predict-sparse --config configs/base.yaml --device cuda \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" "$@"
