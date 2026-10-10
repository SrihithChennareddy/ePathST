#!/bin/bash
#SBATCH --job-name=leiden_official_hvg50
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell:1
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/leiden_official_hvg50_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/leiden_official_hvg50_%j.err

# Leiden on the official HEST-1k 50-gene panels over every official test sample, for
# ePathST-E, ePathST and H-Optimus-0 (patch head). See 35_leiden_official_hvg50.py.
# Extra args pass through, e.g.: sbatch ... --cancers COAD --max-folds-per-cancer 1 --out-name leiden_smoke

set -euo pipefail
export HF_HOME=/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/hf_cache/huggingface
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

python3 src/scripts/25_leiden_official_panel.py official --config configs/base.yaml \
  --bench-root /lustre/nvwulf/scratch/schennareddy/HEST-1k/hest_bench --device cuda "$@"

echo "[run_leiden_official_hvg50] DONE"
