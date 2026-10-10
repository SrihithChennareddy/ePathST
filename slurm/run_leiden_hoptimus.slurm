#!/bin/bash
#SBATCH --job-name=leiden_hoptimus
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=h200x4-long
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/cptac_hoptimus0_distill_hest_pna/leiden_clustering/logs/hoptimus_%j.out
#SBATCH --error=/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/cptac_hoptimus0_distill_hest_pna/leiden_clustering/logs/hoptimus_%j.err

# Zero-shot H-Optimus-0 Leiden/ARI baseline, full HEST-1k cohort. No training
# (just frozen-encoder forward passes over already-cached patches), so this
# should be much faster than the ePathST-E/ePathST GNN-training job.

set -euo pipefail
mkdir -p /lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/cptac_hoptimus0_distill_hest_pna/leiden_clustering/logs
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

python3 src/scripts/legacy/33_leiden_hoptimus_baseline.py --config configs/base.yaml --device cuda

echo "[run_leiden_hoptimus] DONE"
