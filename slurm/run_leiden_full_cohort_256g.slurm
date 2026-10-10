#!/bin/bash
#SBATCH --job-name=leiden_full_cohort_256g
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4-long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/cptac_hoptimus0_distill_hest_pna/leiden_clustering/logs/run_%j.out
#SBATCH --error=/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/cptac_hoptimus0_distill_hest_pna/leiden_clustering/logs/run_%j.err

# Leiden/ARI (ePathST-E vs ePathST) across the FULL HEST-1k cohort, not just
# TENX149. Retrains only the A8 GNN per fold (A2's patch head is reused from
# the already-saved checkpoint) using cached student-encoder embeddings --
# no WSI access anywhere. GPU needed for GNN training; Leiden clustering
# itself is CPU-only and cheap.

set -euo pipefail
mkdir -p /lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/cptac_hoptimus0_distill_hest_pna/leiden_clustering/logs
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

python3 src/scripts/legacy/32_leiden_full_cohort.py --config configs/base.yaml --device cuda

echo "[run_leiden_full_cohort] DONE"
