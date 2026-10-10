#!/bin/bash
#SBATCH --job-name=leiden_hopt_debug
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=debug-b40x4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/cptac_hoptimus0_distill_hest_pna/leiden_clustering/logs/hoptimus_debug_%j.out
#SBATCH --error=/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/cptac_hoptimus0_distill_hest_pna/leiden_clustering/logs/hoptimus_debug_%j.err

set -euo pipefail
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1
python3 src/scripts/legacy/33_leiden_hoptimus_baseline.py --config configs/base.yaml --device cuda --max-samples 3
echo "[debug_leiden_hoptimus] DONE"
