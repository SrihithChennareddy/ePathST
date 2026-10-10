#!/bin/bash
#SBATCH --job-name=tcga_download_slides
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/nvwulf/scratch/schennareddy/TCGA/logs/download_slides_%j.out
#SBATCH --error=/lustre/nvwulf/scratch/schennareddy/TCGA/logs/download_slides_%j.err

# TCGA-BRCA/KIRC/COAD FFPE diagnostic slide download (~1.8TB, no GPU needed --
# network-bound, not compute-bound). Runs fully independent of any interactive
# session: persists across SSH/laptop disconnects, resumable on rerun or
# requeue (src/scripts/24_tcga_validation.py download skips already-downloaded
# files by size check). ETA was ~4h07m at probe time; --time=08:00:00 leaves
# real buffer.

set -euo pipefail
mkdir -p /lustre/nvwulf/scratch/schennareddy/TCGA/logs
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src

python3 src/scripts/24_tcga_validation.py download --config configs/base.yaml --workers 6

echo "[run_tcga_download_slides] DONE"
