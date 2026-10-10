#!/bin/bash
#SBATCH --job-name=epathst_figs
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=debug-b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell:1
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/figures_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/figures_%j.err

# Regenerates the paper figures into a scratch folder and compares them with the reference images.
# Usage: sbatch slurm/run_figures_check.slurm [out_dir]
set -euo pipefail
cd "${REPO_DIR:-/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src}"
module load pytorch/2.7.1

python3 tests/e2e_figures.py "${1:-/lustre/nvwulf/scratch/schennareddy/refactor_baseline/figures_$SLURM_JOB_ID}"
echo "[run_figures_check] DONE"
