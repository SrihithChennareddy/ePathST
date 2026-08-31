#!/bin/bash
#SBATCH --job-name=official_hest_benchmark
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell:1
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_official_bench_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_official_bench_%j.err

# Official HEST-Benchmark reproduction (final-experiment diagnostic, 2026-08-30):
# GenBio-PathFM + distilled ConvNeXt V2-Atto on the official IDC/COAD/CCRCC
# hest-bench splits/genes/PCA256/Ridge protocol. Answers a DIFFERENT question
# from the custom 3-cancer patient-held-out experiment (scripts/10) -- see
# scripts/19_official_hest_benchmark.py's module docstring.

set -euo pipefail
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

python3 scripts/19_official_hest_benchmark.py --config configs/base.yaml \
  --bench-root /lustre/nvwulf/scratch/schennareddy/HEST-1k/hest_bench \
  --cancers IDC COAD CCRCC --device cuda --seed 1 --feature-seed 0

echo "[run_official_hest_benchmark] DONE"
