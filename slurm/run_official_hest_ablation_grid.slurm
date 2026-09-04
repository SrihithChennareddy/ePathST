#!/bin/bash
#SBATCH --job-name=official_hest_ablation_grid
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell:1
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_official_ablation_grid_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_official_ablation_grid_%j.err

# Full A0-A9 ablation grid (sts.ablations.ABLATION_GRID) under the OFFICIAL
# HEST-1k Benchmark protocol (official splits/genes/log1p-only targets),
# all 3 cancers (IDC/COAD/CCRCC), every official k-fold split (IDC 4,
# COAD 2, CCRCC 6 -- 12 fold x 9 ablations = 108 trained-model runs total).
# See scripts/22_official_hest_ablation_grid.py's module docstring for the
# exact protocol/adaptations. Smoke-tested on CPU (COAD fold 0, 2-epoch
# config) before this real run -- see docs/RUN_STATUS.md.

set -euo pipefail
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

python3 scripts/22_official_hest_ablation_grid.py --config configs/base.yaml \
  --bench-root /lustre/nvwulf/scratch/schennareddy/HEST-1k/hest_bench \
  --cancers IDC COAD CCRCC --device cuda --seed 1 --feature-seed 0

echo "[run_official_hest_ablation_grid] DONE"
