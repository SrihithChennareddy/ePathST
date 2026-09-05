#!/bin/bash
#SBATCH --job-name=morans_i_diagnostic
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell:1
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_morans_i_diagnostic_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_morans_i_diagnostic_%j.err

# Trains only A2 (patch-only) and A8 (flagship) across every official fold,
# then correlates each gene's (A8 - A2) R gain against its ground-truth
# Moran's I (real spatial autocorrelation) -- direct test of whether the
# graph correction exploits real spatial structure, replicating POC_5's
# Corr(delta, MoranI)=0.1148, p=0.630 null result on the current official
# HEST-1k protocol/data. See src/scripts/23_morans_i_diagnostic.py.

set -euo pipefail
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

python3 src/scripts/23_morans_i_diagnostic.py --config configs/base.yaml \
  --bench-root /lustre/nvwulf/scratch/schennareddy/HEST-1k/hest_bench \
  --cancers IDC COAD CCRCC --device cuda --seed 1 --feature-seed 0

echo "[run_morans_i_diagnostic] DONE"
