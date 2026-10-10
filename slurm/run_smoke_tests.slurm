#!/bin/bash
#SBATCH --job-name=epathst_smoke
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell:1
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/smoke_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/smoke_%j.err

# Unit tests, deterministic regression digests and the one-slide TCGA end-to-end check (one slide per cancer).
# Usage: sbatch slurm/run_smoke_tests.slurm [digest_out.json]
set -euo pipefail
export HF_HOME=/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/hf_cache/huggingface
cd "${REPO_DIR:-/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src}"
module load pytorch/2.7.1

python3 -m pytest -q tests/test_graph_and_stats.py tests/test_tcga_queue_claims.py
python3 tests/regression_baseline.py --out "${1:-logs/regression_digests_$SLURM_JOB_ID.json}"
python3 tests/e2e_one_slide.py TCGA-BRCA:TCGA-A8-A06O TCGA-COAD:TCGA-AA-3846 TCGA-KIRC:TCGA-A3-3385
echo "[run_smoke_tests] DONE"
