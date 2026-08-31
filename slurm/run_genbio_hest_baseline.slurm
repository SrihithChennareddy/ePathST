#!/bin/bash
#SBATCH --job-name=genbio_hest_baseline
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gres=gpu:nvidia_rtx_pro_6000_blackwell:1
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_genbio_baseline_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_genbio_baseline_%j.err

# Standalone frozen-GenBio-PathFM patch-only baseline for the HEST
# IDC/COAD/CCRCC cohorts (scripts/run_genbio_hest_baseline.py). Runs fully
# independently of, and in parallel with, the CNN/GNN experiment job
# (run_hest_downstream_full.slurm, job 84327/84331 as of submission) --
# its own separate GPU allocation, no job dependency, and it only READS
# that job's manifests/hest_samples.csv, manifests/hest_spot_manifest.csv,
# and hest_patches/*.h5 (never writes to them). Its own outputs live under
# a new persist_dir subdir (genbio_baseline_embeddings/) and a new results
# subdir (results/genbio_baseline/) that the CNN/GNN job never touches.
#
# --mem=160G (bumped from 44G, 2026-08-30, after a real OOM kill on
# COAD): this script's per-cancer downstream-head loop reuses scripts/
# 10_run_patient_cv.py's load_fold_expression, which pools every
# training sample's full dense expression matrix -- same memory
# footprint that forced the CNN/GNN job's --mem up to 160G, since it's
# literally the same shared function.
#
# 24h wall-clock (vs. the CNN/GNN job's 8h) because a single run here does
# the full 3-cancer GenBio feature-extraction pass PLUS downstream-head
# training PLUS evaluation in one shot, with no multi-job resubmit chain --
# feature extraction alone (frozen GenBio-PathFM, ~1.1B params, over all
# ~200 samples' ST spots) is the dominant cost and is checkpointed/
# resumable per-sample if a rerun is ever needed.
#
# Runs its built-in 1-sample-per-cancer smoke test first (WSI/ST alignment,
# 4608-D embedding shape, cache reload, HVG/PROGENy spot-ID alignment, one
# downstream-head forward/backward pass) before the full 3-cancer run --
# see scripts/run_genbio_hest_baseline.py's run_smoke_test(). A smoke-test
# assertion failure aborts the job immediately (set -e) rather than wasting
# the 24h budget on a broken run.

set -euo pipefail
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== GenBio-PathFM HEST baseline (IDC/COAD/CCRCC), seed=0 fold=0 ==="
python3 scripts/run_genbio_hest_baseline.py --config configs/base.yaml --seed 0 --fold 0

log "=== DONE ==="
