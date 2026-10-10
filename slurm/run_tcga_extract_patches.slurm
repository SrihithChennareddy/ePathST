#!/bin/bash
#SBATCH --job-name=tcga_extract_patches
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=56G
#SBATCH --time=2-00:00:00
#SBATCH --output=/lustre/nvwulf/scratch/schennareddy/TCGA/logs/extract_patches_%j.out
#SBATCH --error=/lustre/nvwulf/scratch/schennareddy/TCGA/logs/extract_patches_%j.err

# Runs patch extraction (CPU-only, 48 parallel workers) repeatedly in a loop
# while the slide download job (25_tcga_download_slides.py, job id passed as
# $1) is still active, so it eats through newly-landed slides as they arrive
# instead of waiting for the full ~1.8TB download to finish first. Stops once
# the download job is no longer queued/running AND one more extraction pass
# finds nothing new to do.

set -uo pipefail
mkdir -p /lustre/nvwulf/scratch/schennareddy/TCGA/logs
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

DOWNLOAD_JOB_ID="${1:-}"

download_still_running() {
    [ -z "$DOWNLOAD_JOB_ID" ] && return 1
    squeue -h -j "$DOWNLOAD_JOB_ID" 2>/dev/null | grep -q .
}

iteration=0
while true; do
    iteration=$((iteration + 1))
    echo "[run_tcga_extract_patches] iteration $iteration: running extraction pass over slides currently on disk..."
    python3 src/scripts/24_tcga_validation.py patches --config configs/base.yaml --workers 40 \
        > /lustre/nvwulf/scratch/schennareddy/TCGA/logs/extract_pass_${iteration}.log 2>&1
    tail -5 /lustre/nvwulf/scratch/schennareddy/TCGA/logs/extract_pass_${iteration}.log

    if download_still_running; then
        echo "[run_tcga_extract_patches] download job $DOWNLOAD_JOB_ID still running -- more slides may land, sleeping 180s"
        sleep 180
    else
        echo "[run_tcga_extract_patches] download job no longer running -- one final pass to confirm nothing's left"
        python3 src/scripts/24_tcga_validation.py patches --config configs/base.yaml --workers 40 \
            > /lustre/nvwulf/scratch/schennareddy/TCGA/logs/extract_pass_final.log 2>&1
        tail -5 /lustre/nvwulf/scratch/schennareddy/TCGA/logs/extract_pass_final.log
        break
    fi
done

echo "[run_tcga_extract_patches] DONE"
