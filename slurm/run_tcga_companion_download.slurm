#!/bin/bash
#SBATCH --job-name=tcga_companion
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/tcga_companion_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/tcga_companion_%j.err
# Downloads + organizes the TCGA companion data (RNA-seq, TCGA-CDR, CPE purity, cBioPortal, GDC clinical); resumable.
set -uo pipefail
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1
python3 src/scripts/24_tcga_validation.py companion --workers 8 "$@"
