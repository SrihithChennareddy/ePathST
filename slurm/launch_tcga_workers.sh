#!/bin/bash
# Tops the TCGA worker fleet up to the target number of workers per partition (running + pending counted, so it is safe to
# run repeatedly, e.g. to restart after a long outage). Targets sit at the per-user running-job limits of each partition.
# Extra arguments are passed to every worker (e.g. --models-dir ... --tag ...).
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src || exit 1
declare -A TARGET=( [h200x4]=3 [b40x4]=2 [h200x4-long]=2 [b40x4-long]=2 [h200x8]=2 [h200x8-long]=1 )
declare -A TIME=( [h200x4]=08:00:00 [b40x4]=08:00:00 [h200x4-long]=2-00:00:00 [b40x4-long]=2-00:00:00 [h200x8]=08:00:00 [h200x8-long]=2-00:00:00 )
for part in h200x4 b40x4 h200x4-long b40x4-long h200x8 h200x8-long; do
  have=$(squeue -u "$USER" -h -n tcga_worker -p "$part" -t RUNNING,PENDING -o "%i %E" | grep -vc "afterany")
  for ((i=have; i<${TARGET[$part]}; i++)); do
    echo "$part: submitted job $(sbatch --parsable --partition=$part --time=${TIME[$part]} slurm/run_tcga_worker.slurm "$@")"
  done
  echo "$part: ${have} existing primary workers, target ${TARGET[$part]}"
done
