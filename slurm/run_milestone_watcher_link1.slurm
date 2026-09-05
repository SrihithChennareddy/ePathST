#!/bin/bash
#SBATCH --job-name=cptac_milestone_watch
#SBATCH --account=pn_alyu012826n
#SBATCH --partition=b40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_milestone_watch_%j.out
#SBATCH --error=/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/logs/slurm_milestone_watch_%j.err

# CPU-only, no GPU requested. Bridges milestone-checkpoint saving (every
# 10 completed epochs -> a permanent {baseline}_seed{seed}_epochN.pt) for
# job 83825 (chain link 1), which already loaded the training code into
# memory BEFORE src/sts/distill_train.py was updated to save these natively.
# Chain links 2 (83826) and 3 (83827) start fresh Python processes and
# will pick up the code change on their own -- this watcher is only
# needed for link 1's remaining lifetime. Submitted as a real SLURM job
# (not a background shell process) so it survives terminal disconnection
# exactly like the training jobs themselves.

set -euo pipefail
cd /lustre/nvwulf/home/schennareddy/cellvit_project/STS_src
module load pytorch/2.7.1

python3 - <<'PYEOF'
import time
from pathlib import Path
import torch

CKPT_DIR = Path("/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/cptac_genbio_distill_hest_pna/student_checkpoints")
RESUME_PATH = CKPT_DIR / "S1_seed0_resume.pt"
POLL_SECONDS = 120

seen_milestones = set(
    int(p.stem.split("epoch")[-1]) for p in CKPT_DIR.glob("S1_seed0_epoch*.pt")
)
print(f"[milestone_watcher] already-saved milestones: {sorted(seen_milestones)}", flush=True)

while True:
    if RESUME_PATH.exists():
        try:
            state = torch.load(RESUME_PATH, map_location="cpu")
            completed = state["epoch"] + 1
            milestone = (completed // 10) * 10
            if milestone > 0 and milestone not in seen_milestones:
                dest = CKPT_DIR / f"S1_seed0_epoch{milestone}.pt"
                val_cos = (state["history"][milestone - 1]["val_cosine_sim"]
                           if len(state["history"]) >= milestone else None)
                torch.save({"model_state_dict": state["model_state_dict"], "epoch": milestone - 1,
                            "val_cosine_sim": val_cos, "baseline": "S1", "seed": 0,
                            "saved_by": "milestone_watcher_external"}, dest)
                seen_milestones.add(milestone)
                print(f"[milestone_watcher] saved epoch {milestone} -> {dest}", flush=True)
        except Exception as e:
            print(f"[milestone_watcher] transient read error (probably mid-write, will retry): {e}", flush=True)
    time.sleep(POLL_SECONDS)
PYEOF
