# Compute resources used (for Methods / Acknowledgments)

Ledger of every GPU-hour spent on this project's real pipeline runs,
compiled from `sacct` (SLURM's accounting database) plus
`scontrol`/`nvidia-smi` hardware queries — not estimated or reconstructed
from memory. Update this file whenever a new job completes so the total
stays accurate through publication. Queued/cancelled jobs that never
acquired a GPU are listed separately and excluded from the GPU-hour
total (zero compute consumed).

## Summary (last updated 2026-08-29, job 84150 still in progress)

- **GPU used**: 1x NVIDIA RTX PRO 6000 Blackwell (98GB) per job,
  throughout — no other GPU model contributed compute (H200 was
  attempted but never scheduled; see below).
- **Cluster**: SLURM-managed institutional cluster, `b40x4` partition.
- **Total GPU-hours so far**: **~19.65** across 5 substantially-run jobs
  (83723, 83737, 83825, 83826, 83827), not counting job 84150, which is
  still running and accruing more.
- **What that compute produced**: full CPTAC WSI selection/QC (598
  candidates screened -> 363 selected), patch extraction (564,312
  patches), one complete H-Optimus-0 (1,134,774,272-parameter)
  teacher-embedding pass, and **39 completed epochs** of ConvNeXt V2-Atto
  distillation training (S1/seed0), with validation cosine similarity to
  the teacher climbing 0.7646 -> 0.9019 over that span.
- **Two of the five jobs (83723, 83737) failed** partway through (a real
  bug, since fixed — see `docs/RUN_STATUS.md`) after ~1.3-1.6 GPU-hours
  each; their successful Steps 1-4 work was reused by later jobs rather
  than wasted.
- **Job 84150** (in progress) continues training from epoch 39 toward
  the 50-epoch target; add its final elapsed time to the ledger below
  and refresh this summary once it completes.

## Hardware

All completed pipeline compute in this project ran on a single GPU model:

| GPU | Memory | Cluster partition used | Node(s) |
|---|---|---|---|
| NVIDIA RTX PRO 6000 Blackwell Server Edition | 97,887 MiB (~98GB) | `b40x4` (SLURM) | b40x4-01, -09, -10, -11 (one GPU per job, single-node single-GPU jobs throughout) |

An 8x/4x NVIDIA H200 (141GB) partition (`h200x4`/`h200x8` and their
`-long` variants) was targeted for several early submission attempts
(job IDs 83715, 83730, 83733, 83735, 83816, 83817) but **none ever
started** — the cluster's H200 GPUs were fully allocated cluster-wide at
submission time, and separately the `-long` partition variants proved
structurally lower scheduling priority than their short-partition
siblings on this cluster (see `docs/RUN_STATUS.md` for the full
scheduling investigation). All were cancelled with zero elapsed runtime
and zero GPU-hours consumed. **No H200 compute was used in this
project** — if reporting hardware in a paper, report RTX PRO 6000
Blackwell only.

Suggested Methods-section sentence:
> "All experiments were run on a single NVIDIA RTX PRO 6000 Blackwell GPU
> (98GB) per job, on a SLURM-managed institutional cluster."

## GPU-hour ledger (completed / substantially-run jobs only)

One GPU per job throughout (no multi-GPU jobs completed), so GPU-hours =
wall-clock elapsed hours.

| Job ID | Purpose | Start | End (or "running") | Elapsed | GPU-hours | Outcome |
|---|---|---|---|---|---|---|
| 83723 | CPTAC Steps 1-4 (WSI QC, patch extraction, H-Optimus-0 teacher embedding) + Step 5 attempt | 2026-08-28 03:03:30 | 2026-08-28 04:22:54 | 1:19:24 | 1.32 | FAILED — crashed in Step 5 due to the HDF5 fixed-width-string truncation bug (see RUN_STATUS.md); Steps 1-4's work (QC + patch extraction + teacher embedding) completed and was reused by later runs |
| 83737 | Same pipeline, longer-horizon attempt, ran concurrently with 83723 | 2026-08-28 03:21:24 | 2026-08-28 05:00:17 | 1:38:53 | 1.65 | FAILED — self-detected the same truncation-bug corruption via its own integrity check and crashed cleanly in Step 4's verification |
| 83825 | CPTAC full pipeline (Steps 1-6, chain link 1) — first clean run after the truncation-bug fix | 2026-08-28 11:19:29 | 2026-08-28 19:19:45 | 8:00:16 | 8.00 | TIMEOUT (expected, 8h SLURM wall-clock limit) — re-embedded all 564,312 patches from scratch cleanly, trained S1 seed0 through epoch 14 (val cosine similarity 0.7646 -> 0.8880) |
| 83826 | Distillation training continuation (chain link 2), resumed from 83825's checkpoint | 2026-08-28 19:19:45 | 2026-08-29 03:19:46 | 8:00:01 | 8.00 | TIMEOUT (expected) — trained epochs 15-26 (val cosine similarity -> 0.8972) |
| 83827 | Distillation training continuation (chain link 3), resumed from 83826's checkpoint | 2026-08-29 03:19:48 | 2026-08-29 11:20:16 | 8:00:28 | 8.01 | TIMEOUT (expected) — trained epochs 27-38 (val cosine similarity -> 0.9019) |
| 84150 | Distillation training continuation, resumed from 83827's checkpoint, targeting epoch 50 | 2026-08-29 13:35:55 | *(running as of last check)* | in progress | *(accruing)* | RUNNING |
| **Subtotal (completed jobs only, excludes 84150 in progress)** | | | | **19:38:41** | **~19.65** | |

### Minor/negligible-cost attempts (included for completeness, not the paper total)
| Job ID | Purpose | Elapsed | GPU-hours | Outcome |
|---|---|---|---|---|
| 83721 | First b40x4 attempt, no `--mem` set | 0:00:59 | 0.016 | OOM-killed during Step 1 (CPU/mem misconfiguration, not a GPU-compute cost) |

### Interactive validation session (pre-dates the job ledger above)
Early end-to-end validation of the pipeline against one real CPTAC WSI
and the real H-Optimus-0 teacher (confirming the teacher loads with
1,134,774,272 parameters, embeds real patches correctly, and the
ConvNeXt V2-Atto student trains with genuine learning signal) ran
interactively on an allocated **NVIDIA RTX PRO 6000 Blackwell** (same
model, confirmed via `nvidia-smi`) rather than through a tracked `sbatch`
job, so it has no discrete `sacct` record/job ID. GPU time was on the
order of minutes (single-WSI teacher embedding ~23s for 1664 patches; a
short multi-epoch smoke-test on a few hundred patches) — negligible
against the totals above, but it's the first confirmation that the real
pipeline (not synthetic data) runs correctly end to end.

### Zero-compute attempts (cancelled before acquiring any GPU — excluded from all totals)
83715, 83730, 83733, 83735, 83815, 83816, 83817 — all CANCELLED with
0:00:00 elapsed, either by explicit `scancel` (to avoid concurrent-write
conflicts) or because the target partition never scheduled them (the
H200/`-long` issue above). Listed here only so the full job history is
traceable; zero compute cost.

## Running total (as of last update)

**~19.65 GPU-hours** on NVIDIA RTX PRO 6000 Blackwell across 5
substantially-run jobs (83723, 83737, 83825, 83826, 83827), covering:
full CPTAC WSI selection/QC (598 candidates -> 363 selected), patch
extraction (564,312 patches), one complete H-Optimus-0
teacher-embedding pass, and 39 completed epochs of ConvNeXt V2-Atto
distillation training (S1/seed0). Job 84150 is still accruing GPU-hours
toward the 50-epoch target — add its final elapsed time once it completes.

A CPU-only support job (83876, milestone-checkpoint watcher, 1 CPU/4GB,
no GPU, 8:00:17 elapsed) ran alongside 83825 but is not a GPU-hour cost.
