# Run status & recovery log

Living document tracking the real (non-simulated) execution of the CPTAC
distillation pipeline: what's been submitted, what happened, what's
resumable, and how to recover after a failure/interruption. Update this
file whenever a job is submitted, finishes, fails, or a config/path
changes -- it should always be enough on its own to reconstruct "what do I
do next" without re-deriving anything from chat history.

## Key paths (2026-08-28)

| what | path |
|---|---|
| source code | `/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src` |
| config | `STS_src/configs/base.yaml` (mirrored to `base.json`) |
| CPTAC input data | `/lustre/nvwulf/scratch/schennareddy/CPTAC/data/{BRCA,COAD,ccRCC}/{patient_id}/*.dcm` |
| CPTAC metadata | `/lustre/nvwulf/scratch/schennareddy/CPTAC/metadata/` |
| HEST input data | `/lustre/nvwulf/projects/YurovskyGroup-nvwulf/simons_2026` (colon/COAD only so far) |
| **ephemeral work_dir** (patch shards, manifests, QC cache, logs) | `/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/cptac_genbio_distill_hest_pna_work/` |
| **persistent artifacts** (reusable across runs) | `/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/cptac_genbio_distill_hest_pna/` |
| &nbsp;&nbsp;-- GenBio-PathFM teacher embeddings (run-once cache) | `.../cptac_genbio_distill_hest_pna/teacher_embeddings/cptac_teacher_embeddings.h5` |
| &nbsp;&nbsp;-- student checkpoints (S1/S2 best, locked encoders) | `.../cptac_genbio_distill_hest_pna/student_checkpoints/` |
| SLURM job scripts | `STS_src/slurm/run_cptac_distillation.slurm` (H200x4), `run_cptac_distillation_b40x4.slurm` (b40x4, 1 GPU) |
| SLURM stdout/stderr | `STS_src/logs/slurm_b40_<jobid>.{out,err}` or `slurm_<jobid>.{out,err}` |

**IMPORTANT -- do not use `/lustre/nvwulf/home/schennareddy` or
`/lustre/nvwulf/scratch/schennareddy` for new large outputs.** `lfs quota -u
schennareddy /lustre/nvwulf` reports a 20GB per-user quota that is already
~9x over (≈200GB used) across home+scratch. All large pipeline outputs
were moved to the YurovskyGroup-nvwulf project space, which has a real
10TB group quota (~150GB used as of 2026-08-28).

## Why every stage is resumable

- **Step 1** (`01_select_cptac_wsi.py`): appends each WSI's QC record to
  `work_dir/qc/wsi_qc_cache.jsonl` as it completes; a rerun skips any
  `wsi_id` already in that file.
- **Step 2** (`02_extract_cptac_patches.py`): skips any WSI whose HDF5
  patch shard already exists on disk under `work_dir/cptac_patches/`.
- **Step 4** (`04_run_teacher_genbio.py`): skips any `patch_id` already
  present in the canonical `teacher_embeddings/cptac_teacher_embeddings.h5`
  cache (or, when GPU-sharded, its own shard's partial file). Never
  recomputes an embedding that already exists -- this is the "run GenBio
  only once" guarantee.
- **Step 5** (`05_train_student_distill.py` / `sts/distill_train.train_student`):
  saves a full training-state snapshot (`{baseline}_seed{seed}_resume.pt`
  -- model+optimizer+scaler+step counters) every epoch, separate from the
  best-val checkpoint (`{baseline}_seed{seed}_best.pt`). A rerun with the
  same baseline/seed reloads the resume checkpoint and continues from the
  exact epoch/step it left off, rather than restarting.

**Consequence:** if a job dies for any reason (OOM, wall-clock limit,
node failure, preemption), just resubmit the same `.slurm` script. Do not
delete anything under `work_dir` or `persist_dir` to "start clean" unless
you specifically intend to discard real progress.

**Concurrency warning:** Steps 1-2's caches are append-only files with no
file locking. Do NOT run two Step-1/Step-2 jobs at the same time against
the same `work_dir` -- interleaved writes can corrupt
`wsi_qc_cache.jsonl`/`cptac_patch_manifest.csv`. Step 4's shards are safe
to run concurrently (each writes its own file). If you need two GPU jobs
running simultaneously, make sure only one of them is still inside Steps
1-2 at a time (check the log for "=== Step 3:" before starting a second
job's Step 1-2).

## Run log

### Job 83715 -- h200x4, CANCELLED before starting (never ran)
- Submitted to the `h200x4` partition (4xH200, 8h limit) per the user's
  original request.
- `scontrol show node` on every h200x4-0[1-4] AND h200x8-0[1-4] node
  showed `gres/gpu` fully allocated cluster-wide (0 free H200s anywhere);
  Slurm's backfill predicted a start time ~24h out.
- Cancelled (`scancel 83715`) once a b40x4 (RTX PRO 6000 Blackwell)
  alternative was found with idle GPUs -- avoids two jobs racing on the
  same Step 1/2 caches, and 24h of queue wait was judged worse than
  starting immediately on different-but-adequate hardware (validated
  against this exact GPU model earlier in the project).
- **To retry on H200 later**: `sbatch slurm/run_cptac_distillation.slurm`
  once `sinfo -N -o "%N %G %C" | grep h200` shows a node with free GPUs,
  or just resubmit and accept the queue wait -- resumability means no
  progress is lost either way.

### Job 83721 -- b40x4-09, OOM-KILLED after 59s
- First attempt on b40x4 partition, 28 CPUs, **no `--mem` set**.
- Got Slurm's small default memory allocation; 28 parallel WSI-QC workers
  (each holding a multi-hundred-MB DICOM pyramid + decompressed tile
  buffers) blew past it. `slurmstepd: error: Detected 4 oom_kill events`.
- 18 additional WSIs got QC'd (70 -> 88) before the kill -- preserved by
  the resumable cache, not lost.
- **Fix applied**: added `#SBATCH --mem=220G` (and 20 CPUs instead of 28)
  to `run_cptac_distillation_b40x4.slurm`; also added `--mem=700G` to the
  H200x4 script preemptively for the same reason.

### Job 83723 -- b40x4-09, RUNNING (started 2026-08-28T03:03:35)
- Same script as 83721, with the `--mem=220G` fix.
- Confirmed past the point where 83721 died; Step 1 QC progressing
  (95 -> 100+ WSIs QC'd and climbing at last check).
- **This is the currently active real run as of 2026-08-28.** Check
  `squeue -j 83723` for live status, or `tail -f
  STS_src/logs/slurm_b40_83723.out` for progress.
- Expected sequence once this job is healthy: Step 1 (WSI QC, ~598 WSIs)
  -> Step 2 (patch extraction) -> Step 3 (patient split) -> Step 4
  (GenBio-PathFM teacher embedding, single GPU, no sharding since b40x4
  only granted 1 GPU here) -> Step 5 (ConvNeXt V2-Atto distillation
  training, S1 seed 0 prioritized, time-budgeted to whatever remains of
  the 8h wall clock) -> Step 6 (eval) -> Step 9 (lock student, strip
  teacher-projection head, save final encoder to `persist_dir`).
- Given a single-GPU teacher-embedding pass over ~500 WSIs x up to ~2000
  patches, Step 5 (the compute-dominant stage) will very likely NOT reach
  full convergence (spec wants 30-100 epochs) within the remaining 8h
  budget -- expect a partial, resumable run. **Resubmit the same script
  after this job ends** to continue training where it left off.

## Recovery checklist (read this first after any interruption)

1. `squeue -u schennareddy` -- is anything still running/queued?
2. `sacct -j <jobid> --format=JobID,State,ExitCode,Start,End,Elapsed` --
   how did the last job actually end?
3. `tail -100 STS_src/logs/slurm_b40_<jobid>.out` (or `slurm_<jobid>.out`
   for the H200 script) and the matching `.err` file -- what was the last
   thing it did, and is there an error signature (Traceback, OOM, CANCELLED)?
4. Check what's actually on disk before assuming anything is missing:
   - `wc -l .../work/qc/wsi_qc_cache.jsonl` -- WSIs QC'd so far
   - `ls .../work/cptac_patches/*.h5 | wc -l` -- WSIs patch-extracted so far
   - `python3 -c "import h5py; print(h5py.File('.../teacher_embeddings/cptac_teacher_embeddings.h5')['patch_id'].shape)"` -- patches teacher-embedded so far
   - `ls .../student_checkpoints/*_resume.pt` -- which (baseline, seed) distillation runs have a resume point
5. If the failure was OOM or wall-clock: just resubmit the same `.slurm`
   script (see the two scripts under `STS_src/slurm/`). Resumability
   means this is always safe and never redoes finished work.
6. If the failure looks like a real code bug: fix it in `STS_src/sts/` or
   `STS_src/scripts/`, `python3 -m py_compile sts/*.py scripts/*.py` to
   sanity check, then resubmit.
7. Update this file's Run log with the new job ID and outcome.
