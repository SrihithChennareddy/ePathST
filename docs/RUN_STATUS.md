# Run status & recovery log

See also `experimental_runs/` (sibling to `docs/`) for captured result
snapshots used for cross-run comparison/analysis -- this file is the
operational log (what ran, what broke, what to do next); that folder is
the scientific record (actual numbers). Both SLURM scripts call
`src/scripts/snapshot_experiment.py` automatically at the end of their run.

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
- **Step 5** (`05_train_student_distill.py` / `src/pathology_st/distill_train.train_student`):
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
- **Step 1 completed successfully**: 598 initial WSIs / 598 patients ->
  363 passed hard-filter QC -> 363 selected (below the ~500 target, so no
  QC-score capping applied -- worth investigating exclusion_reasons in
  `qc/cptac_wsi_selection_report.json` later, but not blocking).
- **Step 2 (patch extraction) found a real bug**: `WsiDicomOutOfBoundsError`
  when a sampled patch center is close enough to a WSI's edge that the
  read region extends past the slide boundary (e.g. `position=Point(x=-3,
  y=218)`). Because the per-WSI extraction function wraps its ENTIRE
  patch loop in one try/except, a single edge-adjacent coordinate anywhere
  in a WSI's sample list aborted that WSI's ENTIRE patch set, not just the
  one bad patch. ~7/363 WSIs (~2%) lost to this before the fix landed.
  **Fixed** in `src/pathology_st/wsi_patches.py` (new `_clamp_region` helper, applied
  in both `WsiDicomReader.read_region_at_mpp` and
  `OpenSlideReader.read_region_at_mpp`): shifts the read region to stay
  within the level's bounds instead of raising, trading a small
  (sub-patch-width) shift in the exact patch center for keeping real
  tissue content instead of losing the whole WSI.
- The fix was applied to the source while job 83723 kept running (its
  already-started Python process doesn't pick up the change); the ~7
  WSIs it lost never got a shard file written, so Step 2's resumability
  means a **later resubmission will automatically retry them** -- no
  manual intervention needed, just don't forget this when checking final
  WSI/patch counts against the 363-selected number.
- **Steps 1-3 complete** (as of 03:16): 363 WSIs selected -> 321 WSIs
  successfully patch-extracted (42 lost to the WsiDicomOutOfBoundsError
  bug above, now fixed for future runs) -> 494,268 total CPTAC patches ->
  patient split written for all 3 seeds (e.g. seed2: BRCA 85/11/10
  train/val/test, CCRCC 168/21/21, COAD 38/5/4).
- **Step 4 (GenBio-PathFM teacher embedding) started 03:16:10**: single
  GPU, 493,469 patches to embed (1,664 already cached from the earlier
  validation session). At the ~72 patches/sec measured earlier on this
  GPU family, expect roughly ~1.9h, i.e. finishing around **05:10-05:20**.
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

### Job 83737 -- b40x4-01, RUNNING (started 2026-08-28T03:21:27), 24h budget
- Second, LONGER-HORIZON run requested by the user: target 50-100 epochs
  for full convergence (vs. 83723's partial run), 24h wall clock,
  **without cancelling 83723**.
- `configs/base.yaml`'s `distillation.{max_epochs,min_epochs,
  early_stop_patience}` raised 60/30/10 -> 100/50/15 to actually support
  this target -- shared config, so 83723 also sees the higher ceiling
  (harmless: 83723 is still bounded by its own much shorter
  `--time-budget-hours`, so this doesn't change its real behavior).
- **Runs on `b40x4-long`, NOT h200x8-long as first attempted.**
  h200x8-02 showed 2/8 free H200s, but h200x8-long's backfill scheduler
  predicted a ~24h wait (reason: "reserved for jobs in higher priority
  partitions") -- **-long partitions sit behind their short-partition
  siblings for the same physical GPUs**, so a node showing free GPUs
  right now does not mean a `-long` job can claim them immediately.
  Switched to b40x4-long targeting b40x4-01 (1/4 GPUs free) instead.
- **Also hit, and fixed, a resource-sizing mismatch**: first submission
  requested `--cpus-per-task=32 --mem=400G` (sized for the H200 node) and
  sat `PENDING(Resources)` indefinitely on b40x4-01, which only had ~25
  free CPUs and ~135GB free RAM. Reduced to `--cpus-per-task=20 --mem=100G`
  to match what was actually free; started running immediately after.
  **Lesson: always check the SPECIFIC target node's free CPU/mem via
  `scontrol show node <name> | grep -E "CPUAlloc|FreeMem"` before sizing
  `--cpus-per-task`/`--mem`, not just GPU availability.**
- **Safety design**: this job's script begins with a poll loop (max 4h
  wait) checking `logs/slurm_b40_83723.out` for the `"=== Step 5"` marker
  (or 83723 leaving the queue) before touching Steps 1-4 at all, since
  83723 was still mid-Step-4 (writing directly to the canonical teacher-
  embeddings cache, unsharded) at 83737's submission time. This avoids a
  concurrent-HDF5-writer conflict on
  `persist_dir/teacher_embeddings/cptac_teacher_embeddings.h5`. Once past
  the wait, it reruns Steps 1-3 (safe, resumable, and recovers the ~42
  WSIs 83723 lost to the WsiDicomOutOfBoundsError bug) then Step 4 (only
  new patches) then Step 5 with the remaining ~23h budget.
- Both 83723 and 83737 are `sbatch` batch jobs, not interactive
  `srun`/`salloc` sessions -- they run under the SLURM controller
  independent of any terminal/session, so they continue unattended even
  if this session disconnects. Only live *monitoring* (Monitor-tool
  notifications in a Claude Code session) pauses on disconnect; re-attach
  by tailing the log files in any future session, no functionality lost.
- Job scripts: `slurm/run_cptac_distillation_b40x4.slurm` (83723's, 8h),
  `slurm/run_cptac_distillation_full_convergence.slurm` (83737's, 24h).

### End-of-day status (2026-08-28, ~03:38 in job-relative time)
Both jobs confirmed healthy and left running unattended, per explicit user
request ("are we good to close for the day... ensure both continue even
if I disconnect"):
- **83723** (8h budget): 28:53 elapsed, 0 error signatures, mid-Step-4
  (teacher embedding), on track for the ~05:10-05:20 Step-4-completion
  estimate from earlier.
- **83737** (24h budget): 10:59 elapsed, 0 error signatures, correctly
  parked in its wait loop for 83723's Step 5 marker (by design -- not
  stuck, this is expected until 83723 clears Step 4).
- Both are `sbatch` batch jobs -- confirmed independent of terminal/session
  connection (see the note under Job 83737 above). Safe to disconnect.
- **Next time you check in**: read this file's Run log top-to-bottom for
  anything added after this entry, then run the Recovery checklist below
  regardless of whether anything looks wrong -- it's cheap and confirms
  the real on-disk state rather than trusting any stale summary.

### CRITICAL BUG -- both jobs 83723 and 83737 crashed (found ~05:00-08:00)
**Root cause**: `src/pathology_st/teacher_genbio.cache_teacher_embeddings` created the
teacher-embeddings cache's `patch_id` HDF5 dataset with a FIXED-width
numpy `dtype="S"` string type, sized from whatever the FIRST batch's
longest patch_id string happened to be (71 bytes, as it turned out). This
function is called repeatedly -- once per batch, thousands of times over
a run -- and every subsequent write of a LONGER patch_id (CPTAC patch_ids
are `{dicom_series_uid}_{x}_{y}`; the (x, y) pixel-coordinate suffix
varies in digit count by slide, up to ~6 characters between a slide's
smallest and largest coordinates) got **silently truncated to 71 bytes**,
with h5py/numpy raising no error or warning.

**Effects**:
1. Different (long) patch_ids that happened to truncate to the same
   71-byte prefix collided into identical-looking duplicate entries.
2. Resumability broke silently: a fresh, untruncated patch_id string
   (from `patch_manifest.csv`) never string-matches its own truncated
   form already on disk, so `_already_cached_patch_ids` never recognized
   already-cached long patch_ids as done -- causing job 83737's Step 4 to
   redundantly re-embed ~109 WSIs worth of patches job 83723 had already
   embedded (`n_patches=728901` but only `565178` unique -- 163,723
   silently-corrupted duplicate entries), which its own integrity check
   correctly caught (`no_duplicate_patch_ids: False`) and the script's
   `assert report["passed"]` correctly turned into a hard crash.
3. Job 83723's Step 5 (training) inherited the same corrupted cache from
   disk and crashed with `AssertionError: 135107 patches in manifest have
   no cached teacher embedding` inside `CPTACDistillDataset.__init__` --
   the truncated cached ids never matched the full-length manifest ids
   for any long patch_id.

Both `sacct` confirms FAILED, ExitCode 1:0. **No training epochs had run
in either job** -- the crash happened during dataset construction, before
any checkpoint was written, so there was nothing to lose on this front.

**This was not fundamentally a concurrency bug** (though running two jobs
against the shared cache surfaced it sooner) -- it would eventually have
corrupted even a single solo run, the moment any one run's patch_id
lengths varied enough from its own first batch.

**Fix**: `src/pathology_st/teacher_genbio.cache_teacher_embeddings` now uses
`h5py.string_dtype(encoding="utf-8")` (h5py variable-length strings)
instead of fixed-width `dtype="S"` for the `patch_id` dataset -- no width
limit, verified with a round-trip test (short batch then a much-longer
batch, ids preserved exactly, `no_duplicate_patch_ids: True`). Audited
the other 3 `dtype="S"` call sites in the codebase
(`src/pathology_st/wsi_patches.write_patch_shard`, `src/pathology_st/feature_cache.cache_features`)
-- both write their string dataset in a single one-shot `create_dataset`
call (correctly auto-sized to that exact array, never resized/appended
across separate calls afterward), so they were never at risk of this
class of bug; only the repeatedly-appended teacher-embeddings cache was.

**Remediation taken**:
- Quarantined (renamed, not deleted) the corrupted canonical cache:
  `teacher_embeddings/cptac_teacher_embeddings.h5` ->
  `cptac_teacher_embeddings.CORRUPTED_2026-08-28_truncation_bug.h5`
  (6.8GB, kept as an audit trail; safe to delete once the rebuilt cache
  is verified healthy and no longer needed for reference).
- No `*_resume.pt` files existed for either job (crashed before any
  training step), so nothing to clean up there.
- Resubmitted the 8h job fresh: **83815** (replaces 83723). Step 4 will
  re-embed everything from scratch (~2h again) since the cache was wiped
  -- this is the real, unavoidable cost of the corruption, paid once.
- Same lesson as before applies to node selection: had to explicitly
  target free nodes via `--nodelist` again since availability shifts
  quickly on this shared cluster.

### The 24h job (`run_cptac_distillation_full_convergence.slurm` on b40x4-long) could not be scheduled at all -- switched to a job-dependency CHAIN instead

Every attempt to run the 24h script on `b40x4-long` (both explicit
`--nodelist` targeting and free scheduler choice, across two different
free-GPU nodes) sat `PENDING` with reason `"Nodes required for job are
DOWN, DRAINED or reserved for jobs in higher priority partitions"`,
despite `scontrol show node` confirming a genuinely free GPU on the
target node each time. This is a structural cluster policy, not
transient: **`-long` partitions are lower-priority than their short
siblings for the SAME physical nodes**, and with 100+ jobs permanently
queued on the regular `b40x4` partition, a `-long` job essentially never
gets to claim a node the scheduler might want to give a short-partition
job instead.

**Solution**: abandoned the single-24h-job approach. Instead, chained
THREE 8-hour jobs on the regular (fast-scheduling) `b40x4` partition via
SLURM job dependencies -- functionally equivalent to one 24h run, given
Step 5's already-built resume-checkpoint mechanism (`src/pathology_st/distill_train.
train_student` saves full training state every epoch and auto-resumes),
and Steps 1-4's resumable caches (near-instant on links 2 and 3, since
everything is already extracted/embedded by link 1):

```
83815 (link 1, 8h) --[afterany]--> 83818 (link 2, 8h) --[afterany]--> 83819 (link 3, 8h)
```

Each link is literally `run_cptac_distillation_b40x4.slurm` (no new
script needed) submitted with `--dependency=afterany:<previous_jobid>` --
link 2/3 start automatically the moment the previous link ends (success
OR failure/timeout), and each one's Step 5 call picks up mid-training via
the resume checkpoint rather than restarting. **This chain, not the
retired `run_cptac_distillation_full_convergence.slurm`/`b40x4-long`
approach, is now how to get a ~24h training budget on this cluster.**
To extend further, chain a 4th link the same way:
`sbatch --dependency=afterany:83819 --job-name=cptac_distill_chain4 slurm/run_cptac_distillation_b40x4.slurm`.

### Second scheduling bug: `RealMemory - AllocMem` vs live `FreeMem`
Job 83815 (chain link 1, first resubmission after the truncation-bug fix)
sat `PENDING(Resources)` for ~45+ minutes despite showing as the
TOP-priority pending job on the whole `b40x4` partition with a
confirmed-free GPU on its `--nodelist`-pinned target (b40x4-11). Root
cause: it requested `--mem=220G` (sized for a different node that had
~468GB truly free back when that number was chosen), but
`scontrol show node b40x4-11 | grep -E "RealMemory|AllocMem"` showed only
`RealMemory - AllocMem` ~= 47GB actually allocatable by Slurm's own
scheduler -- the node's live `FreeMem` OS metric (~420GB) looked far more
generous but is NOT what the scheduler checks. **Always compute
`(RealMemory - AllocMem) / 1024` in GB for the SPECIFIC target node
before setting `--mem`, not `FreeMem`.** Fixed: reduced to
`--cpus-per-task=12 --mem=44G` (fits under the tightest free-GPU node's
budget) and dropped the `--nodelist` pin so the scheduler can place the
job on whichever node actually fits. Resubmitted as a fresh 3-link chain:
**83825 (link 1) -> 83826 (link 2) -> 83827 (link 3)**, replacing the
83815/83818/83819 chain. 83825 started running within seconds of this fix.

**Confirmed healthy on real data (2026-08-28 ~12:41)**: 83825 finished
Steps 1-3 in under a minute (QC cache + patch shards survived the earlier
corruption -- only the teacher-embeddings cache needed rebuilding),
re-embedded all 564,312 patches fresh in ~1h20min, and the truncation fix
verified clean: `n_patches=564312, n_unique_patch_ids=564312,
no_duplicate_patch_ids: True, passed: True`. Zero error signatures. Now in
Step 5 (training) with a 6.3h budget. This is the first fully-clean
teacher-embedding pass since the bug was introduced -- the fix holds under
real load, not just the synthetic round-trip test.

### Milestone checkpoints added (per user request, ~17:50)
Only a rolling "best" checkpoint (`{baseline}_seed{seed}_best.pt`,
overwritten whenever val_cos improves) and a rolling "resume" checkpoint
(`{baseline}_seed{seed}_resume.pt`, overwritten every epoch) existed --
no way to compare the model at different training stages afterward.
Added permanent, never-overwritten snapshots every 10 completed epochs:
`{baseline}_seed{seed}_epoch{N}.pt` (N=10,20,30,...100), covering the
requested 20/50/100 examples. Implemented in
`src/pathology_st/distill_train.train_student` -- link 2 (83826) and link 3 (83827)
will save these natively since they start fresh Python processes that
load the updated code.

Link 1 (83825) had ALREADY loaded the old training code into memory
before this fix, so it won't pick up the change mid-run. Bridged with a
separate, lightweight (1 CPU, 4GB, no GPU) SLURM job, **83876**
(`slurm/run_milestone_watcher_link1.slurm`), which polls 83825's
`S1_seed0_resume.pt` every 2 minutes and copies it to a permanent
`S1_seed0_epoch{N}.pt` the moment the epoch count first crosses each new
multiple of 10. Submitted as a real SLURM job (not a background shell
process) so it survives terminal disconnection like everything else.
Safe to let 83876 keep running/idling after link 1 ends -- it costs
nothing and link 2/3 don't need it.

### Chain completed (39 epochs); found a second timing gap; job 84150 continues toward 50
The 3-link chain (83825/83826/83827) ran its full ~24h and ended normally
(`sacct` shows `TIMEOUT`, expected -- Step 5 hits its internal time
budget just before the SLURM wall clock, saves a resume checkpoint, and
exits 0). Final state: **39 epochs completed** (epoch 0-38), S1 seed0,
val cosine similarity climbing steadily 0.7646 -> 0.9019, train loss
0.3479 -> 0.0731. No error signatures in any link. Milestone checkpoints
present and correct: `S1_seed0_epoch10.pt` (val_cos 0.8832),
`_epoch20.pt` (0.8940), `_epoch30.pt` (0.8994).

**Found (not yet fixed) a second timing gap**: links 2 and 3 both hit
their SLURM 8h wall-clock kill while still inside Step 6 (evaluation),
BEFORE reaching Step 9 (lock) or the snapshot step -- `grep "Step 9"` on
both logs returns nothing, and `locked_encoder_seed0.pt`'s mtime is stuck
at link 1's end time. The `RESERVE_SECONDS` budget in
`run_cptac_distillation_b40x4.slurm` only accounts for Step 9 + buffer,
not Step 6's own runtime, so a mid-Step-6 kill loses the rest of the
pipeline for that link. Consequence: **the "locked" final encoder is
stale (frozen at epoch 14)** even though the raw best checkpoint
(`S1_seed0_best.pt`) is current and correct (epoch 38 as of the chain's
end). Not fixed yet since the user hasn't asked for it and nothing
downstream is reading the locked encoder yet -- but re-run
`src/scripts/07_lock_student.py --mode A` manually (fast, no GPU compute
needed beyond one forward pass to strip the head) once training is
where you want it, or fix the time budget in the `.slurm` script to
reserve enough for Step 6 + Step 9 both.

**Job 84150** (`cptac_distill_to50`, submitted per explicit user request:
"make a 8hr run on b40x4 RTX PRO 6000 Blackwell, to continue epochs until
50 epochs"): same script, no code changes, single 8h run (not a further
chain link with a new job name, so it does NOT auto-continue past 8h --
resubmit manually if it doesn't reach epoch 50). Started running
immediately on b40x4-01 (1 free GPU, 287GB free mem confirmed before
submitting). At the ~40min/epoch pace observed in link 3, ~11 more epochs
(39->50) should take ~7.3h, fitting inside the 8h budget -- expected to
either reach epoch 50 or land very close to it.

### HEST Phase-2 (Step 8-12) full 3-cancer run + new GenBio-PathFM baseline (2026-08-30)

All 3 real HEST cancers (breast/IDC, colon/COAD, kidney/CCRCC -- 202
samples total) became available together for the first time this session
(`docs/RUN_STATUS.md`'s earlier entries only had colon). This section
covers: (1) parallelizing Step 8 (the previous bottleneck), (2) building
a second, independent frozen-GenBio-PathFM patch-only baseline for
comparison, and (3) four real, previously-hidden bugs Step 10 hit the
first time it ever ran end-to-end against the full 3-cancer dataset --
Steps 8-12 had never gotten this far before (earlier attempts always hit
an 8h wall-clock cutoff mid-Step-8).

**Job chain**: 84327 (b40x4, 8h) -> hit wall-clock mid-Step-8 -> 84331
(dependent resume) -> also hit wall-clock -> 84527/84530/84531 (manual
resubmits, added parallelization) -> 84717/84722/84733 (successive bug
fixes below, main pipeline) running in parallel with 84343/84344/84715/
84723 (GenBio baseline chain, see its own bugs below). Every step in
this chain is a plain resubmit of the same `.slurm` script after a code
fix -- no manual data surgery was ever needed, confirming Steps 8-12's
resumability design holds under real, repeated failure.

**Step 8 parallelization** (previously ~single-threaded, one WSI region
read at a time -- confirmed the actual bottleneck, not GPU-bound: Step 8
touches no CUDA code path at all):
- `src/pathology_st/hest_prep.extract_hest_patches_for_sample` now reads a sample's
  spots concurrently via a `ThreadPoolExecutor` -- openslide's
  `read_region` releases the GIL during the tile decode, so this is
  genuine I/O parallelism on one WSI handle.
- `src/scripts/08_prepare_hest_metadata.py` now also runs multiple SAMPLES
  concurrently via `ProcessPoolExecutor` (`--sample-workers`, default 4)
  each running its own `--threads-per-sample` (default 6) thread pool --
  4x6=24 matches the job's `--cpus-per-task=24`. Checkpointing (every 5
  completed samples) and misalignment-exclusion semantics unchanged.
- Measured real speedup on the same 4 samples (131-134) that took 59.1
  min combined sequentially: **~7.35 min combined in parallel, ~8x**.
  `--mem`/`--cpus-per-task`/`--time` were also bumped (12->24 CPUs,
  8h->24h on `b40x4-long` since `b40x4` caps at 8h) to give the new
  concurrency room -- see the OOM bug below for why `--mem` needed a
  SECOND, much larger bump on top of this.
- New H200x4 attempt (backup job for a possible faster GPU) was tried and
  abandoned: `h200x4-04` showed 3/4 GPUs free but its RAM was already
  ~99% allocated by another user's job (`AllocMem=768000` of
  `RealMemory=773578`), so the scheduler's backfill estimate for this
  job was ~6h out despite the "free GPU" appearance -- same class of
  lesson as the earlier `RealMemory - AllocMem` vs `FreeMem` bug in this
  file's Job 83815 entry, just for a different partition. Reverted to
  `b40x4`/`b40x4-long`, which has consistently had usable headroom.

**New GenBio-PathFM baseline** (`src/scripts/run_genbio_hest_baseline.py` /
`slurm/run_genbio_hest_baseline.slurm`): a standalone, frozen
GenBio-PathFM (1.1B params, 4608-D embedding) patch-only comparison --
no GNN, no fine-tuning, reuses the main pipeline's exact seed=0/fold=0
split and HVG1000/HVG50/PROGENy14 target definitions via
`src/scripts/10_run_patient_cv.load_fold_expression` (imported via
`importlib`, never copy-pasted, so every Step-10 fix below applies to it
automatically). Writes `results/genbio_baseline/genbio_summary.csv` (one
row per cancer: `HVG50_r/HVG1000_r/PROGENy14_r` + now also
`*_mse`/`*_mae`, see below), plus per-patient-per-gene detail in
`genbio_gene_metrics.csv`/`genbio_pathway_metrics.csv` for later
graphing.

**GenBio bug 1 -- manifest-wait race** (job 84343): read the shared
`hest_samples.csv`/`hest_spot_manifest.csv` once at startup and got a
snapshot showing only IDC (120 samples) since the main pipeline's Step 8
hadn't reached COAD/CCRCC yet -- would have silently run the entire
experiment on 1/3 of the intended cohort. **Fixed**: added
`wait_for_hest_manifest_complete()`, a read-only poll loop that waits
until every configured cancer has at least one QC-eligible sample AND
the count has gone stable across 3 consecutive polls. First version had
a `timeout_seconds` escape hatch (originally 7h) that could still let it
proceed with a partially-missing cancer -- **that itself caused job
84344 to burn 7h08m of GPU time** (proceeded with IDC+COAD only, then
hard-crashed the smoke test on 0 CCRCC samples). **Re-fixed**: the
timeout now only fires once every cancer has at least one sample but the
count hasn't gone fully stable yet -- a cancer that's still completely
absent just waits for the SLURM wall-clock, a clean failure mode instead
of a wasted silent-partial run.

**Step 10 bug 1 -- OOM (job 84714, then 84717)**: `load_fold_expression`
pools EVERY train-fold sample's FULL (non-HVG) dense expression matrix
via `pd.concat` -- IDC fold0 alone is ~92k spots x ~18k genes (~13GB for
one copy), and holding the per-sample DataFrame list plus the
concatenated result simultaneously roughly doubles that peak, well past
the `--mem=44G` sized for the much lighter earlier CPTAC-only jobs.
Confirmed via a real `oom_kill` event. **Fixed**: bumped `--mem` to
160G (b40x4 nodes have 500G+ RAM, confirmed ample headroom).

**Step 10 bug 2 -- gene-ID convention mismatch (job 84717)**:
`AssertionError: No sources with more than tmin=5 targets` from
decoupler's PROGENy scoring. Root cause: some breast/IDC HEST slides
(`SPA1xx`, confirmed live) index `var` by Ensembl gene ID
(`ENSG00000000003`) with the actual symbol in a separate `var['SYMBOL']`
column, while others (`NCBI*`) already index by symbol directly --
0 gene-symbol overlap with PROGENy's symbol-keyed net for the
Ensembl-indexed ones. **Fixed**: `pathology_st.expression.ensure_gene_symbol_index`
detects an Ensembl-style index with a `SYMBOL` column and remaps to it
(dropping genes with no symbol, deduping). Confirmed COAD/CCRCC (a
different data source, HuggingFace HEST-1k) don't have this specific
issue.

**Step 10 bug 3 -- fixed-dimension model vs. variable HVG count (job
84722)**: `RuntimeError: tensor a (1000) must match tensor b (171)`.
Root cause: `load_fold_expression` selects ONE shared HVG1000 list per
cancer (the intersection of every training sample's genes -- necessary
since the model is one network trained across all of that cancer's
pooled training spots at once), but `build_patch_head`/
`GenBioDownstreamHead` were both built with the fixed CONFIG value
(1000) rather than the ACTUAL selected count. The real cause of the
undersized intersection: **11/202 samples (6 IDC, 5 COAD) run on a
Xenium-style targeted ~541-gene panel** instead of whole-transcriptome
ST, and `common_genes = set.intersection(...)` requires every sample in
the fold to have a gene -- a handful of tiny-panel outliers silently
collapsed IDC's shared pool from ~1000 to 171. **Fixed per explicit user
decision** (excluding the outliers, not accepting a degraded HVG171):
`pathology_st.cv_folds.exclude_targeted_panel_samples` (threshold 2000 genes --
real whole-transcriptome panels seen are 14700+, real targeted panels
seen are 541) drops these 11 samples from the CV split entirely in both
`src/scripts/10_run_patient_cv.py` and `src/scripts/run_genbio_hest_baseline.py`,
so the remaining 191 samples (IDC 114, COAD 53, CCRCC 24) all get the
full HVG1000/HVG50, keeping the `HVG1000_r`/`HVG50_r` comparison
apples-to-apples across all 3 cancers. Also fixed the dimension bug
itself in both scripts (`len(hvg["hvg1000"])` instead of the fixed
config value) as defense in depth.

**Step 10 bug 4 -- duplicate gene symbols (job 84722, again)**:
`AssertionError: mat contains repeated feature names` from decoupler,
DIFFERENT root cause from bug 2 above -- 4 real COAD slides
(TENX156/155/154/128) already index by symbol correctly but have a
handful (3/18085) of naturally duplicate gene symbols, which the earlier
`ensure_gene_symbol_index` fix didn't dedupe since it only touched the
Ensembl-remap code path. **Fixed**: that function now unconditionally
calls `adata.var_names_make_unique()` regardless of which path was
taken -- duplicate symbols get suffixed (`GENE`, `GENE-1`, ...) rather
than crashing; the suffixed copies simply won't match any gene-symbol-
keyed net/HVG list, same harmless outcome as any other absent gene.

**MSE/MAE surfaced for GenBio too** (per explicit user request, "for
later"): `pathology_st.metrics.evaluate_predictions` (used by the main pipeline's
`cv_ablation_metrics.csv`) already computed `mean_mse`/`mean_mae`
alongside Pearson r for every target block -- nothing to do there.
GenBio's own summary row only had the `_r` values though (it uses
`per_patient_per_column_pearson`/`target_summary` directly, not
`evaluate_predictions`); added `HVG50_mse/mae`, `HVG1000_mse/mae`,
`PROGENy14_mse/mae` to `genbio_summary.csv`'s summary row (the
per-patient mse/mae values were already being computed either way, just
not aggregated into the summary before).

### MILESTONE: job 84733 completed end to end -- first real Steps 8-12 results (2026-08-30, ~20:27)

**Job 84733 finished cleanly** (`sacct`: COMPLETED, exit 0:0, 39:45
elapsed) -- Steps 8/9b skipped instantly (already cached), Step 10 ran
the full 9-ablation grid (see A1 bug below) across all 3 cancers with no
crash, Step 12 ran and wrote both `primary_hypothesis_test.json` and
`secondary_hypothesis_tests.json`. This is the first fully clean
end-to-end pass this project has ever completed against the real 3-cancer
dataset. Snapshotted to `experimental_runs/
2026-08-30_job84733_hest_downstream_full_3cancer/` (the auto-snapshot
step's stdout said "snapshot failed -- non-fatal", but the actual files
were written correctly regardless -- worth checking why that step's own
success/failure detection is wrong, not investigated further here).

**Fifth bug found from these real results, not a crash this time**:
`A1_imagenet_patch` and `A2_distilled_patch` produced BYTE-IDENTICAL
metrics across all 3 cancers. Root cause: `src/scripts/10_run_patient_cv.py`'s
`main()` loads features exactly once (`feat_dir = paths[
"student_embeddings"] / "distilled"`), before the ablation loop, and
`run_ablation`'s `graph_source == "none"` branch (which handles both A1
and A2) just forward-passes the shared `patch_head` on that same
distilled `x_test` -- A1's declared `"imagenet"` encoder field was never
actually consulted, even though Step 9b DOES separately cache ImageNet
features. Confirmed every OTHER ablation (A0, A3-A9) already specs
`"distilled"` or `"none"` as its encoder, matching what actually gets
loaded, so only A1's own row and Gate 1 (`A2 > A1`, "distillation
helped") were affected -- Gate 2/3 and every other ablation are
trustworthy as-is.

**Per explicit user decision**: rather than fix the feature-loading and
rerun, removed `A1_imagenet_patch` from `pathology_st.ablations.ABLATION_GRID`
entirely, and `gate1_distillation` from `configs/base.yaml`'s
`primary_gates` (also updated `src/scripts/12_primary_hypothesis_test.py`'s
docstring/gate_names and `src/scripts/17_make_results_tables_and_figures.py`'s
`MODEL_ORDER` for consistency). **Important gotcha caught while doing
this**: `pathology_st.utils.load_config` prefers a sibling `.json` over `.yaml`
when both exist, and `configs/base.json` DOES exist -- editing only
`base.yaml` would have silently had NO effect on any future run, since
every job actually reads `base.json`. Regenerated `base.json` fresh from
the edited `base.yaml` (via `yaml.safe_load` + `json.dump`) to keep them
in sync; diffed clean (only the intended `gate1_distillation` key
removed). **Grid is now 9 ablations, not 10** (A0, A2-A9).

**Real headline numbers from this run** (`results/cv_ablation_metrics.csv`,
`hvg1000_fisher_z_mean_patient_R`, single fold/seed -- these are noisy,
provisional numbers from the FIRST completed run, not a converged final
result):

| | A2 (patch-only, distilled) | A3 (matched MLP) | A5 (shuffled, negative control) | A8 (scale-aware PNA, proposed) |
|---|---|---|---|---|
| IDC | 0.0704 | 0.0867 | 0.0043 | 0.0075 |
| COAD | 0.1159 | 0.1071 | 0.0282 | 0.0314 |
| CCRCC | 0.0587 | 0.0656 | -0.0023 | -0.0017 |

**Gate 2 (spatial context, A8 > A2): NOT SUPPORTED** -- the GNN-corrected
prediction is currently much WORSE than the plain patch-only anchor in
every cancer, not better. **Gate 3 (true topology, A8 > A5, the decisive
test): SUPPORTED** -- A8 does beat the shuffled-graph negative control,
but both are extremely small numbers close to each other and close to
zero, so "supported" here is a weak signal, not a strong one. Overall
reading: the patch-only heads (A2/A3) currently carry essentially all of
the (still modest, r~0.06-0.12) predictive signal; the GNN correction
layer isn't adding value yet in this quick 1-fold/1-seed run -- worth
investigating (more folds/seeds? more GNN training epochs given the
time-budgeted run? a real bug in the graph correction path?) before
treating this as a real negative result.

**GenBio baseline (separate job) also hit two more real bugs after
84733 finished**, both now fixed and pushed:
- **Bug 5 (job 84735)**: `build_targets` hard-crashed on `AssertionError:
  IDC: NaN/Inf in standardized targets` -- but this was never a real
  bug: `pathology_st.pathways.compute_progeny_scores` documents that
  decoupler-dropped degenerate spots come back NaN by design ("handled
  by the same missingness-aware QC as any other target"), confirmed live
  (decoupler dropped 1/306 and 1/292 spots as degenerate on this run),
  and the shared `patch_head_loss` (same one `src/scripts/10_run_patient_cv.py`
  uses, which has no equivalent assertion) already masks NaN correctly
  via `gene_mask`/`path_mask`. Narrowed the assertion to only catch real
  `Inf` values.
- **Bug 6 (job 84736)**: OOM-killed on COAD after successfully finishing
  IDC's training (confirming the NaN fix held) -- this script's own
  `--mem` was never bumped to 160G when the main pipeline's was, despite
  sharing the exact same `load_fold_expression` memory footprint. Fixed
  (`slurm/run_genbio_hest_baseline.slurm`, 44G -> 160G).
- **Operational hiccup, not a code bug**: while resubmitting after the
  memory fix, `sbatch` returned `error: Unexpected message received`
  (a transient SLURM controller communication glitch) on two consecutive
  attempts -- but both submissions actually succeeded server-side despite
  the client-side error, and a third retry produced a THIRD job. Ended up
  with 3 concurrent `genbio_hest_baseline` jobs (84737, 84739, 84743)
  racing against the same shared embedding-cache/output paths for ~10
  minutes before caught via `squeue`. Cancelled the two extras,
  verified no corrupted `.h5` embedding shards resulted (checked every
  cached file opens cleanly with `h5py`), then cancelled the
  survivor too and did one single clean resubmit (**84744**) rather than
  trust a run that spent part of its life racing another process --
  cheap to redo since extraction is ~fully cached. **Lesson: an `sbatch`
  client-side error does NOT reliably mean the submission failed on this
  cluster -- always `squeue` and check for a duplicate before retrying.**

### MILESTONE 2: job 84744 completed -- first real GenBio-PathFM baseline results (2026-08-30, ~20:46)

**Job 84744 finished cleanly** (`sacct`: COMPLETED, exit 0:0, 7:20
elapsed -- fast since extraction was ~fully cached). All 6 accumulated
fixes held under the real full-3-cancer run: manifest-wait, targeted-
panel-exclusion, gene-symbol mapping, var_names dedup, NaN-assertion
relaxation, memory bump. Snapshotted to `experimental_runs/
2026-08-30_job84744_genbio_baseline_3cancer/` (copied directly, since
this script doesn't call `snapshot_experiment.py` the way the main
pipeline's slurm script does).

**Real results** (`results/genbio_baseline/genbio_summary.csv`, single
fold/seed, frozen GenBio-PathFM patch-only, no GNN):

| Cancer | HVG50 r | HVG1000 r | PROGENy14 r | HVG1000 MSE | HVG1000 MAE |
|---|---|---|---|---|---|
| IDC | 0.1478 | 0.1053 | -0.0157 | 0.998 | 0.833 |
| COAD | 0.3357 | 0.1941 | -0.0023 | 0.661 | 0.631 |
| CCRCC | 0.2058 | 0.1227 | 0.0240 | 1.076 | 0.836 |
| **MEAN** | **0.2297** | **0.1407** | **0.0020** | -- | -- |

Interesting comparison point against the main pipeline's A2 (distilled
CNN, patch-only, same architecture role): GenBio-PathFM's HVG1000 r is
meaningfully HIGHER in every cancer (IDC 0.105 vs 0.070, COAD 0.194 vs
0.116, CCRCC 0.123 vs 0.059) -- the frozen 1.1B-param foundation model's
patch embeddings currently carry more gene-expression signal than the
project's own distilled 3.4M-param CNN encoder, at least for this
one-fold/one-seed run. PROGENy14 r is near-zero-to-slightly-negative for
both approaches in IDC/COAD, essentially no pathway-level signal yet.

**As of this entry**: both pipelines have now completed at least one
full clean pass. Main pipeline (84733) and GenBio baseline (84744) real
results are both above. Nothing currently running (`squeue -u
schennareddy` shows only an unrelated `interactive` session, 84713).
Six total real bugs found and fixed across both pipelines this session
(gene-symbol mapping, OOM x2, targeted-panel exclusion, duplicate
var_names, NaN-assertion) plus the A1/Gate1 removal and one operational
sbatch-reliability incident -- see above for details on each. Next
steps (not started): more folds/seeds for a less noisy result (this was
all `--max-folds-per-cancer 1 --seeds 0`, not the full V2 Phase 3
5-fold x 3-seed design), and investigating why the GNN correction layer
(A8) currently underperforms the patch-only anchor (A2) rather than
improving on it.

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
6. If the failure looks like a real code bug: fix it in `STS_src/src/pathology_st/` or
   `STS_src/src/scripts/`, `python3 -m py_compile src/pathology_st/*.py src/scripts/*.py` to
   sanity check, then resubmit.
7. Update this file's Run log with the new job ID and outcome.
