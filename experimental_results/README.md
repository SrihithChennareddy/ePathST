# Experimental results

Curated, final results for this project -- the tables/figures/numbers
that would actually go into a paper or report, as opposed to
`experimental_runs/` (sibling directory), which holds raw, dated
per-job snapshots captured automatically by `scripts/snapshot_experiment.py`
for cross-run comparison and debugging.

**Distinction from `experimental_runs/`:**
- `experimental_runs/<run_id>/` -- one folder per SLURM job, auto-captured,
  everything that job produced (results/, figures/, config used, a
  headline metric). Point-in-time, may be partial, many of these will
  exist over the life of the project.
- `experimental_results/` (this folder) -- hand-curated, final/reportable
  results only. Fewer files, each one meant to be cited or read on its
  own, each one traceable back to the specific run(s)/job ID(s) it came
  from. Never fabricated or estimated -- every number here must be
  copied from a real executed run's actual output (per the project's
  CRITICAL RULE), same as everywhere else in this repo.

## Layout (populate as results become final)

```
experimental_results/
  README.md                 this file
  cptac_distillation/        final CPTAC distillation numbers (S0/S1/S2,
                              cosine similarity, retrieval@K, per-cancer
                              breakdown) once training converges/locks
  hest_downstream/            final per-cancer (IDC/COAD/CCRCC) A0-A9
                              results tables, the 3 primary gates, figures
  poc5/                       already-final PoC_5 results -- see
                              docs/POC5_RESULTS.md for the source writeup;
                              a curated table extracted here if/when
                              directly comparing against the current
                              project's numbers
```

Each subfolder, when added, should include a short `SOURCE.md` (or a
header in its own README) naming the exact job ID(s)/script(s) the
numbers came from -- mirroring `docs/COMPUTE_RESOURCES.md`'s and
`docs/POC5_RESULTS.md`'s citation style, so anything here stays traceable
back to a real, reproducible run.

Nothing has been placed here yet -- CPTAC distillation is still training
(see `docs/RUN_STATUS.md`) and the HEST downstream CV hasn't been run at
full scale yet (see `docs/V1_TO_V2_MIGRATION.md`'s Known follow-ups).
