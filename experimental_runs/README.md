# Experimental runs

Captured, point-in-time snapshots of this project's actual experiment
results (never fabricated numbers -- everything here was copied out of a
real completed or in-progress run's `results/`, per the project's CRITICAL
RULE). This is the place to look for cross-run/cross-seed/cross-ablation
comparison later; `docs/reports/RUN_STATUS.md` is the operational recovery log
(job IDs, what to do after a crash) -- this folder is the scientific
record.

## Layout

```
experimental_runs/
  README.md          this file
  INDEX.csv           one row per snapshot: run_id, date, job_id, stage,
                       partition/GPU, headline metrics, notes
  <run_id>/            one folder per snapshot (see naming below)
    manifest.json       what was captured, from where, when, and the exact
                         config used (so a snapshot is self-describing even
                         detached from this repo's later state)
    results/            copy of the run's results/ directory as of capture
                         time (distillation_training_summary.json,
                         distillation_evaluation.json, model_efficiency.json,
                         cv_ablation_metrics.csv, primary_hypothesis_test.json,
                         final_results_table.csv, etc. -- whichever exist)
    figures/             copy of results/ figures as of capture time, if any
    notes.md             free-text: what changed since the last snapshot,
                         anything noteworthy about this run
```

`run_id` naming: `YYYY-MM-DD_job<slurm_jobid>_<short_label>`, e.g.
`2026-08-28_job83723_cptac_distill_b40x4`.

## How to add a snapshot

Run `scripts/snapshot_experiment.py` (see its docstring for options) after
any run reaches a milestone worth preserving -- not just at the very end.
Good snapshot points: after Step 4 (teacher embedding) completes, after
Step 5 partially/fully converges, after a full CV pass over the HEST
ablation grid, after a bug fix that changes results. It:

1. Copies `results/` (and `figures/` if present) from the run's `work_dir`
   into a new `experimental_runs/<run_id>/` folder.
2. Writes `manifest.json` recording the source paths, the exact
   `configs/base.json` used, the SLURM job ID/partition/GPU if known, and
   a timestamp.
3. Appends one row to `INDEX.csv`.

Never hand-edit numbers into `INDEX.csv` or a snapshot's `results/` --
only `snapshot_experiment.py` (or a manual `cp` from a real `results/`
directory) should populate this folder.

## Comparing across runs later

Each `results/final_results_table.csv` (once produced by
`scripts/17_make_results_tables_and_figures.py`) has the
`MODEL x {HVG50, HVG1000, PATHWAY}` table with patient-level mean +/- 95%
CI. Load several snapshots' tables side by side (e.g. by
`run_id`) to compare seeds, ablations, or before/after a code change --
that is the intended use of keeping multiple dated snapshots rather than
only the latest.
