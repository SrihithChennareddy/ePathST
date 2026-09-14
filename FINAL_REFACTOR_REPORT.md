# ePathST Presentation-Only Refactor — Final Report

Branch: `refactor/publication-quality` (6 commits ahead of `main`, not yet merged/pushed).
Governing spec: `/lustre/nvwulf/home/schennareddy/cellvit_project/fix_code.txt`.
Scope: tracked files only. The untracked `src/pathology_st/a10_hiertme/` package, 8
untracked `src/scripts/a10_*.py` scripts, and ~80 untracked root-level `.tmp_*.py`
scratch files were explicitly excluded per an earlier scoping decision and are
unchanged.

## Commits

| Commit | Pass | Summary |
|---|---|---|
| `a67c3bf` | 2 | Formatting for the DATA-group modules (first batch, committed separately) |
| `8007f86` | 2 | Formatting + comment/docstring compression, remaining 57 files |
| `a94994f` | — | Added `docs/style_references/` (research notes on the 5 style-reference repos) |
| `7261692` | 3 | Renamed generic 2-letter batch-tensor variables |
| `f0361c2` | 4 | Package rename `pathology_st`→`epathst`, subpackage reorganization, 3 merges |
| `7b630e7` | 5 | `pyproject.toml`, README install/structure sections, one doc-accuracy fix |

## 1. Final repository structure

```
src/
├── epathst/            (28 modules across 8 subpackages)
│   ├── __init__.py, utils.py, ablations.py      (cross-cutting, stay at root)
│   ├── data/            cptac_select.py, wsi_patches.py, hest_prep.py, splits.py,
│   │                    expression.py, pathways.py, feature_cache.py
│   ├── models/          teacher.py, student.py, patch_head.py, graph_layers.py,
│   │                    graph_model.py
│   ├── graph/           builder.py, destruction.py, training.py
│   ├── distillation/    training.py, evaluation.py
│   ├── evaluation/      metrics.py, analysis.py, pathway_posthoc.py,
│   │                    genbio_comparison.py
│   ├── statistics/      stats_tests.py
│   ├── efficiency/      efficiency.py
│   └── reporting/       figures.py, report.py
└── scripts/             01-23 numbered pipeline stages (unchanged, in place),
                         run_genbio_hest_baseline.py (unchanged, in place),
                         tools/snapshot_experiment.py, tools/debug_idc_pipeline.py
```

`configs/`, `docs/`, `design/`, `tests/`, `experimental_runs/`, `slurm/` are
otherwise unchanged in location. `src/pathology_st/` still exists on disk,
containing only the untracked, out-of-scope `a10_hiertme/` package.

## 2. Major organization changes

- **Package rename**: `pathology_st` → `epathst` (matches the released model name,
  pre-approved as safe by the governing spec since it's a pure rename with zero
  behavioral surface).
- **Subpackage split**: the previously flat 31-module package is now organized by
  responsibility into 8 subpackages (see tree above), with `ablations.py`/`utils.py`
  staying at the root because both are genuinely cross-cutting (imported by nearly
  every subpackage).
- **Scripts stay flat and numbered** — the ordering communicates reproducibility and
  was deliberately not reorganized into subdirectories. Only the two non-pipeline
  utility scripts moved, to `scripts/tools/`.
- **`pyproject.toml` added** — the package is now pip-installable (`pip install -e .`);
  scripts remain plain `python src/scripts/NN_*.py` invocations, not console entry
  points, since they're pipeline stages meant to run in order, not general commands.

## 3. Files merged (all move-only, no algorithm changes)

| Old files | New file | Rationale |
|---|---|---|
| `patient_split.py` + `cv_folds.py` | `data/splits.py` | Same conceptual responsibility (leakage-free patient partitioning), no coupling between them, called from adjacent pipeline stages. |
| `pna_block.py` + `gatv2_block.py` | `models/graph_layers.py` | Explicitly documented as structurally parallel (`gatv2_block.py`'s own docstring said it "mirrors PNABlock's structure exactly"). |
| `distill_loss.py` + `distill_train.py` | `distillation/training.py` | `distill_loss.py` had exactly one consumer (`distill_train.py` itself); merging removes a single-purpose file with no loss of separation. |

Two merges considered and **rejected** during the Pass 1 audit (kept separate):
`pathways.py`+`pathway_posthoc.py` (the latter is currently dead code; merging
would misfile the live half), and `teacher_genbio.py`+`genbio_pca.py` (the latter
is a symmetric teacher-vs-student comparison harness, not teacher-specific).

## 4. File rename/move table

| Old path | New path |
|---|---|
| `src/pathology_st/__init__.py` | `src/epathst/__init__.py` |
| `src/pathology_st/utils.py` | `src/epathst/utils.py` |
| `src/pathology_st/ablations.py` | `src/epathst/ablations.py` |
| `src/pathology_st/cptac_select.py` | `src/epathst/data/cptac_select.py` |
| `src/pathology_st/wsi_patches.py` | `src/epathst/data/wsi_patches.py` |
| `src/pathology_st/hest_prep.py` | `src/epathst/data/hest_prep.py` |
| `src/pathology_st/patient_split.py` + `cv_folds.py` | `src/epathst/data/splits.py` (merged) |
| `src/pathology_st/expression.py` | `src/epathst/data/expression.py` |
| `src/pathology_st/pathways.py` | `src/epathst/data/pathways.py` |
| `src/pathology_st/feature_cache.py` | `src/epathst/data/feature_cache.py` |
| `src/pathology_st/teacher_genbio.py` | `src/epathst/models/teacher.py` |
| `src/pathology_st/student_model.py` | `src/epathst/models/student.py` |
| `src/pathology_st/patch_head.py` | `src/epathst/models/patch_head.py` |
| `src/pathology_st/pna_block.py` + `gatv2_block.py` | `src/epathst/models/graph_layers.py` (merged) |
| `src/pathology_st/graph_model.py` | `src/epathst/models/graph_model.py` |
| `src/pathology_st/graph_builder.py` | `src/epathst/graph/builder.py` |
| `src/pathology_st/destruction.py` | `src/epathst/graph/destruction.py` |
| `src/pathology_st/graph_training.py` | `src/epathst/graph/training.py` |
| `src/pathology_st/distill_loss.py` + `distill_train.py` | `src/epathst/distillation/training.py` (merged) |
| `src/pathology_st/distill_eval.py` | `src/epathst/distillation/evaluation.py` |
| `src/pathology_st/genbio_pca.py` | `src/epathst/evaluation/genbio_comparison.py` |
| `src/pathology_st/pathway_posthoc.py` | `src/epathst/evaluation/pathway_posthoc.py` |
| `src/pathology_st/metrics.py` | `src/epathst/evaluation/metrics.py` |
| `src/pathology_st/analysis.py` | `src/epathst/evaluation/analysis.py` |
| `src/pathology_st/stats_tests.py` | `src/epathst/statistics/stats_tests.py` |
| `src/pathology_st/efficiency.py` | `src/epathst/efficiency/efficiency.py` |
| `src/pathology_st/figures.py` | `src/epathst/reporting/figures.py` |
| `src/pathology_st/report.py` | `src/epathst/reporting/report.py` |
| `src/scripts/snapshot_experiment.py` | `src/scripts/tools/snapshot_experiment.py` |
| `src/scripts/debug_idc_pipeline.py` | `src/scripts/tools/debug_idc_pipeline.py` |
| all 23 numbered scripts + `run_genbio_hest_baseline.py` | unchanged path |

~90 import occurrences across the library, all 26 scripts, and `tests/` were
rewritten to the new `epathst.<subpackage>.<module>` paths.

## 5. Symbol rename summary

| Old | New | Reason |
|---|---|---|
| `genbio_pca.py` (module) | `genbio_comparison.py` | Old name read as "PCA utility for GenBio"; real role is a teacher-vs-student comparison harness. |
| `teacher_genbio.py` / `student_model.py` (modules) | `teacher.py` / `student.py` | Redundant suffixes once nested under `models/`. |
| `p` (local, 4 constructors in `graph_model.py`/`graph_training.py`) | `pna_cfg` | `p` was easily confused with "patient," used elsewhere in the same files. |
| `L` (accumulator, `21_results_summary_doc.py`) | `lines` | Read dozens of times across a ~200-line function. |
| `xb`/`gb`/`pb`/`xv`/`gv`/`pv`/`pw` (locals, patch-head training loops in `10_run_patient_cv.py` and duplicated in `run_genbio_hest_baseline.py`) | `x_train_batch`/`gene_train_batch`/`pathway_train_batch`/`x_val_batch`/`gene_val_batch`/`pathway_val_batch`/`pathway_weight` | Several visually-similar 2-letter names in the same ~30-line block. |
| `xb`/`gb`/`pb` (locals, `run_genbio_hest_baseline.py`'s smoke test) | `x_smoke_batch`/`gene_dummy_target`/`pathway_dummy_target` | Distinct naming from the real training loop's identically-shaped tensors — these are literal zero placeholders, not real data. |

Not renamed, deliberately: `S1-S9` sanity-check IDs, `A0-A9` ablation names — real,
load-bearing project vocabulary referenced across dozens of files.

## 6. Tests and verification performed

- **PASS 0 baseline**: recorded deterministic outputs for graph construction, edge
  features, degree-preserving rewire, bootstrap CI, and BH-FDR (fixed seeds), plus a
  full test-suite run (9/9 passing), before any edit.
- **After every pass**: `python3 -m py_compile` across every in-scope file; full
  `tests/test_graph_and_stats.py` suite (9/9); the same deterministic baseline
  script re-run and diffed byte-for-byte against the Pass-0 recording — identical
  after every single pass.
- **After Pass 4 specifically** (the highest-risk pass): additionally imported all
  28 `epathst` modules standalone and all 26 in-scope scripts' top-level code
  (catches import errors the test suite's narrow coverage wouldn't reach), and
  re-ran `debug_idc_pipeline.py` end-to-end (a self-verifying bug-reproduction
  script) to confirm its assertions still pass identically
  (`before_fix_fisher_z_mean_patient_R≈0.084`, `after_fix≈0.9999`, `bug_confirmed:
  true` — same numbers as before the refactor).
- **Pass 6**: an independent fresh reviewer (no context from having made the
  edits) audited the complete diff across all 6 commits for accidental behavioral
  changes, and independently re-ran the same verification suite.

## 7. Bugs discovered incidental to the move, fixed (path-only, not scientific)

Moving `snapshot_experiment.py` and `debug_idc_pipeline.py` into `scripts/tools/`
changed their depth relative to the repo root, which broke their
`Path(__file__).resolve().parents[N]` computations (used for `sys.path` setup and
locating `experimental_runs/`/output directories). `N` was incremented by 1 in
each case to preserve the exact original resolved paths. This is a mechanical
consequence of the file move, not a scientific-behavior change — verified by
re-running `debug_idc_pipeline.py` and confirming identical output.

## 8. Anything intentionally left unchanged because of functional risk

- **`graph_model.ScaleAwareGraphCorrectionModel`'s class docstring** (a 22-line
  formal math specification of the A8 architecture) was explicitly exempted from
  the "single-line comment" rule — compressing it would destroy the actual
  per-branch equations, gate-softmax formula, and edge-routing contract, not just
  reformat them.
- **Dual config sources** (`configs/base.yaml` hand-mirrored to `base.json`,
  `load_config` silently preferring `.json`) — flagged by the Pass 1 audit as a
  real footgun (already caught once per `docs/RUN_STATUS.md`), but fixing the
  loading mechanism itself was judged out of presentation-only scope; not touched.
- **A3-vs-`train_patch_head` code duplication** in `10_run_patient_cv.py` — Agent 2
  suggested extracting the shared training loop, but doing so changes structure,
  not just formatting; left as two near-identical blocks.
- **`figures.py`/`analysis.py`/`student_model.DownstreamProjection` dead code**
  (`fig2/fig3/fig6/fig8`, `multiscale_contribution`, `per_pathway_delta`) — zero
  call sites, self-documented as not-yet-wired follow-up work; not deleted, per
  the "do not delete uncertain files, flag them" rule.

## 9. Suspected bugs discovered but NOT fixed (recorded only, per the absolute rule)

1. `20_build_final_results.py` reads `results/genbio_teacher_param_count.json`,
   but no script in the repo writes that exact filename — if missing,
   `teacher_params` stays `None` and an unguarded f-string later raises `TypeError`.
2. `12_primary_hypothesis_test.py` rebuilds `cv_grid` without calling
   `exclude_targeted_panel_samples` first, unlike script 10 and
   `run_genbio_hest_baseline.py`, which both do.
3. `06_eval_distillation.py` evaluates S0 through a randomly-initialized, never-trained
   `ProjectionHead` — its reported cosine similarity isn't a meaningful baseline number.
4. `wsi_patches.py`'s `OpenSlideReader.read_region_at_mpp` and
   `WsiDicomReader.read_region_at_mpp` have different mpp fallback defaults (0.25
   vs 0.5) for the otherwise-identical function.
5. `experimental_runs/INDEX.csv` has a stale-value bug (jobs 88050/88100 both show
   job 84871's headline metric) traced to `snapshot_experiment._headline_metric`
   reading whatever result file currently sits in the shared work_dir; also
   missing 3 of 8 snapshot rows.
6. **Key-name mismatch**: `12_primary_hypothesis_test.py` writes
   `scale_aware_vs_local`; `18_final_interpretation.py` checks for
   `multiscale_vs_local` (never written) — one of the five auto-generated final
   interpretation conclusions always reports "missing, rerun script 12," even on a
   complete run. The single most "worth fixing later" item found across the audit.
7. **`debug_idc_pipeline.py`'s `OUT_DIR`** resolves to `src/results/debug_idc_pipeline/`
   (verified by direct execution), not the repo-root `results/debug_idc_pipeline/`
   where an older, unrelated `T1_alignment_bug.json` (dated 2026-08-31) already
   sits — this mismatch predates this refactor (confirmed identical in the
   pre-refactor `parents[1]` value from the file's original location) and was
   preserved exactly, not fixed.

## 10. Out-of-scope governance note

The untracked `src/pathology_st/a10_hiertme/` package (17 files) and 8 untracked
`src/scripts/a10_*.py` scripts import several of the now-moved core modules
(`utils`, `cv_folds`, `expression`, `student_model`) by their old `pathology_st.*`
path. These imports are now broken and will need a small fix (a handful of import
lines) whenever that subsystem is picked back up. This was a deliberate
consequence of respecting the explicit out-of-scope boundary rather than editing
untracked, unreviewed code — not an oversight.

## 11. Unresolved cleanup recommendations

- Fix the `scale_aware_vs_local`/`multiscale_vs_local` key mismatch (bug #6 above)
  — highest-value, lowest-risk fix identified.
- Resolve the dual `base.yaml`/`base.json` config-source footgun (§8).
- Fix A10's broken core-module imports whenever that subsystem resumes (§10).
- Consider deciding whether `src/pathology_st/a10_hiertme/` should be committed
  properly or gitignored deliberately — it currently sits on disk unprotected.

## 12. Final statement

1. **Did any scientific functionality change?** No. Every algorithm, tensor
   operation, mathematical expression, and control-flow order is byte-for-byte
   the same code, only relocated and/or reformatted.
2. **Did model architecture change?** No. All model classes, layer definitions,
   initialization, and forward-pass logic are unchanged; only their file location
   and, in two files, their class/module names changed.
3. **Did evaluation/statistical methodology change?** No. Metrics, statistical
   tests, and their parameters are unchanged; verified byte-identical via the
   deterministic baseline check after every pass.
4. **Did efficiency methodology change?** No. `efficiency.py`'s logic is unchanged,
   only relocated to `efficiency/efficiency.py`.
5. **Do all available tests pass?** Yes, 9/9, after every pass and in the final
   state.
6. **Were deterministic baseline checks preserved?** Yes, byte-identical after
   every pass, including the final state.
