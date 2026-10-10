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
  real footgun (already caught once per `docs/reports/RUN_STATUS.md`), but fixing the
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

---

# Round 2 (2026-10-09): whole repository, script consolidation, H-Optimus-0 naming, figure code

Same governing spec (`fix_code.txt`), applied to everything written after round 1: scripts 24-53, the TCGA and Leiden
pipelines, tests, SLURM files and the paper-figure code. Nothing is committed yet.

## 1. Structure after round 2

```
src/epathst/
├── data/ models/ graph/ distillation/ statistics/ efficiency/ reporting/   (unchanged from round 1)
├── evaluation/   + ablation_runner.py (from script 10), official_hest.py (from script 22),
│                   leiden.py (script 35), leiden_cohort.py (script 37)
├── tcga/         gdc.py (was data/tcga.py), cases.py (24), download.py (25), patches.py (28),
│                 prediction.py (38), hexgrid.py (40), worker.py (43), audit.py (44), companion.py (50)
├── audits/       panel_coverage.py (41), frozen_weights.py (42), training_samples.py (53),
│                 leiden_ari.py (45), leiden_reproducibility.py (46)
└── cli.py        stage dispatch for the three drivers below
src/scripts/      01-23 unchanged in place (04, 15 renamed), 24_tcga_validation.py, 25_leiden_official_panel.py,
                  26_hest_audits.py, run_hoptimus_hest_baseline.py, tools/, legacy/
figures/          one script per paper figure, inference/ for their GPU inputs, data/ (small inputs)
tests/            unit tests, regression_baseline.py (determinism digests), e2e_one_slide.py, e2e_figures.py,
                  e2e_embed_equivalence.py
```

Numbered top-level scripts went from 47 (01-53 with gaps, plus 51/52) to 26. The 01-23 pipeline keeps its numbering, as
the spec requires.

## 2. How the merge was done

Scripts 22, 23, 35, 37, 40, 43, 44, 45, 46 and the baseline loaded each other by file path with `importlib`
(10 -> 22 -> 23/35 -> 37/45/46; 38 -> 40 -> 43 -> 44). The reusable definitions were moved verbatim into the package and
every caller now imports them normally. Each former script's `main()` became a library entry point, and three drivers call
them with the original options unchanged (`24_tcga_validation.py worker --limit 1` takes exactly what
`43_tcga_worker.py --limit 1` took). Scripts 32, 33, 34, 36 and 39 are earlier variants kept in `legacy/` with only their
imports fixed. Old name -> new command: `docs/reports/RESULTS_INDEX.md` section 5.

Script 35 patches `predict_graph_model` inside the module that defines `run_ablation`, so it can save the fitted ePathST
model. That patch now targets `epathst.evaluation.ablation_runner` and is applied when `epathst.evaluation.leiden` is imported,
exactly as importing script 35 applied it before.

## 3. Renames

| Old | New | Why |
|---|---|---|
| `04_run_teacher_genbio.py`, `15_genbio_fair_comparison.py`, `run_genbio_hest_baseline.py` (+ SLURM) | `04_run_teacher_hoptimus.py`, `15_hoptimus_comparison.py`, `run_hoptimus_hest_baseline.py` | teacher is H-Optimus-0 |
| config keys `baselines.S1_genbio_distilled`, `genbio_fair_comparison` | `S1_hoptimus_distilled`, `hoptimus_comparison` | same; values unchanged, only 05 reads the first |
| inputs read by 20/21: `genbio_baseline/genbio_summary.csv`, `genbio_teacher_param_count.json`, encoder `genbio_pathfm` | `hoptimus0_baseline/hoptimus0_summary.csv` (what the baseline script writes), `hoptimus0_teacher_param_count.json`, `h_optimus_0` (what script 19 writes) | readers now match the current writers |
| `_row_for`, `_decode_barcodes` (script 22) | `row_for_sample`, `decode_barcodes` | now imported across modules |
| module aliases `s38`, `s40`, `s43` | `prediction`, `hexgrid`, `worker` | readability |
| log prefixes `[35_...]`, `[37]`, `[38]`, `[40]`, `[43 ...]`, `[50 ...]` | module names | the scripts no longer exist under those names |
| `tests/test_tcga_embed_equivalence.py` | `tests/e2e_embed_equivalence.py` | GPU script with argv; pytest collected it and failed |

GenBio-PathFM no longer appears in any code, config or SLURM file. It remains in the docs and in `figures/data/*.csv`
as one of the compared foundation models, and in the file names of the `experimental_runs/` snapshots, which are records.

## 4. Formatting and comments

black (line length 100) over every in-scope file, long docstring paragraphs and comments rewrapped, long string literals
split into adjacent literals, section banners removed, and a comment/docstring pass in the spec's style (four parallel
reviewers, comments and docstrings only). After each pass every file was checked:
whole-module AST identical to the previous state (ignoring docstrings for the comment passes, not ignoring them for the
string splits), and for the rewrap pass the exact word sequence of every docstring and comment preserved.

## 5. Behaviour checks (before = frozen copy of the tree taken before any round-2 edit)

| Check | Before | After |
|---|---|---|
| Unit tests (`test_graph_and_stats.py`, `test_tcga_queue_claims.py`) | 9/9 | 9/9 |
| Determinism digests, 21 sections (graph edges and features per scale, random/rewired/partial graphs, ablation graphs, bootstrap CI, sign-flip, BH-FDR, patient Pearson/Fisher-z, Moran's I, CPTAC split, CV grid, parameter counts and forward outputs of patch head, union PNA, GATv2, scale-aware PNA, residual predictor, student encoder) | recorded | byte-identical |
| One TCGA slide per cancer, end to end on GPU (embeddings from the WSI vs cache; predictions vs the stored production files) | PASS, max diffs 2.4e-3 to 3.9e-3 | PASS, the same numbers to every printed digit |
| Every moved function, class and constant | | AST-identical to its original (aliases and the two renames normalised) |
| Paper figures regenerated (15 images) | | pixel-identical to the files in `paper_figures/` |
| Leiden, official panel, COAD fold 0, old script 35 vs `25_leiden_official_panel.py official` (same GPU, same seed, scratch output) | ePathST-E 0.3729, ePathST 0.3574, H-Optimus-0 0.3234 | ePathST-E and H-Optimus-0 predictions bit-identical (0.3729, 0.3234); ePathST 0.3566. The graph model's GPU training is not bit-reproducible: two earlier identical seed-0 runs of the original script gave 0.3633 and 0.3528 |

Parameter count of the student encoder: 3,387,400 (paper: 3.39 M).

## 6. Deliberate changes to how things are invoked (not to what they compute)

- The merged scripts run as stages of the three drivers; every SLURM file was updated.
- `epathst.tcga.worker.run_final_audit` called `44_tcga_audit.py` next to itself; after the move that path would not
  exist, so it now calls `24_tcga_validation.py audit` with the same arguments.
- Six SLURM files called `src/scripts/snapshot_experiment.py`, which round 1 moved to `src/scripts/tools/`; the path is fixed.
- Scripts 20 and 21 read the H-Optimus-0 file names above. The files currently in the results folder are the older
  GenBio-labelled ones, so 20 and 21 will not find their B1 and teacher-parameter inputs until the H-Optimus-0 baseline is rerun
  (and 20 still has round-1 bug 1: it fails on a missing teacher-parameter file).
- Exploratory SLURM files that run `.tmp_*` scratch scripts moved to `slurm/scratch/` (git-ignored, like `.tmp_*`); the A10
  job files moved to `slurm/a10/`.

## 7. Incident during testing

A `--help` sweep over the scripts executed 20 and 21, which take no arguments. Script 20 rewrote
`cptac_hoptimus0_distill_hest_pna_work/results/final/tables/` and `final/folds/` before stopping. Seven of the eight files are
byte-identical to their committed snapshot copies; `efficiency.csv` lost its teacher-parameter and compression values. A copy of
the folder as it is now is in the session scratch area. To restore the committed version:
`cp experimental_runs/2026-08-31_job84871_final_5fold_3cancer_seed0_with_results_final/results/final/tables/efficiency.csv <work_dir>/results/final/tables/`.
The same sweep re-ran `26_hest_audits.py panel-coverage`, which rewrote `results/hest_panel_coverage/` with recomputed values.

## 8. Left unchanged on purpose

- 01-23 stay numbered and in place; their code was only reformatted in round 1 and here.
- Script 19 keeps its own copies of `official_target` and `_decode_barcodes`, which duplicate `official_hest`: the spec forbids
  deduplicating scientific code.
- The A10 subsystem (`src/pathology_st/`, `a10_*` scripts, configs, tests) is untracked and not part of the paper; its imports of
  the old `pathology_st.*` modules are still broken (round-1 note 10).
- `.tmp_*` scratch scripts are git-ignored, not deleted. The figure code that came from them now lives in `figures/`.
- `experimental_runs/` snapshot files keep their original names and contents.

## 9. Found, not fixed (needs a decision)

1. **Distillation teacher of the stored student.** `cptac_teacher_embeddings.h5` is (564,312 x 4608) and the distillation
   checkpoint `S1_seed0_best.pt` has a projection head 320 -> 1024 -> 4608, with validation cosine 0.9060 at epoch 56. H-Optimus-0
   embeddings are 1536-d; 4608-d is the GenBio-PathFM CLS size, and `genbio_teacher_param_count.json` records 1,133,686,784
   parameters loaded from `genbio-ai/genbio-pathfm`. The paper reports an H-Optimus-0 teacher, a 320 -> 1024 -> 1536 head and
   cosine 0.9060 at epoch 55. `locked_encoder_seed0.pt`, derived from this checkpoint, is the encoder behind every saved model,
   the TCGA predictions, the Leiden runs and the stored ePathST predictions used in two figures. Commit 421d30f (2026-09-13)
   says those earlier results "genuinely were produced using GenBio-PathFM"; later commits (3bd426b, 0439715) relabelled
   them as H-Optimus-0. The code and configs now describe an H-Optimus-0 teacher, which is what a new distillation run would
   use; the stored checkpoints were not distilled from it.
2. `docs/reports/HEST_FINAL_5FOLD_RESULTS.md` section 7 and the `experimental_runs` snapshots show official-protocol teacher
   numbers (IDC 0.5898, COAD 0.3264, ccRCC 0.2638) and B1 numbers under an H-Optimus-0 label; they equal the GenBio-labelled
   files on disk (`official_hest_benchmark_summary.csv`, `genbio_summary.csv`).
3. The file named `gene_context.png` in `paper_figures/` is a third-party GigaTIME figure, not the gene-context dot plot the
   caption describes; check what was uploaded to the paper.
4. Round-1 items 1-4, 6 and 7 are still open (item 5, the missing `INDEX.csv` rows, is fixed).

## 10. Final statement

1. Scientific functionality changed? No. Moved code is AST-identical, digests and end-to-end outputs are identical.
2. Model architecture changed? No.
3. Evaluation or statistical methodology changed? No.
4. Efficiency methodology changed? No.
5. All available tests pass? Yes: unit tests 9/9, determinism digests, one-slide end to end, figure regeneration.
6. Deterministic baseline checks preserved? Yes, byte-identical.
