# V1 -> V2 migration (Final_Project_V1.txt -> Final_Project_V2.txt)

Source specs: `/lustre/nvwulf/home/schennareddy/cellvit_project/
Final_Project_V1.txt` (original, the codebase's foundation) and
`Final_Project_V2.txt` ("PathDistill-ST — Revised Implementation Plan",
written after seeing PoC_5's real results -- see `docs/POC5_RESULTS.md`).

**Explicit constraint honored throughout this migration**: nothing related
to CPTAC WSI selection/QC, patch extraction, GenBio-PathFM teacher
inference, or ConvNeXt V2-Atto distillation training was touched --
`src/pathology_st/teacher_genbio.py`, `src/pathology_st/student_model.py`, `src/pathology_st/distill_train.py`,
`src/pathology_st/distill_loss.py`, `src/scripts/01`-`07`, and `configs/base.yaml`'s
`teacher:`/`student:`/`distillation:` sections are byte-for-byte unchanged
except where an earlier, unrelated fix (the HDF5 truncation bug) already
touched `src/pathology_st/teacher_genbio.py` before this migration started -- see
`docs/RUN_STATUS.md`. Verified by diffing those sections' resolved config
values before/after (`distillation.max_epochs=100`,
`teacher.hf_repo=genbio-ai/genbio-pathfm`, `student.backbone=
convnextv2_atto` all confirmed unchanged) and by confirming the live
training job (84150 at migration time) kept running throughout with zero
interruption or config drift.

## What changed, and where

### 1. Edge-feature dimensionality bug fix (V2 Phase 9)
V1's single edge-feature function (16 RBF + 3-D scale one-hot = **19-D**)
omitted the explicit normalized-distance scalar V2's spec calls for.
Fixed in `src/pathology_st/graph_builder.py`:
- `union_edge_features` (new name; `rbf_edge_features` kept as a
  back-compat alias) -- now correctly **20-D** (1 distance + 16 RBF + 3
  one-hot), used by A4/A5/A6/A7/A9 (any backend where one message-passing
  layer sees edges from possibly-mixed scales).
- `per_scale_edge_features` (new) -- **17-D** (1 distance + 16 RBF, no
  one-hot), used by A8's per-branch PNAs, where branch identity already
  encodes scale.
- `split_edges_by_scale` (new) -- splits one multiscale edge set into
  local/meso/macro subsets.
Confirmed via `tests/test_graph_and_stats.py` (updated: old test asserted
the buggy 19-D shape, now asserts 20-D/17-D) and a live forward-pass smoke
test (see below).

### 2. New graph-model backends (V2 Phases 12, 15)
`src/pathology_st/graph_model.py` (previously one backend) now has three, all sharing
the same residual-correction wrapper (`ResidualGraphPredictor`, backend-
agnostic via `*args` passthrough):
- `GraphCorrectionModel` -- union multiscale PNA (A4/A5/A6/A7), unchanged core logic.
- `ScaleAwareGraphCorrectionModel` (**new, A8, the proposed final model**)
  -- three independent per-scale PNA branches + a learned softmax gate
  over `[h^0; h^L; h^M; h^G]`. Reuses the SAME single-`Data`-object PyG
  pipeline as every other backend via a per-edge `edge_scale` tag
  (`src/pathology_st/ablations.scale_aware_edge_data`) rather than a parallel 3-edge-set
  data path -- `src/pathology_st/graph_training.py`'s make_pyg_data/train_graph_model/
  predict_graph_model needed only a `backend` parameter and an optional
  `edge_scale` field, no structural rewrite.
- `GATv2CorrectionModel` (**new, A9, architecture control**) --
  `src/pathology_st/gatv2_block.py`, identical structure to the PNA block with GATv2 as
  the message-passing operator, isolating PNA-vs-GATv2 as V2 Phase 15
  requires ("everything except the message-passing layer should remain
  identical").
- `assert_zero_residual_at_init` (**new**) -- the explicit unit test V2
  Phase 11 requires, directly targeting PoC_5's observed failure (patch
  CNN 0.1875 > multiscale PNA 0.1268): asserts `y_final == y_patch` at
  init, for whichever backend is active. Called automatically at the start
  of every `train_graph_model` call, not just in sanity checks.

**Real bug this surfaced and fixed**: `src/pathology_st/patch_head.py`'s V2-mandated
bottleneck (LayerNorm+GELU+**Dropout**) made the "frozen" patch head
non-deterministic across forward calls in train mode, since nothing had
put it in `eval()`. `ResidualGraphPredictor.__init__` now calls
`self.patch_head.eval()` and overrides `.train()` to keep it there
permanently -- caught by the new S1 sanity check failing on first run,
fixed, then reconfirmed passing (see `docs/RUN_STATUS.md`-style
incident record, kept here since it's part of this migration's own work).

### 3. Ablation grid restructured (V2 Phase 14)
`src/pathology_st/ablations.py`'s `ABLATION_GRID` is now ordered **primary block first**
(A1, A2, A5, A6, A7, A8) **then secondary** (A0, A3, A4, A9) per V2 Phase
14/18's explicit run-priority. A0 (mean-expression) and A3 (parameter-
matched MLP) are now REAL implementations in `src/scripts/10_run_patient_cv.py`
(V1's driver silently `continue`'d past them). V1's old A8/A9 slots
(GenBio-teacher fair comparison) are REMOVED from this grid entirely --
V2 Phase 21 replaces that comparison with a methodologically stricter,
separate PCA-256-based analysis (see #6 below), run only after this grid
is locked.

### 4. Patch-head loss changed to gene-only by default (V2 Phase 7)
`src/pathology_st/patch_head.py` / `configs/base.yaml`'s `patch_head.pathway_loss_weight`
default changed **0.2 -> 0.0** (PoC_5 showed no consistent benefit from the
mandatory pathway auxiliary term). `pathway_loss_sweep: [0.0, 0.05, 0.10,
0.20]` added to config for the post-lock ablation V2 Phase 19 calls for
(sweep on validation patients only, after architecture is frozen -- not
yet wired into a driver script, see Known follow-ups).

### 5. Fold-completeness enforcement (V2 Phase 3) -- the PoC_5 fix
The most consequential change. PoC_5's own cross-fold aggregate silently
pooled only 3 of 4 completed folds (fold 3 finished but was never
included; fold 4 failed both/all 3 seeds on a broken script path -- see
`docs/POC5_RESULTS.md` for the full forensic writeup). V2 Phase 3 requires
a hard assertion against exactly this. Added:
- `src/pathology_st/cv_folds.expected_run_keys` -- enumerates every (cancer, fold,
  seed, ablation) the CV design calls for.
- `src/pathology_st/cv_folds.assert_run_completeness` -- prints every included run,
  then **raises** (not warns) on any mismatch between expected and
  completed sets.
- Wired into `src/scripts/10_run_patient_cv.py` (end-of-run gate) and
  `src/scripts/12_primary_hypothesis_test.py` (pre-aggregation gate, so no
  statistic is ever computed over a run set that doesn't match the design).
- `src/scripts/16_sanity_checks.py`'s new **S9** proves the guard itself
  correctly passes on a complete run set and raises on an incomplete one,
  before relying on it for anything real.

### 6. Per-cancer as primary analysis unit (V2 Phase 1)
`src/pathology_st/cv_folds.build_cv_grid` now returns `primary_analysis_unit:
"per_cancer"`; `configs/base.yaml`'s `cross_validation.primary_analysis_unit`
mirrors this. `src/scripts/12_primary_hypothesis_test.py` reports every gate
BOTH pooled and per-cancer, with per-cancer as the stated primary result.

### 7. Three primary statistical gates replace the single hypothesis test (V2 Phase 22)
`src/scripts/12_primary_hypothesis_test.py` rewritten around
`configs/base.yaml`'s new `primary_gates` block:

| Gate | Comparison |
|---|---|
| 1 -- Distillation | R(A2 distilled patch) > R(A1 ImageNet patch) |
| 2 -- Spatial context | R(A8 scale-aware PNA) > R(A2 distilled patch) |
| 3 -- True topology (decisive) | R(A8 scale-aware PNA) > R(A5 shuffled graph) |

Gate 3 replaces V1's Section 25 test (which used A7, union multiscale) --
V2 explicitly redefines the primary spatial-topology comparison to use A8
now that it's the proposed final architecture.

### 8. GenBio fair comparison redesigned around training-fold PCA (V2 Phase 21)
New `src/pathology_st/genbio_pca.py`: `TrainOnlyPCA` (256-D, fit on train patients
only, leakage-asserted like `pathology_st.expression.TrainOnlyStandardizer`) +
Ridge/report helpers, replacing V1's ambiguous "feed the raw 4608-D GenBio
embedding into the same head" approach. `configs/base.yaml`'s new
`genbio_fair_comparison` block (`pca_n_components: 256, ridge_alpha: 1.0`).
**Library-level only** -- the driver script rewrite (replacing
`src/scripts/15_genbio_fair_comparison.py`'s raw-4608-D approach with the
three-head x two-encoder PCA comparison V2 Phase 21 specifies) is a
follow-up, not yet done (see below).

### 9. Post-hoc pathway scoring (V2 Phase 19)
New `src/pathology_st/pathway_posthoc.py`: computes PROGENy pathway activity from a
model's PREDICTED genes (reusing `pathology_st.pathways.compute_progeny_scores`)
rather than only from a dedicated pathway head -- V2's stated reasoning:
"tests biological consistency rather than giving the model the answer
through an auxiliary objective." Library-level only; not yet wired into
an evaluation driver script.

## Verification performed

- `python3 -m py_compile src/pathology_st/*.py src/scripts/*.py` -- clean.
- `tests/test_graph_and_stats.py` -- all 9 pass (3 new/updated for the
  edge-feature dimensionality fix).
- `src/scripts/16_sanity_checks.py` -- all 9 (S1-S9) pass, including the two
  brand-new checks (S3 zero-residual, S9 completeness-guard-works) and the
  real S1 bug this migration introduced-then-fixed (dropout
  non-determinism in the frozen patch head).
- Live end-to-end smoke test of all three graph-model backends
  (pna_union/gatv2/scale_aware) through the FULL `train_graph_model` ->
  checkpoint save/load -> `predict_graph_model` path on synthetic data
  (not just isolated model construction) -- all three train, save, reload,
  and predict correctly with the right output shapes.
- The live CPTAC distillation training job (84150) was checked before and
  after every edit in this migration and never interrupted or altered.

## Known follow-ups (not done in this pass -- explicitly out of scope for now)

- `src/scripts/15_genbio_fair_comparison.py` still uses V1's raw-4608-D
  approach; needs a rewrite using `src/pathology_st/genbio_pca.py`'s `TrainOnlyPCA` to
  match V2 Phase 21's three-head (Ridge/MLP/PNA) x two-encoder
  (GenBio-PCA256 vs. student-256) comparison.
- `src/pathology_st/pathway_posthoc.py` and the `pathway_loss_sweep` config value exist
  but aren't yet driven by a script -- V2 Phase 19's post-lock ablation
  (run only after the architecture matrix above is frozen) is future work.
- V2 Phase 20's spatial-destruction dose-response "ideally preserve
  local/meso/macro edge counts per scale" refinement (vs. the current
  aggregate-degree-only preservation in `src/pathology_st/destruction.
  partial_edge_replacement`) not yet implemented.
- V2 Phase 16's small COAD-fold0-seed0 engineering smoke run (a real-data,
  not synthetic-data, pre-flight check) has no dedicated script yet --
  `src/scripts/16_sanity_checks.py` covers the synthetic-data S1-S9 checks V2
  Phase 17 requires, but not the separate real-data engineering run V2
  Phase 16 also asks for.
- `src/pathology_st/cv_folds.build_cv_grid`'s "make the three cancers separate primary
  experiments" (V2's `experiment_idc/`, `experiment_coad/`,
  `experiment_ccrcc/` directory suggestion) is reflected in
  `primary_analysis_unit` and per-cancer reporting, but the pipeline
  doesn't yet physically separate outputs into per-cancer directories --
  current outputs stay in the shared `results/`/`predictions/` with a
  `cancer` column, achieving the same analysis separation without the
  directory split V2 suggested as one option.
- `src/scripts/12_primary_hypothesis_test.py`'s bootstrap resampling unit is
  (cancer, fold, seed) cells, not individual patients -- flagged explicitly
  in the script's own printed output; a stricter fix requires plumbing
  per-patient R_p values through from `src/scripts/10_run_patient_cv.py`
  rather than the current per-fold aggregate metrics.
