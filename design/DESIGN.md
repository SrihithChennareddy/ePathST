# ePathST — High-Level Design

This document describes the end-to-end system: what each stage does, how
data flows between stages, and the key architectural decisions. See the
top-level `README.md` for the paper abstract and a one-line-per-script
summary; this doc goes one level deeper into *why* the pipeline is shaped
the way it is.

This reflects the **V2** architecture (`Final_Project_V2.txt`, the current
spec) — see `docs/V1_TO_V2_MIGRATION.md` for the full change log against
the original V1 build (ablation grid restructuring, the new Scale-Aware
PNA and GATv2 backends, fold-completeness enforcement, etc.). Nothing
about Phase 1 (CPTAC distillation) changed in that migration.

## 1. Problem and approach

Spatial transcriptomics (ST) measures gene expression at spatial locations
in tissue but is expensive and low-throughput. The goal is to predict ST
readouts directly from routine H&E histology images, using two ideas
combined:

1. **Knowledge distillation** — compress a large pathology foundation
   model (GenBio-PathFM, ~1.1B params) into a tiny ConvNeXt V2-Atto student
   (~3.7M params, ~300x smaller) that still produces useful morphology
   embeddings, trained on CPTAC whole-slide images (WSIs).
2. **Explicit spatial graph reasoning** — on the downstream ST task
   (HEST cohorts), represent each ST spot as a node (student-embedding
   feature vector) and connect spatially nearby spots via a physically
   defined multiscale graph (local/meso/macro radii in real microns), then
   run a lightweight PNA graph neural network as a *residual correction*
   on top of a patch-only baseline prediction.

The central scientific question the pipeline is built to answer: **does
explicit tissue topology (the graph) add predictive power for spatial gene
expression beyond what patch-level morphology alone already captures?**
This is tested by comparing the graph-corrected model against patch-only
baselines, parameter-matched non-spatial baselines, random graphs,
degree-preserving shuffled graphs, and a progressive spatial-edge
destruction dose-response (as edges are randomized, does performance decay
toward the patch-only floor?).

## 2. Two-phase architecture

```
PHASE 1: CPTAC DISTILLATION                    PHASE 2: HEST DOWNSTREAM
──────────────────────────────                 ──────────────────────────────
CPTAC WSIs (DICOM, BRCA/COAD/ccRCC)             HEST WSIs + paired ST (.h5ad)
        │                                               │
        ▼                                               ▼
 [1] select 1 WSI/patient (QC)               [8] audit metadata, alignment QC
        │                                               │
        ▼                                               ▼
 [2] tissue-masked patch extraction           [9] extract ST-spot-centered
        │                                          patches, run LOCKED student
        ▼                                          encoder -> 320-D node features
 [3] patient-level 80/10/10 split                        │
        │                                                ▼
        ▼                                    [10] per-fold: fold-safe HVG
 [4] frozen GenBio-PathFM teacher                  target selection, patch-head
     (1.1B params) embeds every patch                baseline, multiscale physical
        │                                             graph construction, residual
        ▼                                             PNA/GATv2 correction, full
 [5] train ConvNeXt V2-Atto student                   A0-A9 ablation grid
     (S0 imagenet / S1 genbio-distilled /                    │
      S2 random-init-distilled)                              ▼
        │                                    [11] spatial destruction dose-
        ▼                                          response (q=0,25,50,75,100%
 [6] evaluate on held-out test patients             edges randomized)
     (cosine sim, retrieval, linear probe)                   │
        │                                                    ▼
        ▼                                    [12] primary + secondary
 [7] lock + strip student to encoder-only            hypothesis tests (bootstrap,
     (projection head discarded)                      Fisher-z, FDR)
        │                                                     │
        └──────────────► [locked encoder] ─────────────┐      ▼
                                                         │  [13] additional analyses
                                                         │      (multiscale contrib,
                                                         │       Moran's I, pathways)
                                                         │             │
                                                         │             ▼
                                                         │  [14] efficiency (params,
                                                         │       FLOPs, throughput)
                                                         │             │
                                                         │             ▼
                                                         │  [15] fair GenBio-PathFM
                                                         │       comparison
                                                         │       (PCA-256 encoder,
                                                         │        separate from A8/A9)
                                                         │             │
                                                         └──────► [16] sanity checks
                                                                  (must pass first)
                                                                        │
                                                                        ▼
                                                          [17] results tables + figures
                                                                        │
                                                                        ▼
                                                          [18] final interpretation
```

Phase 1 produces exactly one durable artifact that Phase 2 depends on: the
**locked student encoder** (backbone-only checkpoint, projection head
removed). Everything else in Phase 1 (teacher embedding cache, training
checkpoints, distillation eval reports) is intermediate. Phase 2 never
touches CPTAC data or the teacher again — it only loads the locked encoder
as a frozen feature extractor.

## 3. Data flow and formats

| stage | input | output |
|---|---|---|
| CPTAC WSI selection | `CPTAC_METADATA/{cancer}_selected_WSI_per_patient.txt` + `CPTAC_ROOT/{cancer}/{patient}/*.dcm` | one QC'd WSI path per patient, quality score `Q` |
| CPTAC patch extraction | selected WSI (DICOM pyramid or plain TIFF/SVS) | tissue-masked 224×224 px patches @ ~0.5 µm/px, manifest CSV |
| Teacher embedding | patches | `[N, 4608]` GenBio-PathFM embeddings, HDF5 cache (run-once) |
| Student distillation | teacher embeddings + patches | student checkpoint (backbone 320-D + training-only projection head to 4608-D) |
| Lock/strip | best student checkpoint | encoder-only state dict, 3,387,400 params |
| HEST metadata audit | per-cancer `{organ}_metadata.csv` + `{wsi_dir}/*.tif` + `{st_dir}/*.h5ad` (`paths.hest_by_cancer`, all three cancers) | per-sample manifest, alignment QC pass/fail |
| HEST patch extraction + features | HEST WSI + ST spot coordinates (`obsm['spatial']`) | 320-D node feature per ST spot (via locked encoder) |
| Graph construction | ST spot coordinates (real microns) | multiscale graph: local (≤150µm, ≤8 neighbors), meso (≤300µm, ≤12), macro (≤600µm, ≤16), RBF-encoded edge distances |
| Patch head | node features | baseline per-spot gene/pathway prediction (no graph) |
| Residual graph correction | node features + graph | `y_final = y_patch + GNN(node_features, graph)`, correction head near-zero-initialized so it starts as a no-op; 3 interchangeable backends — union multiscale PNA, scale-aware per-branch PNA (A8, proposed final model), GATv2 (A9, architecture control) |
| Evaluation | predictions vs. held-out targets | patient-level Pearson r (Fisher-z), bootstrap CIs, FDR-corrected significance |

Everything structural (paths, hyperparameters, radii, ablation toggles) is
config-driven from `configs/base.yaml` (mirrored to `base.json` for nodes
without PyYAML) — nothing is hard-coded in the scripts. See
`src/pathology_st/utils.load_config` / `resolve_run_paths`.

## 4. Module map (`src/pathology_st/`)

- **Data / patching**: `cptac_select.py` (WSI QC + selection), `wsi_patches.py`
  (patch extraction; dispatches between DICOM (`wsidicom`) and plain
  openslide-readable formats), `hest_prep.py` (HEST metadata audit,
  alignment QC, ST-spot patch extraction), `patient_split.py` (patient-level
  splitting, no leakage across train/val/test).
- **Models**: `teacher_genbio.py` (frozen GenBio-PathFM wrapper, non-standard
  HF preprocessing/output handling), `student_model.py` (ConvNeXt V2-Atto +
  projection head + lock/strip logic), `distill_loss.py` /
  `distill_train.py` (cosine + feature distillation loop), `patch_head.py`
  (bottleneck MLP baseline, LayerNorm+GELU+Dropout trunk — kept in `.eval()`
  permanently once wrapped by `ResidualGraphPredictor`), `graph_model.py`
  (`ResidualGraphPredictor` residual-correction wrapper, backend-agnostic,
  over three interchangeable backends: `GraphCorrectionModel` (union
  multiscale PNA), `ScaleAwareGraphCorrectionModel` (A8, three per-scale PNA
  branches + learned softmax gate — the proposed final model),
  `GATv2CorrectionModel` (A9, architecture control, identical structure with
  GATv2 as the message-passing op); also `assert_zero_residual_at_init`),
  `pna_block.py` (PNA graph-conv layer), `gatv2_block.py` (GATv2 graph-conv
  layer, mirrors `pna_block.py`), `genbio_pca.py` (`TrainOnlyPCA` + Ridge
  baseline for the fair GenBio-PathFM comparison), `pathway_posthoc.py`
  (post-hoc PROGENy scoring from a model's predicted genes, rather than a
  dedicated pathway head).
- **Graph**: `graph_builder.py` (multiscale radius graph from real spatial
  coordinates; `union_edge_features` 20-D for mixed-scale backends,
  `per_scale_edge_features` 17-D for per-branch backends,
  `split_edges_by_scale`), `graph_training.py` (backend-parameterized
  training loop, shared by all three `graph_model.py` backends via a
  `backend` argument and an optional per-edge `edge_scale` tag),
  `destruction.py` (edge-randomization dose-response controls).
- **Targets / evaluation**: `expression.py` (leakage-free HVG selection —
  computed from the training fold/slide only), `pathways.py` (PROGENy
  pathway activity targets), `metrics.py`, `stats_tests.py` (bootstrap,
  Fisher-z, FDR), `distill_eval.py` (retrieval/probe/UMAP eval for the
  distillation step), `efficiency.py` (params/FLOPs/throughput), `cv_folds.py`
  (patient-level k-fold, `expected_run_keys`/`assert_run_completeness` fold-
  completeness gate), `ablations.py` (A0-A9 ablation definitions, primary
  block A1/A2/A5/A6/A7/A8 then secondary A0/A3/A4/A9),
  `analysis.py` / `report.py` / `figures.py` (aggregation and plotting).
- **`utils.py`**: config loading, path resolution, shared helpers.

`graph_builder.py`, `pna_block.py`, `destruction.py`, `stats_tests.py`,
`expression.py`, `pathways.py`, and `metrics.py` are adapted from the
already-validated `STS_PoC/PoC_5` package, since the graph
radii/caps/RBF/PNA hyperparameters are numerically identical. The
architectures diverge at the GNN head: PoC_5 uses a gated PNA+APPNP fusion,
whereas this project uses a **residual correction** design
(`y_final = y_patch + GNN(...)`, near-zero-initialized) so the graph model
starts equivalent to the patch-only baseline and only earns influence
during training.

## 5. Experimental design (why the ablation set exists)

The core hypothesis test needs to rule out confounds, so the ablation grid
(`src/pathology_st/ablations.ABLATION_GRID`, A0-A9) includes:

- **S0 / S1 / S2 student variants** (Phase 1): ImageNet-only, GenBio-distilled,
  and random-init-distilled — isolates how much of the student's usefulness
  comes from distillation vs. architecture alone.
- **A1 ImageNet patch / A2 distilled patch** — patch-only baselines,
  isolating how much of any downstream benefit is attributable to
  distillation itself (primary Gate 1, see below).
- **A5 shuffled graph / A6 local-only PNA / A7 multiscale union PNA / A8
  scale-aware multiscale PNA** — the graph-corrected models, run against
  the same patch features. A8 (three independent per-scale PNA branches +
  a learned softmax gate) is the **proposed final architecture**; A6/A7 are
  intermediate structural ablations (single-scale vs. union-multiscale
  message passing).
- **A0 mean-expression / A3 parameter-matched MLP** (secondary) — A0 is a
  trivial floor; A3 has the *same parameter count* as the GNN but no
  spatial structure, ruling out "more parameters" as the explanation for
  any graph benefit (a real `MolecularBottleneckHead`, not a placeholder).
- **A4 random graph / A5 degree-preserving shuffled graph** — same graph
  statistics (degree distribution) but destroyed or randomized spatial
  semantics, to isolate *topology* from *graph structure in general*.
- **A9 GATv2 multiscale** (secondary) — identical residual-correction
  wrapper and multiscale graph as A7/A8, but GATv2 instead of PNA as the
  message-passing operator — an architecture control isolating "PNA
  specifically" from "graph message-passing in general."
- **Progressive spatial-edge destruction (q ∈ {0, 25, 50, 75, 100}%)** — a
  dose-response curve: performance should degrade monotonically toward the
  patch-only floor as real edges are replaced with random ones, if the
  graph benefit is genuinely spatial.
- **Fair GenBio-PathFM comparison** (`src/pathology_st/genbio_pca.py`, separate from the
  A-grid above) — a training-fold-only 256-D PCA of the *teacher's* own
  4608-D embeddings + a Ridge baseline, to contextualize how much
  performance the ~300x compression actually costs, run only after the
  A-grid above is locked. Library-level as of this writing; the driver
  script (`src/scripts/15_genbio_fair_comparison.py`) rewrite to use it is a
  follow-up — see `docs/V1_TO_V2_MIGRATION.md`.

Three primary statistical gates (`configs/base.yaml`'s `primary_gates`,
`src/scripts/12_primary_hypothesis_test.py`) formalize the central question,
reported both pooled and per-cancer (per-cancer is primary):

| Gate | Comparison |
|---|---|
| 1 — Distillation | R(A2 distilled patch) > R(A1 ImageNet patch) |
| 2 — Spatial context | R(A8 scale-aware PNA) > R(A2 distilled patch) |
| 3 — True topology (decisive) | R(A8 scale-aware PNA) > R(A5 shuffled graph) |

Statistics: patient-level Pearson r per gene/pathway, Fisher-z aggregation
across patients, 10,000-resample bootstrap CIs, Benjamini-Hochberg FDR
correction (q=0.05) across genes — everything is evaluated at the patient
level (not spot level) to avoid pseudo-replication. Before any statistic is
aggregated, `src/pathology_st/cv_folds.assert_run_completeness` hard-fails on a
mismatch between the expected and completed (cancer, fold, seed, ablation)
run set — added after PoC_5's own aggregate silently pooled an incomplete
fold set (see `docs/POC5_RESULTS.md`).

## 6. Official HEST-1k Benchmark reproduction (scripts 19-23)

Separate from the custom protocol above (own patient-level GroupKFold,
HVG1000/HVG50, PROGENy14), scripts 19-23 answer "how does this project's
model compare on the standard published HEST-1k leaderboard" (official
splits, official curated 50-gene panel per cancer, official pooled-spot
Pearson metric) — see `docs/HEST_OFFICIAL_ABLATION_RESULTS.md` and
`docs/HEST_OFFICIAL_HVG50_GENES.md` for the full protocol/gene-list
writeup and results.

- `19_official_hest_benchmark.py` — official protocol, raw teacher/student
  encoders only (train-fold PCA256 + Ridge, not the trained A0-A9 heads).
- `22_official_hest_ablation_grid.py` — reuses `10_run_patient_cv.py`'s
  `run_ablation`/`train_patch_head` (via `importlib`) to run the *same*
  trained A0-A9 grid against official data/splits/genes instead of
  reimplementing it — the direct A0-A9-vs-leaderboard comparison.
- `23_morans_i_diagnostic.py` — Gate 3 (`docs/HEST_FINAL_5FOLD_RESULTS.md`,
  `docs/HEST_OFFICIAL_ABLATION_RESULTS.md`) already shows A8 statistically
  indistinguishable from A5 (shuffled-graph control) on both protocols —
  an indirect null result. This script tests the same question directly:
  per gene, does (A8 R − A2 R) correlate with that gene's ground-truth
  Moran's I (real spatial autocorrelation)? Replicates the diagnostic
  PoC_5 already ran once (`Corr(delta, MoranI) = 0.1148, p = 0.630` — no
  relationship) on the current official-protocol data.
- `20_build_final_results.py` / `21_results_summary_doc.py` — assemble
  the final cross-script results tables and the git-tracked summary doc.

## 7. Known constraints / follow-ups

- HEST metadata + WSI/ST data is now confirmed present for all three
  cancers as of 2026-08-29 (`configs/base.yaml`'s `paths.hest_by_cancer`;
  202 real samples post metadata-audit — IDC 120, COAD 58, CCRCC 24; see
  `docs/INPUT_DATA.md` §2 for exact locations/provenance/scale). Breast's
  files live at `simons_2026/breast/` (used in place); colon/kidney were
  downloaded from HuggingFace `MahmoodLab/hest` into scratch and supersede
  the earlier 23-slide `simons_2026/colon` CRC23 subset used for PoC_5/early
  validation. `load_hest_raw_metadata` still skips-and-reports any cancer
  with no configured/found source, for whenever a future cohort is added.
  Full alignment QC + patch extraction across all 202 samples at scale has
  not yet been run (only exercised on a 1-2 sample subset so far).
- `10_run_patient_cv.py`'s A0 (mean-expression) and A3 (parameter-matched
  `MolecularBottleneckHead`) ablations are real implementations (fixed
  during the V1->V2 migration — V1's driver silently skipped both).
- Large artifacts (patch caches, teacher embeddings, checkpoints) are kept
  out of `/home` and `/scratch` (20GB per-user quota) and instead live under
  the `YurovskyGroup-nvwulf` project space — see `docs/RUN_STATUS.md` for
  current paths and job status.
- V2 migration follow-ups not yet done: `src/scripts/15_genbio_fair_comparison.py`
  still needs its `src/pathology_st/genbio_pca.py`-based rewrite; `src/pathology_st/pathway_posthoc.py`
  and the `pathway_loss_sweep` config aren't yet driven by a script; the
  spatial-destruction dose-response doesn't yet preserve per-scale
  (local/meso/macro) edge counts, only aggregate degree; the primary
  hypothesis test's bootstrap unit is (cancer, fold, seed), not individual
  patients — full list in `docs/V1_TO_V2_MIGRATION.md`.
- See `docs/RUN_STATUS.md` for what has actually been run against real data
  so far vs. what is still pending, and `experimental_runs/` for dated
  result snapshots once runs complete.
