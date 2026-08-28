# PathDistill-ST — High-Level Design

This document describes the end-to-end system: what each stage does, how
data flows between stages, and the key architectural decisions. See the
top-level `README.md` for the paper abstract and a one-line-per-script
summary; this doc goes one level deeper into *why* the pipeline is shaped
the way it is.

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
        │                                             graph construction, PNA
        ▼                                             residual correction, A0-A7
 [5] train ConvNeXt V2-Atto student                   ablations
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
                                                         │       comparison (A8/A9)
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
| HEST metadata audit | `copy_log_{organ}_cancer.csv` + `{organ}/wsis/*.tif` + `{organ}/st/*.h5ad` | per-sample manifest, alignment QC pass/fail |
| HEST patch extraction + features | HEST WSI + ST spot coordinates (`obsm['spatial']`) | 320-D node feature per ST spot (via locked encoder) |
| Graph construction | ST spot coordinates (real microns) | multiscale graph: local (≤150µm, ≤8 neighbors), meso (≤300µm, ≤12), macro (≤600µm, ≤16), RBF-encoded edge distances |
| Patch head | node features | baseline per-spot gene/pathway prediction (no graph) |
| PNA residual correction | node features + graph | `y_final = y_patch + GNN(node_features, graph)`, correction head near-zero-initialized so it starts as a no-op |
| Evaluation | predictions vs. held-out targets | patient-level Pearson r (Fisher-z), bootstrap CIs, FDR-corrected significance |

Everything structural (paths, hyperparameters, radii, ablation toggles) is
config-driven from `configs/base.yaml` (mirrored to `base.json` for nodes
without PyYAML) — nothing is hard-coded in the scripts. See
`sts/utils.load_config` / `resolve_run_paths`.

## 4. Module map (`sts/`)

- **Data / patching**: `cptac_select.py` (WSI QC + selection), `wsi_patches.py`
  (patch extraction; dispatches between DICOM (`wsidicom`) and plain
  openslide-readable formats), `hest_prep.py` (HEST metadata audit,
  alignment QC, ST-spot patch extraction), `patient_split.py` (patient-level
  splitting, no leakage across train/val/test).
- **Models**: `teacher_genbio.py` (frozen GenBio-PathFM wrapper, non-standard
  HF preprocessing/output handling), `student_model.py` (ConvNeXt V2-Atto +
  projection head + lock/strip logic), `distill_loss.py` /
  `distill_train.py` (cosine + feature distillation loop), `patch_head.py`
  (bottleneck MLP baseline), `graph_model.py` (residual PNA correction
  head), `pna_block.py` (PNA graph-conv layer).
- **Graph**: `graph_builder.py` (multiscale radius graph from real spatial
  coordinates), `graph_training.py` (training loop for the graph model),
  `destruction.py` (edge-randomization dose-response controls).
- **Targets / evaluation**: `expression.py` (leakage-free HVG selection —
  computed from the training fold/slide only), `pathways.py` (PROGENy
  pathway activity targets), `metrics.py`, `stats_tests.py` (bootstrap,
  Fisher-z, FDR), `distill_eval.py` (retrieval/probe/UMAP eval for the
  distillation step), `efficiency.py` (params/FLOPs/throughput), `cv_folds.py`
  (patient-level k-fold), `ablations.py` (A0–A9 ablation definitions),
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
includes:

- **S0 / S1 / S2 student variants** (Phase 1): ImageNet-only, GenBio-distilled,
  and random-init-distilled — isolates how much of the student's usefulness
  comes from distillation vs. architecture alone.
- **Patch-only vs. graph-corrected** (Phase 2 main comparison).
- **Parameter-matched non-spatial baseline** — same parameter count as the
  GNN but no spatial structure, to rule out "more parameters" as the
  explanation for any graph benefit.
- **Random graph / degree-preserving shuffled graph** — same graph
  statistics (degree distribution) but destroyed or randomized spatial
  semantics, to isolate *topology* from *graph structure in general*.
- **Progressive spatial-edge destruction (q ∈ {0, 25, 50, 75, 100}%)** — a
  dose-response curve: performance should degrade monotonically toward the
  patch-only floor as real edges are replaced with random ones, if the
  graph benefit is genuinely spatial.
- **Fair GenBio-PathFM comparison (A8/A9)** — runs the *teacher itself* on
  the same HEST split, to contextualize how much performance the ~300x
  compression actually costs.

Statistics: patient-level Pearson r per gene/pathway, Fisher-z aggregation
across patients, 10,000-resample bootstrap CIs, Benjamini-Hochberg FDR
correction (q=0.05) across genes — everything is evaluated at the patient
level (not spot level) to avoid pseudo-replication.

## 6. Known constraints / follow-ups

- HEST metadata schema is confirmed correct for the `colon` (COAD) cohort;
  `breast` (IDC) and `kidney` (CCRCC) directories exist but have no
  `copy_log_*.csv` yet — `load_hest_raw_metadata` skips-and-reports missing
  cancers rather than failing.
- `10_run_patient_cv.py`'s A3 (parameter-matched MLP) ablation currently
  reuses the patch-head architecture as a placeholder; a dedicated wider
  `MolecularBottleneckHead` needs to be wired in via
  `sts.ablations.matched_mlp_hidden_dim` before the full run.
- Large artifacts (patch caches, teacher embeddings, checkpoints) are kept
  out of `/home` and `/scratch` (20GB per-user quota) and instead live under
  the `YurovskyGroup-nvwulf` project space — see `docs/RUN_STATUS.md` for
  current paths and job status.
- See `docs/RUN_STATUS.md` for what has actually been run against real data
  so far vs. what is still pending, and `experimental_runs/` for dated
  result snapshots once runs complete.
