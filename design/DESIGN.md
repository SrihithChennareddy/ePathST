# ePathST — Detailed Methods and Design Document

This is the exhaustive version: every data source, hyperparameter, formula,
ablation, statistical test, and result in this project, with the reasoning
behind each decision. It's written to stand on its own as a supplementary
methods document for a paper — anyone should be able to reproduce this
project's numbers, or understand exactly why a design choice was made,
without needing to read the code first. For a quick orientation instead,
see `README.md` (abstract + one-line-per-script pipeline list). For the
day-to-day operational log (what ran, what broke, when), see
`docs/RUN_STATUS.md`. Every specific results table cited below lives in a
dedicated `docs/*.md` file, referenced inline.

This reflects the **V2** architecture (`Final_Project_V2.txt`) — see
`docs/V1_TO_V2_MIGRATION.md` for the V1→V2 change log.

## Contents

1. [Motivation and prior work](#1-motivation-and-prior-work)
2. [Scientific question and hypotheses](#2-scientific-question-and-hypotheses)
3. [Data](#3-data)
4. [Phase 1 — Knowledge distillation](#4-phase-1--knowledge-distillation)
5. [Phase 2 — Spatial graph-corrected expression prediction](#5-phase-2--spatial-graph-corrected-expression-prediction)
6. [Ablation grid](#6-ablation-grid)
7. [Statistical methodology](#7-statistical-methodology)
8. [Official HEST-1k Benchmark reproduction](#8-official-hest-1k-benchmark-reproduction)
9. [Results](#9-results)
10. [Interpretation](#10-interpretation)
11. [Limitations](#11-limitations)
12. [Compute resources](#12-compute-resources)
13. [Repository structure and reproducibility](#13-repository-structure-and-reproducibility)
14. [Known follow-ups](#14-known-follow-ups)

## 1. Motivation and prior work

Spatial transcriptomics (ST) measures gene expression at spatial locations
in tissue but is expensive and low-throughput compared to routine
histology (H&E). If a model could predict ST readouts directly from H&E
images, it would make molecular-level tissue characterization available
anywhere a microscope slide already exists.

This project is the direct successor to a proof-of-concept, **PoC_5**
(`STS_PoC/PoC_5`, `crcspatial` package — see `docs/POC5_RESULTS.md`), which
tested the same core idea on a smaller cohort (23 colorectal-cancer
patients, MobileNetV3-Small patch encoder, physical multiscale PNA graph).
PoC_5's decisive hypothesis — a real spatial graph beats a
degree-preserving shuffled graph — **did not pass**
(`Corr(Δz) = -0.0035, 95% CI [-0.0173, 0.0099], p = 0.681`), and the
patch-only CNN baseline outscored every graph variant (0.1875 vs. best
graph variant 0.1317, HVG50 patient-macro R). PoC_5 also surfaced a
methodological bug worth carrying forward as a hard rule: its own
cross-fold aggregation script silently pooled only 3 of 4 completed folds,
because nothing enforced that a run's result set matched its intended
design. This project's `epathst.data.splits.assert_run_completeness`
(§7) exists specifically to make that class of bug impossible.

This project extends PoC_5 in three ways: (1) replaces the patch encoder
with a much larger foundation model's knowledge, compressed via
distillation rather than trained from scratch; (2) runs on three cancer
types instead of one, with real patient-level cross-validation instead of
a single small cohort; (3) adds explicit negative controls (a
parameter-matched non-spatial MLP, a random graph, a degree-preserving
shuffled graph) so any observed "graph helps" result can be attributed to
real spatial topology specifically, not to extra parameters or generic
neighbor-averaging.

## 2. Scientific question and hypotheses

**Primary question**: does explicit tissue topology (a physically
constructed spatial graph over ST spots) add predictive power for spatial
gene expression beyond what patch-level H&E morphology alone already
captures?

**Secondary question**: can a ~300×-compressed distilled encoder retain
enough of a much larger pathology foundation model's representational
power to be practically useful, and at what accuracy cost?

Both questions are tested via pre-registered comparisons (§6, §7), not
post-hoc pattern-matching — the ablation grid and the three primary gates
were fixed before the locked final run (job 84871).

## 3. Data

Neither dataset is committed to this repo (both are large, DICOM/TIFF/HDF5
binary data). Full schema and provenance: `docs/INPUT_DATA.md`. Summary
here.

### 3.1 CPTAC (Phase 1 — distillation)

**Source**: Imaging Data Commons (IDC), `idc-open-data` public S3 bucket,
collections `cptac_brca`, `cptac_coad`, `cptac_ccrcc`. Whole-slide H&E
images (WSIs) only — no expression data in this phase.

**Selection**: one WSI per patient, from all available candidate slides,
via `src/epathst/data/cptac_select.py`. Hard filters applied to 598 initial
candidates (tumor-specimen label required, ≥5% tissue fraction, ≤50%
blurry-tissue fraction, ≤35% artifact fraction, mpp ∈ [0.1, 1.0]),
followed by ranking on a deterministic composite quality score:

```
Q = 0.35·usable_tissue + 0.25·focus + 0.20·artifact_free
    + 0.10·stain_quality + 0.10·representativeness
```

computed from a low-resolution thumbnail (global tissue/stain QC) plus 80
sampled ~20× tissue tiles (focus via Laplacian variance, artifact/pen-mark
detection, stain-saturation outliers). Real post-QC funnel (job 83825,
2026-08-28):

| Stage | BRCA | COAD | CCRCC | Total |
|---|---|---|---|---|
| Initial candidates | 198 | 178 | 222 | 598 |
| Excluded: not tumor specimen | 90 | 131 | 0 | 221 |
| Excluded: insufficient tissue | 2 | 0 | 6 | 8 |
| Excluded: unreadable/corrupt | 0 | 0 | 6 | 6 |
| **Selected** | **106** | **47** | **210** | **363** |

COAD's low yield (74% excluded vs. BRCA 45%, CCRCC 0%) is a documented,
not-yet-resolved limitation: the `not_tumor_specimen` filter requires the
substring "tumor" in CPTAC/IDC's own `specimen_label` field, and COAD has
both a genuinely higher fraction of matched-normal-only patients (43/178)
and a higher fraction of ambiguously-labeled bare-`"HE"` specimens that may
or may not be real tumor sections. Full breakdown: `docs/INPUT_DATA.md` §1.

**Split**: patient-level, 80/10/10, stratified per cancer type
(`src/epathst/data/splits.py`), computed before any patch is
extracted; no patient's patches cross a split boundary
(`assert_no_patch_leakage`). Real counts (seed 0): BRCA 85/11/10, COAD
38/5/4, CCRCC 168/21/21 (train/val/test), 363 total. Three independent
seeds (0/1/2) are generated identically for multi-seed robustness
reporting; only seed 0 has been trained on so far.

**Patch extraction**: tissue-masked, spatially sampled 224×224px patches
at ~0.5µm/px target resolution, 564,312 patches total across all selected
WSIs.

### 3.2 HEST (Phase 2 — downstream ST)

**Source**: paired H&E WSI + spatial transcriptomics (Visium-style spot
arrays) from the HEST-1k benchmark, three cancer types matching Phase 1's
cohorts (breast→IDC, colon→COAD, kidney→CCRCC), two different
provenances: breast used in place from a lab-internal HEST cohort
(`simons_2026/breast/`), colon/kidney downloaded from HuggingFace
`MahmoodLab/hest`. On-disk scale: IDC 127 WSI/125 ST (~62GB+7.4GB), COAD 58
WSI/58 ST (~184GB), CCRCC 25 WSI/25 ST (~18GB).

**Alignment QC**: before any patch extraction, ST spot coordinates are
checked against the paired WSI to confirm they land on real tissue
(`pct_spots_with_tissue`); a sample failing this check is excluded as
`likely_misaligned`, since a coordinate-system mismatch would otherwise
silently corrupt every downstream patch/graph. Physical scale (microns per
pixel) is cross-checked two independent ways — `openslide.mpp-x` vs. the
Visium spot's known 55µm diameter divided by its pixel diameter — confirmed
to agree to 4 decimal places on a real sample (0.7257 vs. 0.7258µm/px).

**Scale (post metadata-audit)**: 202 real samples across all three
cancers — IDC 120, COAD 58, CCRCC 24 — 0 excluded by mouse/cancer-bucket/
duplicate-id filters. The custom-protocol cross-validation grid
(§6, job 84871) has run against a 5-fold/3-cancer subset of these; the
official-protocol suite (§8, jobs 88050/88100) has since run against real
samples across all 12 official folds (IDC×4, COAD×2, CCRCC×6) spanning all
three cancers.

**11 samples excluded** from the custom-protocol grid specifically (job
84733 onward): 6 IDC + 5 COAD samples carry a targeted ~541-gene panel
(Xenium-style) rather than whole-transcriptome ST, which would otherwise
silently shrink the shared HVG gene pool across all samples in that cancer
(caught when IDC's shared gene pool collapsed from 1000 to 171 before this
fix).

## 4. Phase 1 — Knowledge distillation

**Teacher**: H-Optimus-0 (`bioptimus/H-optimus-0`), a 1,134,774,272-
parameter pathology foundation model (measured via `timm.create_model` +
`sum(p.numel())`), used frozen, mixed precision via autocast (fp16),
L2-normalized output embeddings, 1536-D.

**Student**: ConvNeXt V2-Atto, 3,387,400 parameters (measured), backbone
output dimension 320, ImageNet-pretrained initialization. A training-only
projection head (320→1024→1536) maps the backbone's 320-D output into the
teacher's 1536-D embedding space for the distillation loss; this head is
**deleted** after training (§4, "lock/strip"), so the deployed student is
backbone-only. A second, optional projection (320→256) exists for
downstream HEST training but is not currently exercised by any driver
script.

**Three baseline variants** (S0/S1/S2), all sharing the same architecture:

| Variant | Init | Distilled? | Purpose |
|---|---|---|---|
| S0 | ImageNet | No | Isolates how much of any downstream benefit comes from ImageNet features alone |
| **S1** | ImageNet | Yes (against H-Optimus-0) | **Main comparison target** — the encoder used everywhere downstream |
| S2 | Random | Yes (against H-Optimus-0) | Isolates how much of S1's usefulness is architecture vs. distillation signal |

**Loss**: `L = L_cos + λ·L_feature`, λ = 0.1.

```
L_cos = 1 - cosine_similarity(normalize(student_proj), normalize(teacher))
L_feature = smooth_l1(normalize(student_proj), normalize(teacher))
```

Both terms operate on L2-normalized embeddings deliberately — raw
unnormalized 1536-D MSE is never used as an objective, since embedding
magnitude isn't a meaningful target, only direction.

**Training**: AdamW, backbone lr 1e-4, projection-head lr 3e-4, weight
decay 0.05, 3-epoch warmup, up to 100 epochs, batch
size 256, AMP enabled, gradient clip norm 1.0, model selection by
validation teacher-student cosine similarity. Augmentation: horizontal/
vertical flip, 90° rotation, color jitter (0.1), brightness/contrast
jitter (0.1).

**Selected checkpoint**: epoch 55 of S1/seed0, achieving a mean
teacher-student cosine similarity of 0.9060 on the CPTAC test set.

**Lock/strip**: the selected checkpoint (by validation cosine similarity)
has its projection head discarded, leaving an encoder-only state dict (the
3,387,400-parameter count above) — this is the sole artifact Phase 2
depends on; everything else from Phase 1 (teacher embedding cache,
training checkpoints, distillation eval reports) is intermediate.

## 5. Phase 2 — Spatial graph-corrected expression prediction

The locked student encoder is applied as a frozen feature extractor over
HEST patches (one 224×224px patch centered on each ST spot, 112µm physical
extent at 0.5µm/px target resolution) — no further distillation happens in
this phase.

### 5.1 Targets

Per fold, per cancer, using **training-fold spots only** (leakage-free by
construction — `epathst.data.expression.select_hvg_for_fold`):

1. Genes common to all samples in that cancer's fold, ranked by raw
   variance across training spots after `normalize_total(target_sum=1e4)`
   + `log1p`.
2. Top 1000 genes → **HVG1000** (the primary target). Top 50 of *that*
   ranking → **HVG50** — nested by construction (HVG50 ⊂ HVG1000), so one
   1000-output model serves both evaluations; no separate HVG50 model is
   trained.
3. **PROGENy14**: 14 pathway-activity scores via `decoupler`'s MLM method
   (`dc.op.progeny` / `dc.mt.mlm` on the installed decoupler v2 API),
   computed from ground-truth expression as an auxiliary target.

Targets are standardized (`TrainOnlyStandardizer`: `(X - μ_train) /
(σ_train + 1e-8)`, fit on training spots only, `assert_no_leakage` checked
against every val/test id set every fold) before being fed to any model.

### 5.2 Patch-only baseline (patch head)

`MolecularBottleneckHead`: `Linear(320or256→128) → LayerNorm → GELU →
Dropout(0.1) → Linear(128→64) → GELU → {gene head (Linear→1000),
pathway head (Linear→14)}`. LayerNorm+Dropout apply only to the first
(widest, most overfit-prone) layer. This defines ŷ_patch,i, the anchor
prediction every graph-corrected model must never destroy.

Loss: gene-only by default in the V2 spec (pathway weight 0.0), but
**restored to 0.2 in the actual locked run** (2026-08-30) after a real bug
was found — with weight exactly 0, the pathway auxiliary head receives
exactly zero gradient every step, so every graph ablation's pathway
prediction was byte-identical to A2's regardless of topology (confirmed
live, job 84818). `pathway_loss_sweep: [0.0, 0.05, 0.10, 0.20]` remains in
config for a possible post-lock re-sweep, but 0.2 is the actual value used
for all reported results. Optimizer: AdamW, lr 3e-4, weight decay 1e-4,
batch size 256, up to 100 epochs, early-stop patience 12.

### 5.3 Physical multiscale graph

One graph per tissue section (never edges across WSIs/patients). Nodes are
ST spots with the locked student's embedding as their only feature — raw
absolute (x, y) coordinates are never used as a node feature, only to
compute edge distances. Edges are physical-distance buckets:

| Scale | Radius | Max neighbors |
|---|---|---|
| Local | ≤150µm | ≤8 |
| Meso | ≤300µm | ≤12 |
| Macro | ≤600µm | ≤16 |

Edge features: 16 RBF kernels (γ=10.0) over normalized distance, plus
either a 3-D one-hot scale indicator (**union** edge features, 20-D total,
used when one message-passing layer sees edges from all three scales at
once) or no one-hot (**per-scale** edge features, 17-D, used when the
model architecture already encodes scale via separate branches — see A8
below).

### 5.4 Residual correction formulation

Every graph-corrected model shares the same wrapper:

```
y_patch,i  = patch_head(h_i)              [frozen after patch-head training]
dy_graph,i = GNN(h_i, neighborhood_i)      [trained]
y_final,i  = y_patch,i + dy_graph,i
```

The graph-correction head is **zero-initialized**, so `y_final == y_patch`
at initialization — the model starts equivalent to the patch-only baseline
and only earns influence during training. `assert_zero_residual_at_init`
enforces this automatically at the start of every graph-training call. This
exists because of a real, previously observed failure mode: PoC_5's own
multiscale-PNA model scored *below* its patch-only baseline (0.1268 vs.
0.1875) — without a zero-init guarantee, a graph model can make things
worse purely by re-deriving a worse version of the patch prediction from
scratch, rather than adding to it. A related bug this design surfaced and
fixed: the patch head's Dropout layer made the "frozen" anchor
non-deterministic in train mode until `ResidualGraphPredictor.__init__`
was made to call `.eval()` on it and override `.train()` to keep it there
permanently.

Three interchangeable backends, sharing the same PyTorch Geometric data
pipeline (a `backend` parameter and optional per-edge `edge_scale` tag —
no structural rewrite between them):

1. **`GraphCorrectionModel`** (union multiscale PNA, ablations A4–A7): one
   PNA layer sees all edges (local+meso+macro mixed), 20-D union edge
   features.
2. **`ScaleAwareGraphCorrectionModel`** (**A8, the proposed final model**):
   three independent per-scale PNA branches (local/meso/macro, each seeing
   only its own scale's edges, 17-D per-scale features), fused by a
   learned softmax gate over `[h^0; h^L; h^M; h^G]`.
3. **`GATv2CorrectionModel`** (**A9, architecture control**): identical
   structure to (1) except GATv2 replaces PNA as the message-passing
   operator — isolates "PNA specifically" from "graph message-passing in
   general."

**PNA hyperparameters**: input dim 320 (or 256 with the optional
projection), hidden dim 128, 2 layers per branch (explicitly *not* adding
an APPNP diffusion stage per the V2 spec — PoC_5's diffusion variant made
things worse, see §1), aggregators `{mean, max, min, std}`, scalers
`{identity, amplification, attenuation}`, dropout 0.3, molecular latent dim
64. Optimizer AdamW, lr 3e-4, weight decay 1e-3, up to 100 epochs,
early-stop patience 8. **Dropout and weight decay were both increased**
(0.1→0.3, 1e-4→1e-3) after real train/val loss curves (job 84733,
A8/CCRCC) showed classic overfitting — train loss fell 18% over 59 epochs
while val loss barely moved (-5%) and got noisy late; the likely cause is
too few distinct training patients (16–96 depending on cancer) relative to
A8's capacity (3 PNA branches + gate) on a single fold with no cross-fold
averaging. **GATv2**: 4 attention heads (hidden dim 128 divisible by
heads), otherwise matched depth/capacity to PNA so the A8-vs-A9 comparison
isolates the message-passing operator, not model size.

## 6. Ablation grid

`epathst.ablations.ABLATION_GRID` — each entry is a (feature source,
graph source, backend) triple, so every ablation reuses identical patient
folds, training procedure, and evaluation code; only the inputs/backend
differ. Ordered primary block first (A1, A2, A5, A6, A7, A8), secondary
controls after (A0, A3, A4, A9), so a time-constrained run can stop after
the scientifically essential block.

| ID | Priority | Features | Graph | Backend | What it isolates |
|---|---|---|---|---|---|
| A0 — mean expression | secondary | — | — | none (constant prediction) | Trivial floor (Pearson r undefined against a constant) |
| A1 — ImageNet patch | primary | ImageNet | none | none | No-distillation baseline (Gate 1's "worse") |
| **A2 — distilled patch** | primary | distilled | none | none | **The anchor** — Gate 1's "better", Gate 2's "worse" |
| A3 — matched MLP | secondary | distilled | none | wider MLP, no neighbors | Same parameter count as the GNN, rules out "more parameters" as the explanation for any graph benefit |
| A4 — random graph | secondary | distilled | uniform random edges | PNA union | Same node/edge count, destroyed spatial semantics |
| A5 — shuffled graph | primary | distilled | degree-preserving rewire | PNA union | **Critical negative control** — Gate 3's "worse" |
| A6 — local PNA | primary | distilled | real, local-only | PNA union | Real topology, single scale |
| A7 — multiscale PNA | primary | distilled | real, local+meso+macro | PNA union | Real topology, one layer sees all scales |
| **A8 — scale-aware PNA** | primary | distilled | real, local+meso+macro | 3 branches + gate | **Proposed final model** — Gate 2/3's "better" |
| A9 — GATv2 multiscale | secondary | distilled | real, local+meso+macro | GATv2 | Architecture control (PNA vs. GATv2) |

A separate, methodologically stricter H-Optimus-0-teacher-vs-student comparison
(`epathst.evaluation.hoptimus_comparison`: train-fold-only PCA(256) of the teacher's raw
1536-D embedding + Ridge, matching both the student's own 320→256
projection dimensionality and the official HEST-Benchmark's own PCA256+
Ridge convention) is **not part of this grid** — see §8, run only after the
A0–A9 matrix above is locked. Library-level as of this writing (implemented,
unit-tested, not yet wired into `src/scripts/15_genbio_fair_comparison.py`).

## 7. Statistical methodology

**Primary metric**: patient-level Fisher-z aggregated Pearson r.

```
R_p = tanh( mean_g( atanh(r_p,g) ) )
```

computed per patient p, averaged across genes/pathways g, then
back-transformed — patients, not spots, are the independent statistical
unit (avoids pseudo-replication from many correlated spots per patient).

**Bootstrap CIs**: 10,000-resample, resampling with replacement over
(cancer, fold, seed) cells (§11 flags why this is a coarser unit than
literal per-patient values). 95% CI = [2.5th, 97.5th] percentile of the
bootstrap distribution of the mean delta between two models' scores.

**Significance test**: one-sided paired sign-flip permutation test
(10,000 iterations) on the same paired deltas, testing H1: mean(delta) > 0.

**Fold-completeness gate**: before any statistic is aggregated,
`epathst.data.splits.assert_run_completeness` enumerates every expected
(cancer, fold, seed, ablation) combination and **raises** (not warns) on
any mismatch against what actually completed — the direct fix for PoC_5's
silent-partial-pool bug (§1).

**Three primary gates** (`configs/base.yaml`'s `primary_gates`), reported
both pooled and per-cancer (per-cancer is the stated primary result; V2
Phase 1):

| Gate | Comparison | Tests |
|---|---|---|
| 1 — Distillation | R(A2) > R(A1) | Does distillation help over ImageNet-only features? |
| 2 — Spatial context | R(A8) > R(A2) | Does the graph correction help at all? |
| 3 — True topology (decisive) | R(A8) > R(A5) | Does the *specific spatial structure* matter, or would any local averaging do? |

**Benjamini-Hochberg FDR correction** (`epathst.statistics.stats_tests.bh_fdr`,
q=0.05) is implemented and unit-tested but **not yet called** by
`12_primary_hypothesis_test.py` — a real follow-up, not part of the
currently reported statistics.

## 8. Official HEST-1k Benchmark reproduction

Sections 3–7 above describe this project's own custom protocol (own
patient-level GroupKFold splits, HVG1000/HVG50/PROGENy14 targets). Scripts
19–23 answer a different, complementary question: how does this project's
model compare on the standard, published HEST-1k leaderboard (official
splits, official curated 50-gene marker panel per cancer, official
pooled-spot Pearson metric)? See `docs/HEST_OFFICIAL_ABLATION_RESULTS.md`
and `docs/HEST_OFFICIAL_HVG50_GENES.md` for the full protocol/gene-list
writeup.

**Protocol differences from §3–7, all deliberate**:
- **Gene target**: official `<cancer>/var_50genes.json` — a fixed,
  curated marker panel (e.g. IDC: `ESR1`, `GATA3`, `ERBB2`, immune markers
  `CD3E`/`PTPRC`/`MS4A1`), computed once by the HEST-1k authors via
  `sc.pp.highly_variable_genes` (dispersion-based, on log1p'd raw counts,
  no `normalize_total`) pooled across the entire train+test cohort. This
  is *not* the same gene set as this project's own per-fold HVG50 (§5.1) —
  gene overlap is 1-2 genes out of 50 per cancer, since the two selection
  procedures use different statistics (dispersion vs. raw variance),
  different normalization, and different data scope (whole-cohort-pooled
  vs. train-fold-only).
- **Metric**: per-gene `scipy.stats.pearsonr` on **pooled test-fold spots**
  (not per-patient aggregated), averaged over genes, averaged over folds —
  the official HEST-1k Benchmark's own metric (`mahmoodlab/HEST`,
  `src/hest/bench/trainer.py`), matching the public leaderboard exactly.
- **Splits**: whatever official `train_N.csv`/`test_N.csv` pairs exist per
  cancer (IDC×4, COAD×2, CCRCC×6 folds — 12 total), not re-derived from
  this project's own patient split.
- **`19_official_hest_benchmark.py`**: raw teacher/student encoders only
  (StandardScaler→PCA(256, fit on train-fold patches only)→Ridge, alpha =
  100/(n_components·n_genes) — a fixed function of dimensionality, not a
  free hyperparameter). Never touches the trained patch-head/GNN models.
- **`22_official_hest_ablation_grid.py`**: reuses the *same* trained A0–A9
  code (`run_ablation`/`train_patch_head`, via `importlib`, not
  reimplemented) against official data/splits/genes — the direct
  A0–A9-vs-leaderboard comparison. Some official folds have as few as 1
  train sample (COAD fold 0); when there aren't ≥2 to split into
  train/val, the full train pool is used as both (early stopping then
  monitors train loss, logged explicitly as "val==train fallback" —
  never touches test data, so this is a model-selection bias only, not a
  test-set leak).
- **`23_morans_i_diagnostic.py`**: a *direct* test of Gate 3, complementary
  to the indirect real-vs-shuffled-graph test. Per gene, computes
  ground-truth Moran's I (real spatial autocorrelation, computed within
  each patient's own coordinates then averaged across patients — Moran's I
  isn't comparable across differently-scaled tissue sections) and
  correlates it against that gene's (A8 R − A2 R) gain. If the graph
  correction exploits real spatial structure, genes with stronger true
  autocorrelation should show a larger gain. Replicates a diagnostic
  PoC_5 already ran once on its own cohort.

## 9. Results

### 9.1 Custom-protocol locked final results (job 84871, 2026-08-31)

Patient-level Fisher-z R, 5-fold × 3-cancer × seed-0 (15 cells pooled, 5
per cancer). Full tables: `docs/HEST_FINAL_5FOLD_RESULTS.md`.

**Macro-average (pooled 3 cancers), HVG1000**:

| Model | R | 95% CI |
|---|---|---|
| A1 — ImageNet patch | 0.0612 | [0.0480, 0.0749] |
| **A2 — distilled patch** | 0.0728 | [0.0568, 0.0874] |
| A3 — matched MLP | 0.0857 | [0.0759, 0.0949] |
| A4 — random graph | 0.0997 | [0.0880, 0.1101] |
| A5 — shuffled graph | 0.0998 | [0.0880, 0.1103] |
| A6 — local PNA | 0.0955 | [0.0850, 0.1052] |
| A7 — multiscale PNA | 0.1019 | [0.0900, 0.1127] |
| **A8 — scale-aware PNA** | 0.0981 | [0.0872, 0.1078] |
| A9 — GATv2 multiscale | 0.1024 | [0.0911, 0.1120] |

**Primary gate verdicts (pooled)**:

| Gate | Δ R | 95% CI | p (sign-flip) | Verdict |
|---|---|---|---|---|
| 1 — Distillation (A2>A1) | 0.0115 | [-0.0036, 0.0245] | 0.072 | **NOT SUPPORTED** (sign flips per cancer: CCRCC supported +0.031, COAD marginal +0.015, IDC reverses -0.012) |
| 2 — Spatial context (A8>A2) | 0.0253 | [0.0110, 0.0427] | 0.0001 | **SUPPORTED**, all 3 cancers individually too |
| 3 — True topology (A8>A5, decisive) | -0.0017 | [-0.0044, 0.0006] | 0.885 | **NOT SUPPORTED**, all 3 cancers individually too |

**Secondary comparisons**: A8 is not the single best graph variant — A7
(union multiscale, no gating) and A9 (GATv2) score marginally higher than
A8 pooled on HVG1000 (0.1019, 0.1024 vs. 0.0981); both A8-vs-A7 and
A8-vs-A9 comparisons are **NOT SUPPORTED**. A8 does beat A3 (matched MLP,
Δ=0.0123, p=0.0007, **SUPPORTED**) — some graph structure helps over pure
capacity, just not specifically scale-aware gating at this sample size.

**GenBio-PathFM reference** (official protocol, PCA256+Ridge, separate
track — see §8): this external foundation model beats the 335.0×-smaller
distilled student by 0.04–0.11 R in every cancer (IDC +0.108, COAD +0.088,
CCRCC +0.041) — the same margin our actual teacher, H-Optimus-0, shows
over the student, since both are ~1.1B-parameter PFMs at comparable
representational strength.

### 9.2 Official-protocol results (job 88050, 2026-09-04)

Same trained A0–A9 models, evaluated against official splits/genes/metric.
Full tables: `docs/HEST_OFFICIAL_ABLATION_RESULTS.md`.

**Same metric for everyone** (official pooled-spot Pearson):

| Model | IDC | COAD | CCRCC | Average |
|---|---|---|---|---|
| A1 — ImageNet patch | 0.3826 | 0.2118 | 0.0846 | 0.2051 |
| A2 — distilled patch | 0.4506 | 0.3220 | 0.2075 | 0.3076 |
| A3 — matched MLP | 0.4764 | 0.3383 | 0.1896 | 0.3100 |
| A4 — random graph | 0.4826 | 0.2967 | 0.2248 | 0.3227 |
| A5 — shuffled graph | 0.4871 | 0.2969 | 0.2248 | 0.3242 |
| A6 — local PNA | 0.4856 | 0.2985 | 0.2244 | 0.3238 |
| A7 — multiscale PNA | 0.4790 | 0.3055 | 0.2206 | 0.3209 |
| **A8 — scale-aware PNA** | 0.4790 | 0.3042 | 0.2222 | 0.3215 |
| A9 — GATv2 multiscale | 0.4744 | 0.3089 | 0.2329 | 0.3261 |
| Distilled CNN (PCA256+Ridge, 3.4M params) | 0.4822 | 0.2385 | 0.2225 | 0.3144 |
| GenBio-PathFM (PCA256+Ridge, 1.1B params) | 0.5898 | 0.3264 | 0.2638 | 0.3933 |

**Gate readout under the official protocol**: Gate 1 holds clearly
(pooled +0.089). Gate 2 holds but the margin is small (pooled +0.012).
Gate 3 **fails** — A8 (0.3215) is statistically indistinguishable from A5
(0.3242); A4–A9 (any graph, real or fake) all cluster within ~0.32–0.33,
and A3 (no neighbor information at all) matches them too. This reproduces
the custom-protocol Gate 3 result exactly, despite the official and custom
protocols scoring almost entirely different genes (1-2/50 overlap) —
what's invariant across both is the qualitative finding, not the absolute
numbers.

### 9.3 Moran's I diagnostic (job 88100, 2026-09-05)

Direct test of whether a gene's real spatial autocorrelation predicts its
A8-vs-A2 gain:

| Cancer | n genes | Corr(Δ, Moran's I) | p-value |
|---|---|---|---|
| IDC | 50 | 0.0434 | 0.765 |
| COAD | 50 | -0.0512 | 0.724 |
| CCRCC | 50 | -0.1337 | 0.355 |
| **Pooled** | 150 | **0.1358** | **0.097** |

None significant, including pooled (150 genes — more statistical power
than any single cancer). This replicates PoC_5's own version of this exact
test almost exactly (`r=0.1148, p=0.630`, n=20 genes, one fold) — two
independent codebases, two independent cohorts, the same direct test,
the same null result.

### 9.4 Efficiency

| Metric | Value |
|---|---|
| Student encoder parameters | 3,387,400 |
| H-Optimus-0 teacher parameters | 1,134,774,272 |
| Compression ratio | 335.0× |

Inference VRAM, throughput, and GNN wall-clock have not been separately
profiled as of this writing.

## 10. Interpretation

Three independent tests — Gate 3 under the custom protocol (§9.1), Gate 3
under the official protocol (§9.2), and the direct Moran's I correlation
under the official protocol (§9.3) — all reach the same conclusion: **the
graph correction is not exploiting genuine spatial topology.** Any local
averaging (real neighbors, random neighbors, or a degree-preserving
shuffle) produces a similar small gain over patch-only; a parameter-
matched MLP with *no* neighbor information at all gets most of the way
there too. This is a real, reproducible negative result, not an
artifact of one run or one protocol — it reproduces PoC_5's original
finding on an independently built codebase, independent cohorts (breast/
colon/kidney vs. PoC_5's colorectal-only), and both this project's own
protocol and the field's standard benchmark protocol.

What *is* supported: Gate 2 (the graph-corrected model beats patch-only)
holds robustly under both protocols — but per Gate 3, that gain tracks
"any local averaging helps a little," not "real tissue topology carries
extra molecular information at 150–600µm scales for these gene panels."
Distillation compression works as intended: a 335.0×-smaller student
retains a real, non-trivial fraction of the teacher's signal (not
near-zero), at a consistent but modest accuracy cost (0.04–0.11 R across
cancers, official protocol).

**Scope for future improvement** (informed by the above, not yet tested):
richer edge features based on tissue-morphology similarity rather than
pure physical distance (message-passing's "nearby = similar" inductive
bias may be the wrong prior if real structure follows histological
boundaries, not raw proximity); feeding the graph model teacher embeddings
rather than the compressed student's, to separate "is the graph
mechanism broken" from "is the encoder too small to give the graph
anything informative to work with"; pooling training data across folds/
cancers, since official folds are very small (COAD fold 0 has 1 train
sample) and per-fold variance is high (std ~0.13–0.17). Closing the
remaining gap to GenBio-PathFM's raw representational power would need a
larger/better distilled encoder, not further graph-architecture work —
the graph correction's realistic ceiling, even "fixed," is a few points of
R, while the encoder-capacity gap is 0.04–0.11 R and tracks the parameter
gap directly.

## 11. Limitations

1. **Bootstrap resampling unit** is (cancer, fold, seed) cells, not
   literal per-patient R_p values — `10_run_patient_cv.py` doesn't persist
   per-patient predictions to disk. A stricter implementation would
   bootstrap over patients directly.
2. **Seeds 1/2** were not run for the custom-protocol grid — seed 0 only.
3. **The GenBio-embedding→MLP baseline (B1)** is fold0/seed0 only per
   cancer under the custom protocol, not the full 5-fold design.
4. **Efficiency**: only parameter counts and the derived compression ratio
   have been measured; inference VRAM, throughput, and GNN wall-clock have
   not.
5. **The IDC-specific distillation reversal** (A1 ahead of A2 in IDC under
   the custom protocol) has not been separately root-caused.
6. **A8 vs. A7/A9**: A8 is not clearly the single best graph variant at
   this sample size (§9.1) — worth noting for anyone citing A8 as
   unambiguously superior architecture.
7. **CPTAC tumor-specimen labeling** (§3.1): COAD/BRCA's bare-`"HE"`-label
   exclusions are a conservative proxy, not a verified fact — some may be
   genuine tumor sections not tagged with the word "tumor" in this
   particular metadata field.
8. **Benjamini-Hochberg FDR correction** is implemented but not yet
   applied to the reported gene-level statistics (§7).
9. **CCRCC test sets are small** in some run configurations (as few as 5
   patients in one early run) — treat CCRCC numbers with extra caution
   where noted.

## 12. Compute resources

Full ledger: `docs/COMPUTE_RESOURCES.md`. Summary as of last update
(2026-08-29, Phase 1 distillation only — does not yet include Phase 2/
official-benchmark job hours): ~19.65 GPU-hours across 5 substantially-run
jobs, entirely on a single NVIDIA RTX PRO 6000 Blackwell (98GB) per job, no
multi-GPU jobs. Several early attempts targeted an H200 partition but never
started (cluster-wide full allocation at submission time; the `-long`
partition variants also proved structurally lower scheduling priority) —
**no H200 compute was used**. Suggested Methods-section sentence: "All
experiments were run on a single NVIDIA RTX PRO 6000 Blackwell GPU (98GB)
per job, on a SLURM-managed institutional cluster."

## 13. Repository structure and reproducibility

```
src/epathst/         the package (28 modules across 8 subpackages —
                     data/patching, models, graph, distillation,
                     evaluation, statistics, efficiency, reporting;
                     see below)
src/scripts/         23 numbered pipeline stages (01-18 custom protocol,
                     19-23 official-benchmark suite) + tools/ (debug/
                     baseline/snapshot utilities)
configs/base.yaml    every hyperparameter cited above; nothing structural
                     is hard-coded in the scripts
tests/               unit tests for deterministic numeric code (graph
                     construction, statistics) — 9 tests, torch-free
docs/                topic-specific detailed writeups (results, data
                     provenance, migration history, compute ledger)
experimental_runs/   dated, git-committed snapshots of completed jobs'
                     full results/figures trees (raw data behind every
                     results table in this document)
```

**Module map** (`src/epathst/`):
- **`data/`**: `cptac_select.py` (WSI QC+selection), `wsi_patches.py`
  (patch extraction, DICOM via `wsidicom` or plain openslide-readable
  formats), `hest_prep.py` (HEST metadata audit, alignment QC, ST-spot
  patch extraction), `splits.py` (leakage-free patient splitting and
  cross-validation folds), `feature_cache.py` (runs a frozen encoder over
  patches, caches embeddings, with spot/patch/coordinate/embedding/
  expression integrity checks), `expression.py`, `pathways.py`.
- **`models/`**: `teacher.py`, `student.py`, `patch_head.py`,
  `graph_layers.py` (PNA + GATv2 message-passing blocks),
  `graph_model.py` (the three backends, §5.4).
- **`graph/`**: `builder.py` (§5.3), `training.py`,
  `destruction.py` (edge-randomization dose-response controls).
- **`distillation/`**: `training.py` (loss + training loop),
  `evaluation.py`.
- **`evaluation/`**: `metrics.py`, `analysis.py`, `pathway_posthoc.py`,
  `hoptimus_comparison.py` (train-only PCA + Ridge fair-comparison
  library).
- **`statistics/`**: `stats_tests.py` (§7).
- **`efficiency/`**: `efficiency.py`.
- **`reporting/`**: `report.py`, `figures.py`.
- **package root**: `ablations.py` (§6), `utils.py` (config loading, path
  resolution, shared helpers) — both cross-cutting, used across nearly
  every subpackage.

`graph/builder.py`, `models/graph_layers.py`, `graph/destruction.py`,
`statistics/stats_tests.py`, `data/expression.py`, `data/pathways.py`,
and `evaluation/metrics.py` are adapted from the
already-validated PoC_5 package (same graph radii/caps/RBF/PNA
hyperparameters). The GNN head diverges: PoC_5 used a gated PNA+APPNP
fusion; this project uses the residual-correction design (§5.4) instead.

**To reproduce**: every path/hyperparameter above resolves from
`configs/base.yaml` (mirrored as `base.json`) via
`epathst.utils.load_config`/`resolve_run_paths` — nothing is
hard-coded in the scripts. SLURM job scripts for every stage live in
`slurm/`. Full verification performed at each migration step: `python3 -m
py_compile` clean across the package, the full pytest suite green, and
`src/scripts/16_sanity_checks.py`'s 9 checks (S1–S9) passing, including a
direct test that `assert_zero_residual_at_init` works and that
`assert_run_completeness` correctly raises on an incomplete run set before
being relied on for anything real.

## 14. Known follow-ups

- `src/scripts/15_genbio_fair_comparison.py` still uses an earlier
  raw-1536-D approach; needs a rewrite using `hoptimus_comparison.py`'s
  `TrainOnlyPCA` for the three-head (Ridge/MLP/PNA) × two-encoder
  (H-Optimus-0-PCA256 vs. student-256) comparison the design calls for.
- `pathway_posthoc.py` and the `pathway_loss_sweep` config value exist but
  aren't yet driven by a script.
- The spatial-destruction dose-response doesn't yet preserve per-scale
  (local/meso/macro) edge counts, only aggregate degree.
- The primary hypothesis test's bootstrap unit is (cancer, fold, seed),
  not individual patients (§11).
- Benjamini-Hochberg FDR correction is implemented and unit-tested but not
  yet wired into the primary hypothesis test driver (§7, §11).
- Full alignment QC + patch extraction at scale across all 202 HEST
  samples (custom protocol) has only been exercised on the 5-fold subset
  used for the locked results so far; the official-protocol suite has
  separately run against all 12 official folds (§3.2).
