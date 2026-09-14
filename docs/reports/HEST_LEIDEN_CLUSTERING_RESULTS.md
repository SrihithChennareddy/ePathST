# Leiden Clustering Comparison: Predicted vs. Ground-Truth Spatial Domains

**Created**: 2026-09-14 (transcribed from the paper's qualitative evaluation) · **Last updated**: 2026-09-14

## Overview

Predictive accuracy per gene/spot (the pooled-spot Pearson tables in
[`HEST_OFFICIAL_ABLATION_RESULTS.md`](HEST_OFFICIAL_ABLATION_RESULTS.md))
answers "how close is each predicted value to ground truth," but not
"does the predicted expression recover the same *spatial domain
structure* a pathologist or downstream analysis would find." This
document reports the paper's complementary qualitative check: unsupervised
**Leiden clustering** applied independently to ground-truth expression and
to each model's predicted expression, then compared.

Source: *Efficient Histopathology Representation Learning for Spatial Gene
Expression Prediction* (Chennareddy & Yurovsky, 2026), §IV qualitative
evaluation, Fig. 3-4. See `docs/reports/DESIGN.md` §9.4 for how this fits
into the project's full results narrative.

## Method

For a representative HEST-1K sample, Leiden clustering (standard
`scanpy`/Louvain-family community detection over the expression-derived
neighbor graph, default resolution) is run twice:

1. On **ground-truth** measured expression → the reference spatial domain
   partition.
2. On each model's **predicted** expression, independently, using the same
   clustering procedure and parameters.

The two partitions are compared with a cluster-agreement score (higher =
predicted domains better match the true spatial domain structure). This is
a single-sample qualitative diagnostic, not a cross-fold statistic — it
answers "does this look right," complementing the quantitative tables
elsewhere in `docs/`, not replacing them.

## Results

| Model | Cluster-agreement score | Role |
|---|---|---|
| ePathST-E (patch-only, no graph correction) | 0.58 | This project's patch-only anchor |
| **ePathST (scale-aware PNA)** | **0.66** | This project's proposed final model |
| H-Optimus-0 (frozen 1.13B-parameter teacher, PCA256+Ridge) | 0.69 | Full-scale reference — not a fair capacity comparison (216× more parameters), context only |

**Reading**: the ordering (patch-only < scale-aware ePathST < H-Optimus-0)
matches the quantitative pooled-spot Pearson ordering reported in
[`HEST_OFFICIAL_ABLATION_RESULTS.md`](HEST_OFFICIAL_ABLATION_RESULTS.md)
(ePathST-E 0.3438 avg PCC < ePathST 0.3687 < H-Optimus-0 0.3913). Spatial
graph correction (ePathST vs. ePathST-E) recovers +0.08 cluster-agreement
— a larger relative jump than the +0.0249 average-PCC gain over the same
comparison — suggesting the graph correction's benefit shows up more
clearly at the level of coherent spatial domains than at the level of
individual spot/gene accuracy. ePathST closes most (but not all) of the
gap to the 216×-larger H-Optimus-0 teacher on this diagnostic (0.66 vs.
0.69, a 0.03 gap, vs. patch-only's 0.11 gap).

## Per-gene spatial prediction maps (companion diagnostic, paper Fig. 4)

A second, gene-level qualitative check: for one representative gene per
cancer, predicted-vs-ground-truth spatial expression maps are compared
visually, with a per-sample Pearson PCC reported alongside:

| Cancer | Gene | ePathST PCC | H-Optimus-0 PCC | Note |
|---|---|---|---|---|
| IDC | GATA3 | 0.88 | 0.89 | Near parity |
| COAD | CEACAM6 | **0.79** | 0.39 | ePathST substantially ahead on this specific gene/sample |
| CCRCC | STK39 | 0.48 | 0.50 | Near parity |

These are **illustrative single-sample/single-gene** comparisons, not
aggregate statistics. COAD's large gap here is consistent with COAD being
ePathST's strongest cancer in the aggregate comparison
([`HEST_OFFICIAL_ABLATION_RESULTS.md`](HEST_OFFICIAL_ABLATION_RESULTS.md)),
but should not be read as evidence that ePathST generally beats
H-Optimus-0 on COAD — the aggregate pooled-spot table is the
statistically meaningful comparison; this table exists to show what a
prediction map actually looks like for one gene, not to establish a trend.

## Caveats

- Single representative sample per cluster-agreement score, single
  gene/sample per per-gene map — not aggregated across folds or cancers,
  so no confidence interval is reported or should be inferred.
- H-Optimus-0 is shown as a full-scale reference for context (it is this
  project's own distillation teacher), not as a same-capacity baseline —
  it has ~216× ePathST's parameter count.
- This diagnostic has not been run under this project's own custom
  protocol (only the official-protocol paper run); no HVG1000/HVG50
  custom-protocol equivalent currently exists.
