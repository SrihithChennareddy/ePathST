# Ablation Results Under the Official HEST-1k Benchmark Protocol

**Created**: 2026-09-04 (SLURM job 87551) · **Last updated**: 2026-09-14 (superseded numbers — see [Revision history](#revision-history))

## Overview

This document reports ePathST's ablation results under the **official HEST-1k
Benchmark protocol** — official k-fold splits, the official curated 50-gene
marker panel per cancer, and the official pooled-spot Pearson correlation
coefficient (PCC), matching the public HEST-1k leaderboard's own metric
(`mahmoodlab/HEST`, `src/hest/bench/trainer.py`) exactly. This is the
evaluation reported in the paper, *Efficient Histopathology Representation
Learning for Spatial Gene Expression Prediction* (Chennareddy & Yurovsky,
2026) — see `docs/reports/DESIGN.md` §8-§9 for the full methods writeup this
document's numbers are drawn from.

A companion evaluation under this project's own **custom protocol** (own
patient-level splits, per-fold-selected HVG1000/HVG50, patient-level Fisher-z
aggregation) is reported separately in
[`HEST_FINAL_5FOLD_RESULTS.md`](HEST_FINAL_5FOLD_RESULTS.md) — the two use
different gene panels (1-2 genes overlap out of 50 per cancer) and different
metrics, so their absolute numbers are not directly comparable. What
replicates across both is the qualitative finding on real vs. shuffled
spatial topology (§Gate readout below).

## Models evaluated

Only the models reported in the paper are included below. The full A0-A9
ablation grid (including A1 ImageNet-only, A3 capacity-matched MLP, A4
random-graph, A5 shuffled-graph, and A9 GATv2) exists in this project's
codebase and has been run under this protocol (job 87551), but the paper
does not report updated numbers for those rows — see
[Revision history](#revision-history) for why they are omitted here rather
than shown with stale numbers.

| Model | Architecture |
|---|---|
| ePathST-E | Frozen distilled CNN encoder, patch-only, no graph correction |
| Local-only PNA | + graph correction, local scale only (≤150µm, ≤8 neighbors) |
| Meso-only PNA | + graph correction, mesoscopic scale only (≤300µm, ≤12 neighbors) |
| Macro-only PNA | + graph correction, macroscopic scale only (≤600µm, ≤16 neighbors) |
| Multiscale PNA | + graph correction, all three scales, one PNA layer sees all edges (no gating) |
| **ePathST** | + graph correction, all three scales, per-scale PNA branches + learned softmax gate (flagship model) |

## Results

Official pooled-spot Pearson correlation coefficient, per cancer and
averaged (equal-weighted mean across the three cancers):

| Model | IDC | COAD | CCRCC | Average |
|---|---|---|---|---|
| ePathST-E (patch-only) | 0.4890 | 0.3151 | 0.2275 | 0.3438 |
| Local-only PNA | 0.5161 | 0.3238 | 0.2316 | 0.3572 |
| Meso-only PNA | 0.5237 | 0.3261 | 0.2328 | 0.3609 |
| Macro-only PNA | 0.5194 | 0.3249 | 0.2341 | 0.3595 |
| Multiscale PNA (no gate) | 0.5272 | 0.3256 | 0.2337 | 0.3622 |
| **ePathST (scale-aware PNA)** | **0.5384** | **0.3302** | **0.2375** | **0.3687** |

ePathST's gain over the patch-only anchor is **+0.0249** average PCC
(0.3687 vs. 0.3438); its gain over conventional ungated multiscale fusion is
a further **+0.0065** (0.3687 vs. 0.3622). Among single fixed scales,
meso-only is strongest (0.3609), narrowly ahead of macro-only (0.3595) and
local-only (0.3572) — but no single fixed scale matches multiscale or
scale-aware fusion; combining scales helps, and learned gating over the
combination helps a little further.

## Comparison against published pathology foundation models

Parameter counts and relative size are normalized to ePathST's full
5.25M-parameter system (frozen CNN encoder + GNN together, not the 3.39M
encoder-only variant).

| Method | Params (M) | Relative size | IDC | COAD | CCRCC | Average |
|---|---|---|---|---|---|---|
| GigaPath | 1135 | 216.19× | 0.5515 | 0.2992 | 0.2430 | 0.3646 |
| H-Optimus-0 (this project's teacher) | 1134 | 216.00× | **0.5976** | 0.3086 | 0.2676 | 0.3913 |
| GenBio-PathFM | 1133 | 215.94× | 0.5872 | 0.3284 | 0.2615 | **0.3924** |
| UNI2-h | 681 | 129.71× | 0.5898 | 0.3015 | 0.2640 | 0.3851 |
| Virchow2 | 632 | 120.38× | 0.5971 | 0.2581 | **0.2719** | 0.3757 |
| CONCH v1.5 | 307 | 58.48× | 0.5440 | 0.2802 | 0.2176 | 0.3473 |
| Kaiko ViT-B/16 | 86 | 16.38× | 0.5352 | 0.2812 | 0.2291 | 0.3485 |
| CTransPath | 27.5 | 5.24× | 0.4993 | 0.2382 | 0.2362 | 0.3246 |
| ResNet50 | 25.5 | 4.86× | 0.4739 | 0.2500 | 0.2252 | 0.3164 |
| ePathST-E (this project, encoder only) | 3.39 | 0.65× | 0.4890 | 0.3151 | 0.2275 | 0.3438 |
| **ePathST** (this project, full system) | 5.25 | 1.00× | 0.5384 | **0.3302** | 0.2375 | 0.3687 |

ePathST is the single best model on COAD among every model listed, including
every pathology foundation model at 129-1231× its parameter count, and
retains ~94% of H-Optimus-0's average PCC (0.3687 / 0.3913) at 216× fewer
parameters. See `docs/reports/DESIGN.md` §9.5 for the accompanying
compute-normalized (TFLOPs/WSI) efficiency analysis.

## Gate readout

- **Gate 2 (spatial context: does graph correction beat patch-only?)**
  **holds**, and more clearly than earlier runs of this evaluation
  suggested: +0.0249 average PCC (ePathST vs. ePathST-E).
- **Gate 3 (true topology: does *real* spatial structure matter, vs. any
  local averaging?)** — **not verifiable from the paper's own tables**,
  since the paper does not report a shuffled-graph negative control. Under
  this project's own custom protocol, the equivalent test (job 84871,
  [`HEST_FINAL_5FOLD_RESULTS.md`](HEST_FINAL_5FOLD_RESULTS.md) §4) and the
  earlier version of this official-protocol evaluation (job 87551, see
  Revision history) **both found Gate 3 fails** — a real, degree-preserving
  shuffled graph performs statistically indistinguishably from the real
  spatial graph. Whether that finding still holds against the improved
  ePathST numbers above is an open question until the shuffled-graph
  control is rerun under the same setup that produced these numbers.
- **Gate 1 (distillation: does the distilled encoder beat ImageNet-only
  features?)** — similarly not verifiable from the paper alone (no A1
  row reported). Held clearly under job 87551 (pooled +0.089); unconfirmed
  against the run that produced the numbers above.

## Leakage check

Verified for the underlying evaluation infrastructure (`src/scripts/22_official_hest_ablation_grid.py`,
`src/scripts/10_run_patient_cv.py::run_ablation`), and unchanged by the
numeric revision below:

- Official train/test splits are sample-disjoint: 0 overlapping `sample_id`s
  across all 12 official train/test fold pairs.
- `TrainOnlyStandardizer.assert_no_leakage(ids_test)` is called
  unconditionally every fold.
- Feature encoders (distilled + ImageNet) are frozen/eval-mode.
- Graphs are built strictly per `sample_id`; train/val/test graphs only ever
  contain nodes from their own disjoint sample pool.
- Official folds with fewer than 2 training samples (COAD fold 0) use the
  full training pool as both train and validation, so early stopping there
  monitors training loss rather than a held-out signal — an optimistic
  *model-selection* bias only, logged explicitly as `val==train fallback`;
  it never touches test data and does not leak into reported test metrics.

## Revision history

- **2026-09-14**: replaced this document's numbers with the paper's final
  Table III/V figures for the models the paper reports (ePathST-E, ePathST,
  local/meso/macro/multiscale PNA), and added meso-only and macro-only rows
  that did not previously exist in this project's ablation grid at all.
  Two things changed at once and are worth stating plainly:
  1. **Metric correction.** The original version of this document (below)
     reported *Fisher-z-aggregated patient-level Pearson R*, not the
     official HEST-1k Benchmark's actual pooled-spot Pearson metric —
     despite the document's title. The numbers above use the correct
     official metric, matching the paper.
  2. **Model improvement.** Independently of the metric fix, the paper's
     own numbers for ePathST-E/ePathST are higher than job 87551's
     equivalent (mismatched-metric) rows by a margin too large to be
     metric-related alone (e.g. patch-only 0.3438 vs. job 87551's
     0.3100 Fisher-z R). An external reference point, GenBio-PathFM, moved
     by less than 0.001 between the two evaluations, which rules out a
     scoring/pipeline change as the explanation and points to a later,
     improved training run feeding the paper. **What exactly changed is
     not documented anywhere found so far** — flagged here rather than
     guessed at (see `docs/reports/DESIGN.md` §14 for the corresponding
     follow-up item).
  - A0/A1/A3/A4/A5/A9 are omitted from the tables above (rather than shown
    with unconfirmed numbers) because the paper does not report updated
    values for them. The original job-87551 numbers for those rows,
    computed under the (mismatched) Fisher-z metric, are preserved here for
    the historical record:

    | Ablation | Fisher-z R (mean, n=12 folds) | R (std) |
    |---|---|---|
    | A0_mean_expression | n/a (no model) | — |
    | A1_imagenet_patch | 0.2214 | 0.1557 |
    | A3_matched_mlp | 0.3157 | 0.1794 |
    | A4_random_graph | 0.3197 | 0.1669 |
    | A5_shuffled_graph (negative control) | 0.3204 | 0.1685 |
    | A9_gatv2_multiscale | 0.3203 | 0.1661 |

    Job 87551 metadata: all 3 cancers, all 12 official k-folds, seed=1
    (official `BenchmarkConfig` default), feature-seed=0 (locked distilled
    encoder). Runtime 26 minutes (2026-09-04, 11:34-12:01). Driver:
    `src/scripts/22_official_hest_ablation_grid.py`.

## Output files

`/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/cptac_genbio_distill_hest_pna_work/results/official_hest_ablation_grid/`
- `official_ablation_metrics.csv` — one row per cancer × fold × ablation (120 rows, job 87551)
- `official_ablation_summary_macro.csv` — mean/std per ablation, pooled across folds/cancers (job 87551)
- `run_args.json` — exact CLI args used
