# HEST-1K: ablation results and protocol notes

The paper's Table III (comparison with pathology foundation models) and Table V (spatial-context ablation) are reproduced in
[`docs/METHODS.md`](../METHODS.md) section 6, which is their single source in this repository. This document holds what surrounds them:
the evaluation protocol, how the paper reads the ablation, the controls that were run in addition, and the validation checks.

## 1. Evaluation protocol (as in the paper)

- **Genes and metric:** the official HEST-1K 50-gene highly variable panel (HVG50) of each cancer; Pearson correlation (PCC) per gene on held-out
  samples, averaged over genes within each cancer, then over the three cancers for the average.
- **Partitions:** patient-disjoint. The held-out test partition is the official HEST-Benchmark test folds (IDC 4, COAD 2, ccRCC 6 folds);
  model selection uses validation data only and the test partition is used once, for the final evaluation.
- **Baselines (Table III, foundation models):** official frozen-encoder protocol, PCA to 256 dimensions followed by Ridge regression.
- **ePathST-E, ePathST and the ablation variants:** identical patient-level partitions and gene targets, trained on the HEST-1K cohort slides
  outside the held-out test partition (about 200 slides across the three cancers) and evaluated on the held-out test partition.
- **Implementation:** `src/scripts/19_official_hest_benchmark.py` (baselines), `22_official_hest_ablation_grid.py` (ablation grid), `23_morans_i_diagnostic.py`.
  The panel gene lists are in [`HEST_OFFICIAL_HVG50_GENES.md`](HEST_OFFICIAL_HVG50_GENES.md).

## 2. How the paper reads the ablation

- Every spatial-context variant improves on the patch-only ePathST-E (average PCC 0.3438).
- Mesoscopic context is the strongest single scale on average (0.3609). The best single scale varies by cancer: macroscopic is strongest among
  the single-scale variants for ccRCC (0.2341).
- Multiscale PNA raises the average to 0.3622 but does not consistently beat the best single scale within a cohort.
- Scale-aware ePathST is best in IDC, COAD and ccRCC (average 0.3687): +0.0249 over ePathST-E, +0.0065 over multiscale PNA.
- At gene level the best scale is target dependent. ERBB2 (HER2) is already well predicted from local morphology and gains from broader context;
  the immune gene CD3E gains much more from context, consistent with dependence on the surrounding T-cell microenvironment.
- Against pathology foundation models ePathST keeps about 94% of H-Optimus-0's average PCC with 215x fewer parameters and is the best model on COAD.

## 3. Additional controls: full A0-A9 grid (earlier controlled run)

The full A0-A9 grid (`src/epathst/ablations.py`) was also run in a separate controlled experiment, with the models trained on the benchmark's own
training slides in each of the 12 official folds (benchmark-fold training; IDC 3, COAD 1 to 3, ccRCC 20 slides per fold) and scored on the official
test slides, genes and metric. Seed 1; mean and SD over the 12 fold cells (patient-level Fisher-z R and pooled-spot PCC).

| Model | Fisher-z R (mean, SD) | Pooled-spot PCC (mean, SD) |
|---|---|---|
| A1 ImageNet encoder, patch-only | 0.2214, 0.1557 | 0.2051, 0.1478 |
| A2 distilled encoder, patch-only (ePathST-E) | 0.3100, 0.1584 | 0.3076, 0.1267 |
| A3 parameter-matched MLP | 0.3157, 0.1795 | 0.3100, 0.1528 |
| A4 random graph | 0.3200, 0.1672 | 0.3227, 0.1362 |
| A5 degree-preserving shuffled graph | 0.3211, 0.1693 | 0.3242, 0.1365 |
| A6 local PNA | 0.3213, 0.1704 | 0.3238, 0.1367 |
| A7 multiscale PNA | 0.3207, 0.1669 | 0.3209, 0.1327 |
| A8 scale-aware PNA (ePathST) | 0.3199, 0.1686 | 0.3215, 0.1331 |
| A9 GATv2 multiscale | 0.3203, 0.1661 | 0.3261, 0.1284 |

Reading: the distilled encoder beats the ImageNet encoder (A2 vs A1), and every graph variant beats patch-only by about 0.01 to 0.02. In this
controlled run the shuffled-graph control (A5) scores as high as the real scale-aware graph (A8), so it does not show that the real physical
topology, as opposed to generic neighbourhood aggregation, carries the gain. Fold-to-fold SD (0.13 to 0.18) is large relative to differences
between graph variants. Files: `experimental_runs/2026-09-05_job88050_official_hest_ablation_grid/`.

**Moran's I diagnostic** (`23_morans_i_diagnostic.py`; per gene, spatial autocorrelation against the A8 minus A2 gain): IDC r = 0.043 (p = 0.77),
COAD -0.051 (p = 0.72), ccRCC -0.134 (p = 0.36), pooled over 150 genes 0.136 (p = 0.097). None is significant.

## 4. Validation checks (implementation level)

- Official train and test samples are disjoint in every one of the 12 official folds (0 overlapping sample ids).
- `TrainOnlyStandardizer.assert_no_leakage` runs in every fold; encoders are frozen and in evaluation mode; graphs are built per sample, so
  train, validation and test graphs only contain nodes from their own sample pool.
- Where an official fold has fewer than 2 training slides (COAD fold 0) the single training pool also serves as validation; this affects model
  selection only and never touches test data.

## 5. Output files of the controlled run

Snapshots are in `experimental_runs/2026-09-05_job88050_official_hest_ablation_grid/` (`official_ablation_metrics.csv`, one row per cancer x fold x
ablation, and `official_ablation_summary_macro.csv`); run arguments in `run_args.json`.
