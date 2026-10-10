# Step 1 audit report

Inputs read: TCGA predictions, embeddings, slide metadata, HEST split records and HEST held-out predictions. No RNA or outcome data.

| # | Check | Result | Pass |
|---|---|---|---|
| A1 | Inventory | TCGA-BRCA: 1030 files, E/P/T present; TCGA-COAD: 413 files, E/P/T present; TCGA-KIRC: 488 files, E/P/T present | PASS |
| A2 | Slide type | non-DX slides: 0 | PASS |
| A3 | One slide per patient | TCGA-BRCA: 1045 patients, DX1 1010, other DX 35; TCGA-COAD: 429 patients, DX1 426, other DX 3; TCGA-KIRC: 488 patients, DX1 486, other DX 2; duplicates 0 | PASS |
| A4 | Panels | fold heads agree and every slide's gene axis matches the head panel | PASS |
| A5 | Scale | stored values are fold-averaged on the original log1p scale (median |TCGA mean - HEST train mean| = 0.10); same scale for E, P, T | PASS |
| A6 | Ensemble and seed | TCGA-BRCA: seed 0, folds IDC_fold0;IDC_fold1;IDC_fold2;IDC_fold3; TCGA-COAD: seed 0, folds COAD_fold0;COAD_fold1; TCGA-KIRC: seed 0, folds CCRCC_fold0;CCRCC_fold1;CCRCC_fold2;CCRCC_fold3;CCRCC_fold4;CCRCC_fold5 | PASS |
| A7 | Collapse rule | validation predictions were not stored; on held-out TEST predictions 2 of 72 fold x seed x system models meet a collapse criterion; see details below | DECISION NEEDED |
| A9 | HEST training sets | CCRCC: train 17-17 slides/fold; COAD: train 1-2 slides/fold; IDC: train 2-2 slides/fold; overlaps 0 | PASS |
| A10 | Resolution | 224 px at 0.5 um/px (112 um field) for all predicted slides; 21 slides lack mpp and were not predicted | PASS |
| A11 | Grid | median nearest-neighbour 100.0 um (pitch 100) | PASS |
| A12 | Graph (P) | median degree local 6, meso 27, macro 83; isolated (local) 0.0044; whole-slide graph, not tiled (no halo needed) | PASS |
| A13 | Coverage and numerics | spots/slide median 15132 (IQR 6966-23197); NaN/Inf 0; slides < 200 spots 9 (not predicted) | PASS |
| A14 | Embeddings for U3 | per-spot student 320-D and raw teacher 1536-D on the same spots: True | PASS |
| A15 | HEST per-gene results | per-spot held-out predictions stored for every fold, seed and system (standardised scale); per-gene PCC computable | PASS |

## A7 detail: IDC fold 2

|   seed | system   |   test_pcc_mean_over_genes |   median_pred_sd_over_target_sd |
|-------:|:---------|---------------------------:|--------------------------------:|
|      0 | E        |                     0.0114 |                          0.0903 |
|      0 | P        |                     0.4626 |                          0.533  |
|      0 | T        |                     0.5323 |                          0.5186 |
|      1 | E        |                     0.4098 |                          0.4578 |
|      1 | P        |                     0.4528 |                          0.6625 |
|      1 | T        |                     0.4571 |                          0.5397 |

Models meeting a criterion on test predictions:

|   seed | cancer   |   fold | system   |   test_pcc_mean_over_genes |   median_pred_sd_over_target_sd |
|-------:|:---------|-------:|:---------|---------------------------:|--------------------------------:|
|      0 | IDC      |      2 | E        |                     0.0114 |                          0.0903 |
|      1 | CCRCC    |      4 | T        |                    -0.0238 |                          0.1189 |
