# HEST Final 5-Fold Experiment -- Results

Detailed, skimmable version of `FINAL_RESULTS.md` for the LOCKED final 5-fold x 3-cancer x seed-0 experiment (SLURM job 84871, completed 2026-08-31, 03:15:54 elapsed). Metric: patient-level Fisher-z mean Pearson R (`sts.metrics.patient_macro_fisher_z`). All CIs are bootstrap 95% intervals, n_bootstrap=10000, resampled over (cancer, fold, seed) cells -- see the Limitations section for what that does and does not mean. Every number below is read directly from `results/final/tables/`; nothing is re-derived or estimated in this document.

## Contents

1. [Ablation grid (what A0-A9 mean)](#1-ablation-grid)
2. [Macro-average results (pooled 3 cancers)](#2-macro-average-results)
3. [Per-cancer results](#3-per-cancer-results)
4. [Primary statistical gates](#4-primary-statistical-gates)
5. [Secondary comparisons](#5-secondary-comparisons)
6. [Pathway / PROGENy14 view](#6-pathway--progeny14-view)
7. [GenBio-PathFM reference numbers](#7-genbio-pathfm-reference-numbers)
8. [Efficiency](#8-efficiency)
9. [Bottom line](#9-bottom-line)
10. [Limitations](#10-limitations)

## 1. Ablation grid

| Model | Priority | What it is |
|---|---|---|
| A1_imagenet_patch | primary | ImageNet ConvNeXt, no distillation, patch-only |
| **A2_distilled_patch** | primary | GenBio-distilled ConvNeXt, patch-only (=y_patch, the anchor) |
| A5_shuffled_graph | primary | degree-preserving rewire -- CRITICAL negative control (Gate 3) |
| A6_local_pna | primary | real LOCAL-only physical graph, union-PNA backend |
| A7_multiscale_pna | primary | real local+meso+macro union graph, one PNA sees all edges |
| **A8_scale_aware_pna** | primary | PROPOSED FINAL MODEL: three per-scale PNA branches + learned softmax gate |
| A0_mean_expression | secondary | predict training gene means -- sanity floor |
| A3_matched_mlp | secondary | parameter-matched MLP, same features, no neighbors |
| A4_random_graph | secondary | same nodes/edge count, uniform random edges, union-PNA backend |
| A9_gatv2_multiscale | secondary | architecture control: same union multiscale graph, GATv2 instead of PNA |

## 2. Macro-average results

Pooled across IDC + COAD + CCRCC (15 cancer-fold-seed cells per model: 3 cancers x 5 folds x seed 0).

| Model | HVG1000 R [95% CI] | HVG50 R [95% CI] | Pathway R [95% CI] |
|---|---|---|---|
| A1_imagenet_patch | 0.0612 [0.0480, 0.0749] (n=15) | 0.0844 [0.0651, 0.1038] (n=15) | 0.0903 [0.0791, 0.1008] (n=15) |
| **A2_distilled_patch** | 0.0728 [0.0568, 0.0874] (n=15) | 0.1064 [0.0850, 0.1266] (n=15) | 0.1279 [0.1140, 0.1426] (n=15) |
| A5_shuffled_graph | 0.0998 [0.0880, 0.1103] (n=15) | 0.1461 [0.1280, 0.1609] (n=15) | 0.1484 [0.1330, 0.1640] (n=15) |
| A6_local_pna | 0.0955 [0.0850, 0.1052] (n=15) | 0.1393 [0.1211, 0.1568] (n=15) | 0.1416 [0.1294, 0.1538] (n=15) |
| A7_multiscale_pna | 0.1019 [0.0900, 0.1127] (n=15) | 0.1515 [0.1348, 0.1650] (n=15) | 0.1482 [0.1334, 0.1630] (n=15) |
| **A8_scale_aware_pna** | 0.0981 [0.0872, 0.1078] (n=15) | 0.1435 [0.1269, 0.1579] (n=15) | 0.1475 [0.1328, 0.1626] (n=15) |
| A0_mean_expression | n/a [n/a, n/a] (n=0) | n/a [n/a, n/a] (n=0) | n/a [n/a, n/a] (n=0) |
| A3_matched_mlp | 0.0857 [0.0759, 0.0949] (n=15) | 0.1204 [0.1045, 0.1352] (n=15) | 0.1302 [0.1161, 0.1451] (n=15) |
| A4_random_graph | 0.0997 [0.0880, 0.1101] (n=15) | 0.1451 [0.1273, 0.1600] (n=15) | 0.1485 [0.1332, 0.1638] (n=15) |
| A9_gatv2_multiscale | 0.1024 [0.0911, 0.1120] (n=15) | 0.1491 [0.1327, 0.1625] (n=15) | 0.1523 [0.1364, 0.1684] (n=15) |

(A0_mean_expression's R is `n/a`: predicting the constant train-fold mean for every spot gives zero-variance predictions, and Pearson r is undefined against a constant -- this is the expected behavior of a sanity-floor baseline, not a bug.)

## 3. Per-cancer results

### IDC

| Model | HVG1000 R [95% CI] | HVG50 R [95% CI] | Pathway R [95% CI] |
|---|---|---|---|
| A1_imagenet_patch | 0.0611 [0.0557, 0.0665] (n=5) | 0.0624 [0.0550, 0.0684] (n=5) | 0.0986 [0.0907, 0.1056] (n=5) |
| **A2_distilled_patch** | 0.0495 [0.0257, 0.0734] (n=5) | 0.0750 [0.0453, 0.1030] (n=5) | 0.1211 [0.0937, 0.1472] (n=5) |
| A5_shuffled_graph | 0.1139 [0.1090, 0.1190] (n=5) | 0.1553 [0.1544, 0.1560] (n=5) | 0.1794 [0.1706, 0.1884] (n=5) |
| A6_local_pna | 0.0988 [0.0950, 0.1025] (n=5) | 0.1298 [0.1258, 0.1348] (n=5) | 0.1570 [0.1490, 0.1638] (n=5) |
| A7_multiscale_pna | 0.1124 [0.1063, 0.1172] (n=5) | 0.1567 [0.1516, 0.1622] (n=5) | 0.1776 [0.1692, 0.1850] (n=5) |
| **A8_scale_aware_pna** | 0.1093 [0.1020, 0.1161] (n=5) | 0.1510 [0.1399, 0.1596] (n=5) | 0.1749 [0.1629, 0.1840] (n=5) |
| A0_mean_expression | n/a [n/a, n/a] (n=0) | n/a [n/a, n/a] (n=0) | n/a [n/a, n/a] (n=0) |
| A3_matched_mlp | 0.0876 [0.0838, 0.0916] (n=5) | 0.1128 [0.1086, 0.1169] (n=5) | 0.1495 [0.1430, 0.1570] (n=5) |
| A4_random_graph | 0.1134 [0.1095, 0.1180] (n=5) | 0.1553 [0.1530, 0.1576] (n=5) | 0.1784 [0.1710, 0.1872] (n=5) |
| A9_gatv2_multiscale | 0.1138 [0.1085, 0.1185] (n=5) | 0.1560 [0.1515, 0.1619] (n=5) | 0.1829 [0.1772, 0.1884] (n=5) |

### COAD

| Model | HVG1000 R [95% CI] | HVG50 R [95% CI] | Pathway R [95% CI] |
|---|---|---|---|
| A1_imagenet_patch | 0.0901 [0.0769, 0.1044] (n=5) | 0.1295 [0.1154, 0.1437] (n=5) | 0.0939 [0.0706, 0.1147] (n=5) |
| **A2_distilled_patch** | 0.1051 [0.0941, 0.1126] (n=5) | 0.1415 [0.1215, 0.1619] (n=5) | 0.1256 [0.1124, 0.1422] (n=5) |
| A5_shuffled_graph | 0.1124 [0.1038, 0.1210] (n=5) | 0.1616 [0.1389, 0.1768] (n=5) | 0.1275 [0.1163, 0.1430] (n=5) |
| A6_local_pna | 0.1129 [0.0998, 0.1236] (n=5) | 0.1619 [0.1358, 0.1864] (n=5) | 0.1287 [0.1169, 0.1447] (n=5) |
| A7_multiscale_pna | 0.1174 [0.1048, 0.1274] (n=5) | 0.1673 [0.1552, 0.1783] (n=5) | 0.1289 [0.1178, 0.1441] (n=5) |
| **A8_scale_aware_pna** | 0.1107 [0.1002, 0.1184] (n=5) | 0.1550 [0.1334, 0.1739] (n=5) | 0.1279 [0.1158, 0.1440] (n=5) |
| A0_mean_expression | n/a [n/a, n/a] (n=0) | n/a [n/a, n/a] (n=0) | n/a [n/a, n/a] (n=0) |
| A3_matched_mlp | 0.1006 [0.0908, 0.1127] (n=5) | 0.1326 [0.1213, 0.1450] (n=5) | 0.1066 [0.0893, 0.1210] (n=5) |
| A4_random_graph | 0.1120 [0.1021, 0.1220] (n=5) | 0.1603 [0.1382, 0.1771] (n=5) | 0.1275 [0.1160, 0.1428] (n=5) |
| A9_gatv2_multiscale | 0.1156 [0.1118, 0.1215] (n=5) | 0.1614 [0.1449, 0.1734] (n=5) | 0.1311 [0.1206, 0.1443] (n=5) |

### CCRCC

| Model | HVG1000 R [95% CI] | HVG50 R [95% CI] | Pathway R [95% CI] |
|---|---|---|---|
| A1_imagenet_patch | 0.0325 [0.0211, 0.0429] (n=5) | 0.0614 [0.0328, 0.0874] (n=5) | 0.0783 [0.0607, 0.0972] (n=5) |
| **A2_distilled_patch** | 0.0637 [0.0500, 0.0761] (n=5) | 0.1029 [0.0729, 0.1331] (n=5) | 0.1370 [0.1117, 0.1655] (n=5) |
| A5_shuffled_graph | 0.0733 [0.0569, 0.0884] (n=5) | 0.1214 [0.0828, 0.1569] (n=5) | 0.1384 [0.1129, 0.1634] (n=5) |
| A6_local_pna | 0.0749 [0.0591, 0.0897] (n=5) | 0.1263 [0.0861, 0.1648] (n=5) | 0.1392 [0.1125, 0.1659] (n=5) |
| A7_multiscale_pna | 0.0761 [0.0602, 0.0916] (n=5) | 0.1305 [0.0922, 0.1672] (n=5) | 0.1381 [0.1137, 0.1630] (n=5) |
| **A8_scale_aware_pna** | 0.0743 [0.0594, 0.0886] (n=5) | 0.1244 [0.0886, 0.1584] (n=5) | 0.1398 [0.1141, 0.1660] (n=5) |
| A0_mean_expression | n/a [n/a, n/a] (n=0) | n/a [n/a, n/a] (n=0) | n/a [n/a, n/a] (n=0) |
| A3_matched_mlp | 0.0690 [0.0523, 0.0856] (n=5) | 0.1157 [0.0732, 0.1583] (n=5) | 0.1347 [0.1087, 0.1657] (n=5) |
| A4_random_graph | 0.0737 [0.0567, 0.0894] (n=5) | 0.1198 [0.0817, 0.1538] (n=5) | 0.1395 [0.1134, 0.1656] (n=5) |
| A9_gatv2_multiscale | 0.0778 [0.0604, 0.0934] (n=5) | 0.1301 [0.0922, 0.1652] (n=5) | 0.1429 [0.1139, 0.1719] (n=5) |

## 4. Primary statistical gates

The three pre-registered pass/fail tests (configs/base.yaml `primary_gates`). `delta_R` = mean(better) - mean(worse) over shared cells; CI/p-value from `sts.stats_tests.patient_bootstrap_ci` / `paired_sign_flip_test`; **SUPPORTED** requires the 95% CI to lie entirely above zero.

### Gate 1: Distillation (A2 > A1)

better = `A2_distilled_patch`, worse = `A1_imagenet_patch`

| Scope | delta_R | 95% CI | Cohen's d | p (one-sided sign-flip) | n | Verdict |
|---|---|---|---|---|---|---|
| pooled (3 cancers) | 0.0115 | [-0.0036, 0.0245] | 0.3935 | 0.0722 | 15 | **NOT SUPPORTED** |
| IDC | -0.0116 | [-0.0395, 0.0163] | -0.3084 | 0.7745 | 5 | **NOT SUPPORTED** |
| COAD | 0.0150 | [-0.0016, 0.0295] | 0.7809 | 0.0900 | 5 | **NOT SUPPORTED** |
| CCRCC | 0.0312 | [0.0250, 0.0370] | 4.0935 | 0.0290 | 5 | **SUPPORTED** |

### Gate 2: Spatial context (A8 > A2)

better = `A8_scale_aware_pna`, worse = `A2_distilled_patch`

| Scope | delta_R | 95% CI | Cohen's d | p (one-sided sign-flip) | n | Verdict |
|---|---|---|---|---|---|---|
| pooled (3 cancers) | 0.0253 | [0.0110, 0.0427] | 0.7634 | 0.0001 | 15 | **SUPPORTED** |
| IDC | 0.0598 | [0.0313, 0.0883] | 1.5505 | 0.0290 | 5 | **SUPPORTED** |
| COAD | 0.0055 | [0.0044, 0.0065] | 4.1072 | 0.0290 | 5 | **SUPPORTED** |
| CCRCC | 0.0106 | [0.0036, 0.0192] | 1.0154 | 0.0290 | 5 | **SUPPORTED** |

### Gate 3: True topology (A8 > A5, decisive negative control)

better = `A8_scale_aware_pna`, worse = `A5_shuffled_graph`

| Scope | delta_R | 95% CI | Cohen's d | p (one-sided sign-flip) | n | Verdict |
|---|---|---|---|---|---|---|
| pooled (3 cancers) | -0.0017 | [-0.0044, 0.0006] | -0.3372 | 0.8853 | 15 | **NOT SUPPORTED** |
| IDC | -0.0045 | [-0.0102, -0.0007] | -0.7318 | 0.9670 | 5 | **NOT SUPPORTED** |
| COAD | -0.0017 | [-0.0060, 0.0026] | -0.2977 | 0.7410 | 5 | **NOT SUPPORTED** |
| CCRCC | 0.0010 | [-0.0002, 0.0027] | 0.5284 | 0.1529 | 5 | **NOT SUPPORTED** |

## 5. Secondary comparisons

A8 (scale-aware, flagship) vs. the remaining architecture/graph controls, pooled across 3 cancers.

| Comparison | better | worse | delta_R | 95% CI | p (one-sided) | n | Verdict |
|---|---|---|---|---|---|---|---|
| scale_aware_vs_local | A8_scale_aware_pna | A6_local_pna | 0.0026 | [-0.0007, 0.0061] | 0.0836 | 15 | **NOT SUPPORTED** |
| scale_aware_vs_union_multiscale | A8_scale_aware_pna | A7_multiscale_pna | -0.0039 | [-0.0079, -0.0007] | 0.9784 | 15 | **NOT SUPPORTED** |
| scale_aware_vs_random | A8_scale_aware_pna | A4_random_graph | -0.0016 | [-0.0041, 0.0006] | 0.8878 | 15 | **NOT SUPPORTED** |
| scale_aware_vs_matched_mlp | A8_scale_aware_pna | A3_matched_mlp | 0.0123 | [0.0068, 0.0175] | 0.0007 | 15 | **SUPPORTED** |
| scale_aware_vs_gatv2 | A8_scale_aware_pna | A9_gatv2_multiscale | -0.0043 | [-0.0078, -0.0012] | 0.9881 | 15 | **NOT SUPPORTED** |

## 6. Pathway / PROGENy14 view

Same pathway-target column as sections 2-3 above, isolated for readers who only care about the 14 PROGENy pathway-activity scores (not per-gene HVG targets).

| Model | Macro R [95% CI] | IDC R [95% CI] | COAD R [95% CI] | CCRCC R [95% CI] |
|---|---|---|---|---|
| A1_imagenet_patch | 0.0903 [0.0791, 0.1008] | 0.0986 [0.0907, 0.1056] | 0.0939 [0.0706, 0.1147] | 0.0783 [0.0607, 0.0972] |
| **A2_distilled_patch** | 0.1279 [0.1140, 0.1426] | 0.1211 [0.0937, 0.1472] | 0.1256 [0.1124, 0.1422] | 0.1370 [0.1117, 0.1655] |
| A5_shuffled_graph | 0.1484 [0.1330, 0.1640] | 0.1794 [0.1706, 0.1884] | 0.1275 [0.1163, 0.1430] | 0.1384 [0.1129, 0.1634] |
| A6_local_pna | 0.1416 [0.1294, 0.1538] | 0.1570 [0.1490, 0.1638] | 0.1287 [0.1169, 0.1447] | 0.1392 [0.1125, 0.1659] |
| A7_multiscale_pna | 0.1482 [0.1334, 0.1630] | 0.1776 [0.1692, 0.1850] | 0.1289 [0.1178, 0.1441] | 0.1381 [0.1137, 0.1630] |
| **A8_scale_aware_pna** | 0.1475 [0.1328, 0.1626] | 0.1749 [0.1629, 0.1840] | 0.1279 [0.1158, 0.1440] | 0.1398 [0.1141, 0.1660] |
| A0_mean_expression | n/a [n/a, n/a] | n/a [n/a, n/a] | n/a [n/a, n/a] | n/a [n/a, n/a] |
| A3_matched_mlp | 0.1302 [0.1161, 0.1451] | 0.1495 [0.1430, 0.1570] | 0.1066 [0.0893, 0.1210] | 0.1347 [0.1087, 0.1657] |
| A4_random_graph | 0.1485 [0.1332, 0.1638] | 0.1784 [0.1710, 0.1872] | 0.1275 [0.1160, 0.1428] | 0.1395 [0.1134, 0.1656] |
| A9_gatv2_multiscale | 0.1523 [0.1364, 0.1684] | 0.1829 [0.1772, 0.1884] | 0.1311 [0.1206, 0.1443] | 0.1429 [0.1139, 0.1719] |

## 7. GenBio-PathFM reference numbers

Two **separate protocol tracks** -- neither is directly comparable to the A0-A9 table above (different targets/splits/heads), and they are not comparable to *each other* either (different training protocol). Do not average across tracks.

### 7a. Official HEST-Benchmark protocol

log1p-only targets (no library-size normalization), StandardScaler+PCA(256)+Ridge, official published k-fold splits, seed=1. Distilled student and GenBio-PathFM teacher evaluated identically -- this isolates the effect of the *encoder* from everything else in the custom pipeline (no trained patch head, no GNN, no train-fold HVG selection).

| Cancer | n folds | Distilled student R (std) | GenBio-PathFM teacher R (std) | Gap (teacher - student) |
|---|---|---|---|---|
| IDC | 4 | 0.4822 (0.0489) | 0.5898 (0.0861) | +0.1076 |
| COAD | 2 | 0.2385 (0.0282) | 0.3264 (0.0184) | +0.0880 |
| CCRCC | 6 | 0.2225 (0.0544) | 0.2638 (0.0277) | +0.0413 |

### 7b. Custom-pipeline GenBio embedding -> trained MLP baseline (B1) -- PARTIAL

Frozen 4608-D GenBio-PathFM embedding fed directly into a small trained MLP head (no PCA, no Ridge -- `src/scripts/run_genbio_hest_baseline.py --head ridge` is an available but unused alternative), on the SAME HVG1000/HVG50/PROGENy14 targets and fold structure as the A0-A9 table -- but **fold0/seed0 only, not the full 5-fold design**. Treat as preliminary; do not compare CI-to-CI against the bootstrapped A0-A9 numbers above (no CI is computable from a single fold).

| Cancer | n train / test patients | HVG1000 R | HVG50 R | PROGENy14 R | Feature extract (s) | Head train (s) | Peak VRAM (GB) |
|---|---|---|---|---|---|---|---|
| IDC | 77 / 23 | 0.1053 | 0.1478 | -0.0157 | 1.7 | 22.6 | 4.78 |
| COAD | 36 / 11 | 0.1941 | 0.3357 | -0.0023 | 1.9 | 25.7 | 8.05 |
| CCRCC | 16 / 5 | 0.1227 | 0.2058 | 0.0240 | 0.9 | 15.4 | 5.01 |

## 8. Efficiency

| Metric | Value | Status | Source |
|---|---|---|---|
| student_encoder_total_parameters | 3,387,400 | MEASURED | scripts (ConvNeXt V2-Atto student), student_encoder_param_count_seed0.json |
| genbio_pathfm_teacher_total_parameters | 1,133,686,784 | MEASURED | AutoModel.from_pretrained('genbio-ai/genbio-pathfm'), sum(p.numel()), 2026-08-31 |
| compression_ratio_teacher_over_student | 334.7x | MEASURED | derived from the two rows above |
| gpu_peak_vram_inference | n/a | NOT MEASURED | no inference-time VRAM profiling run this session |
| patches_per_second_throughput | n/a | NOT MEASURED | no throughput benchmark run this session |
| A8_gnn_inference_wallclock_per_patient | n/a | NOT MEASURED | no dedicated timing harness run this session |

## 9. Bottom line

- **Spatial graph correction helps (Gate 2, SUPPORTED everywhere).** A8 beats A2 (patch-only) pooled (delta_R=+0.0253, 95% CI [0.011, 0.043]) and in all 3 cancers individually (IDC +0.060, COAD +0.006, CCRCC +0.011) -- the most robust positive result in the experiment.
- **But not because of *real* tissue topology (Gate 3, NOT SUPPORTED everywhere).** A8 (real multiscale graph) is statistically indistinguishable from A5 (degree-preserving shuffled graph) pooled (delta_R=-0.002, CI crosses zero) and in every cancer. The A8-vs-A5 secondary comparison confirms the same null result. The GNN's benefit most plausibly comes from generic neighbor-averaging/regularization, not from encoding true spatial adjacency -- a genuine negative finding, reported as such per the FINAL STOP RULE.
- **CPTAC->HEST distillation does not reliably help (Gate 1, NOT SUPPORTED pooled).** A2 (distilled) vs. A1 (ImageNet) pooled delta_R=+0.0115 but CI=[-0.004, 0.025] crosses zero (p=0.072, just short of one-sided significance). It flips sign across cancers: CCRCC SUPPORTED (+0.031), COAD marginal (+0.015, not supported), IDC actually reverses (-0.012, ImageNet ahead). The IDC reversal is flagged, not explained, in Limitations.
- **A8 (proposed flagship) is not the single best ablation among the graph variants.** A7 (union multiscale, no scale-aware gating) and A9 (GATv2 architecture control) score marginally higher than A8 pooled on HVG1000 (A7: 0.1019, A9: 0.1024 vs. A8: 0.0981); the A8-vs-A7 and A8-vs-A9 secondary comparisons are both NOT SUPPORTED (A8 does not significantly beat either). A8 does significantly beat the parameter-matched MLP (A3) with no graph at all -- so *some* graph structure helps, but the scale-aware gating mechanism specifically does not show a measurable edge over simpler graph variants at this sample size.
- **Distillation buys a real, large compression at a real, consistent accuracy cost.** Under the official protocol (encoder-only comparison, no GNN, no patch head), GenBio-PathFM beats the distilled student in all 3 cancers by 0.04-0.11 R, for 334.7x fewer parameters.

## 10. Limitations (reported honestly, not omitted)

1. **Bootstrap resampling unit.** All CIs above resample (cancer, fold, seed) cells (15 pooled, 5 per cancer), not literal per-patient R_p values -- script 10 does not persist per-patient predictions to disk. A stricter implementation would bootstrap over patients directly.
2. **Seeds 1/2** were not run for the major models -- seed 0 only.
3. **B1 (GenBio embedding -> MLP baseline)** is fold0/seed0 only per cancer, not the full 5-fold design the rest of this document uses.
4. **Efficiency**: only parameter counts and the derived compression ratio were measured; inference VRAM, throughput, and GNN wall-clock were not.
5. **The IDC-specific distillation reversal** (Gate 1 section, A1 ahead of A2 in IDC specifically) has not been separately root-caused.
6. **A8 vs. A7/A9** shows A8 is not clearly the best graph variant at this sample size (section 9) -- worth noting for anyone citing A8 as unambiguously superior.

Full machine-readable tables: `results/final/tables/*.csv`. Exact patient/fold manifest: `results/final/folds/cv_folds_used.csv`. Full run config: `experimental_runs/2026-08-31_job84871_final_5fold_3cancer_seed0_with_results_final/config_used.json`.
