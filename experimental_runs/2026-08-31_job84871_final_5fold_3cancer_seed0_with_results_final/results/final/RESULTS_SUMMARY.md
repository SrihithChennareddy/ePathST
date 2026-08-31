# Results Summary

Quick-reference version of `FINAL_RESULTS.md` (job 84871, LOCKED final 5-fold x 3-cancer x seed-0 run, 2026-08-31). Numbers are patient-level Fisher-z mean Pearson R, pooled over 5 folds (seed 0). All figures read directly from `results/final/tables/`, nothing re-derived here.

## Macro average (pooled across IDC + COAD + CCRCC)

| Model | HVG1000 R | HVG50 R | Pathway (PROGENy14) R |
|---|---|---|---|
| A1_imagenet_patch | 0.0612 | 0.0844 | 0.0903 |
| **A2_distilled_patch** | 0.0728 | 0.1064 | 0.1279 |
| A5_shuffled_graph | 0.0998 | 0.1461 | 0.1484 |
| A6_local_pna | 0.0955 | 0.1393 | 0.1416 |
| A7_multiscale_pna | 0.1019 | 0.1515 | 0.1482 |
| **A8_scale_aware_pna** | 0.0981 | 0.1435 | 0.1475 |
| A0_mean_expression | nan | nan | nan |
| A3_matched_mlp | 0.0857 | 0.1204 | 0.1302 |
| A4_random_graph | 0.0997 | 0.1451 | 0.1485 |
| A9_gatv2_multiscale | 0.1024 | 0.1491 | 0.1523 |

**A2_distilled_patch** = the patch-only anchor (no graph). **A8_scale_aware_pna** = the proposed flagship model (multiscale PNA graph correction).

## Per-cancer averages, A0-A9 (5-fold mean, seed 0)

### IDC

| Model | HVG1000 R | HVG50 R | Pathway R |
|---|---|---|---|
| A1_imagenet_patch | 0.0611 | 0.0624 | 0.0986 |
| **A2_distilled_patch** | 0.0495 | 0.0750 | 0.1211 |
| A5_shuffled_graph | 0.1139 | 0.1553 | 0.1794 |
| A6_local_pna | 0.0988 | 0.1298 | 0.1570 |
| A7_multiscale_pna | 0.1124 | 0.1567 | 0.1776 |
| **A8_scale_aware_pna** | 0.1093 | 0.1510 | 0.1749 |
| A0_mean_expression | nan | nan | nan |
| A3_matched_mlp | 0.0876 | 0.1128 | 0.1495 |
| A4_random_graph | 0.1134 | 0.1553 | 0.1784 |
| A9_gatv2_multiscale | 0.1138 | 0.1560 | 0.1829 |

### COAD

| Model | HVG1000 R | HVG50 R | Pathway R |
|---|---|---|---|
| A1_imagenet_patch | 0.0901 | 0.1295 | 0.0939 |
| **A2_distilled_patch** | 0.1051 | 0.1415 | 0.1256 |
| A5_shuffled_graph | 0.1124 | 0.1616 | 0.1275 |
| A6_local_pna | 0.1129 | 0.1619 | 0.1287 |
| A7_multiscale_pna | 0.1174 | 0.1673 | 0.1289 |
| **A8_scale_aware_pna** | 0.1107 | 0.1550 | 0.1279 |
| A0_mean_expression | nan | nan | nan |
| A3_matched_mlp | 0.1006 | 0.1326 | 0.1066 |
| A4_random_graph | 0.1120 | 0.1603 | 0.1275 |
| A9_gatv2_multiscale | 0.1156 | 0.1614 | 0.1311 |

### CCRCC

| Model | HVG1000 R | HVG50 R | Pathway R |
|---|---|---|---|
| A1_imagenet_patch | 0.0325 | 0.0614 | 0.0783 |
| **A2_distilled_patch** | 0.0637 | 0.1029 | 0.1370 |
| A5_shuffled_graph | 0.0733 | 0.1214 | 0.1384 |
| A6_local_pna | 0.0749 | 0.1263 | 0.1392 |
| A7_multiscale_pna | 0.0761 | 0.1305 | 0.1381 |
| **A8_scale_aware_pna** | 0.0743 | 0.1244 | 0.1398 |
| A0_mean_expression | nan | nan | nan |
| A3_matched_mlp | 0.0690 | 0.1157 | 0.1347 |
| A4_random_graph | 0.0737 | 0.1198 | 0.1395 |
| A9_gatv2_multiscale | 0.0778 | 0.1301 | 0.1429 |

## GenBio-PathFM reference numbers (two separate tracks -- do not average with the A0-A9 table above)

**(1) Official HEST-Benchmark protocol** (log1p-only targets, PCA256+Ridge, official k-fold splits, seed=1) -- distilled student vs. its own teacher, same protocol for both:

| Cancer | Distilled student R | GenBio-PathFM teacher R | Gap |
|---|---|---|---|
| IDC | 0.4822 | 0.5898 | +0.1076 |
| COAD | 0.2385 | 0.3264 | +0.0880 |
| CCRCC | 0.2225 | 0.2638 | +0.0413 |

**(2) Custom-pipeline GenBio-PCA256-Ridge baseline (B1)** -- fold0/seed0 only, **PARTIAL**, not the full 5-fold design the A0-A9 table above uses:

| Cancer | HVG1000 R | HVG50 R | PROGENy14 R |
|---|---|---|---|
| IDC | 0.1053 | 0.1478 | -0.0157 |
| COAD | 0.1941 | 0.3357 | -0.0023 |
| CCRCC | 0.1227 | 0.2058 | 0.0240 |

## Bottom line

- The spatial graph correction (A8 vs. A2) helps in all 3 cancers (Gate 2: SUPPORTED pooled and per-cancer).
- That benefit does not require *real* tissue topology -- A8 is statistically indistinguishable from A5 (shuffled-graph control) everywhere (Gate 3: NOT SUPPORTED).
- CPTAC->HEST distillation (A2 vs. A1) is not reliably better pooled (Gate 1: NOT SUPPORTED); it wins in CCRCC, loses in IDC, is roughly a wash in COAD.
- Under the official HEST-Benchmark protocol, GenBio-PathFM (1.13B params) beats the distilled student (3.39M params, 334.7x smaller) by a real, consistent margin in all 3 cancers -- the accuracy cost of the 334.7x compression.

Full detail, per-cancer gate statistics, CIs, and limitations: `FINAL_RESULTS.md` (same directory).
