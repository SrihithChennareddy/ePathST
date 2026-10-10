# Numbers sheet

Generated from commit `HEAD 15077ed + uncommitted Steps 5-10`.

## Headline numbers

| Number | Value | File | Script |
|---|---|---|---|
| Primary test: BRCA C+P vs C, delta Uno's C | 0.0163 [-0.028, 0.060], p=0.460 | `results/tcga_final/survival/delta_c_primary.csv` | `src/scripts/54_tcga_survival.py` |
| Pooled (conditional) BRCA C+P vs C | 0.0212 [-0.007, 0.053] | `results/tcga_final/survival/pooled_delta_c.csv` | `src/scripts/54b_pooled_ci.py` |
| Secondary: KIRC C+P vs C | 0.0197 [0.004, 0.035], p=0.0136, q=0.231 | `results/tcga_final/survival/delta_c_secondary.csv` | `src/scripts/54_tcga_survival.py` |
| Secondary: KIRC C+B vs C | 0.0255 [0.003, 0.048], p=0.0271, q=0.231 | `results/tcga_final/survival/delta_c_secondary.csv` | `src/scripts/54_tcga_survival.py` |
| Independence HR, KIRC (P score, per SD) | 2.54 [1.64, 3.94], p=3e-5 | `results/tcga_final/survival/independence.csv` | `src/scripts/54_tcga_survival.py` |
| U3 retention ratio R_student, KIRC | 0.650 [0.167, 1.095] | `results/tcga_final/survival/retention_ratio_ci.csv` | `src/scripts/54b_pooled_ci.py` |
| U3 retention ratio R_teacher, KIRC | 1.182 [0.234, 2.307] | `results/tcga_final/survival/retention_ratio_ci.csv` | `src/scripts/54b_pooled_ci.py` |
| HEST main table, 3-cancer mean (P, 2-seed) | 0.341 (all folds) | `results/tcga_final/hest/hest_main.csv` | `src/scripts/57_hest_tables.py` |
| Compute: student MACs/patch | 0.552 GMACs | `results/tcga_final/hest/compute.csv` | `src/scripts/57_hest_tables.py` |
| Compute: H-Optimus-0 MACs/patch | 296.0 GMACs (536x student) | `results/tcga_final/hest/compute.csv` | `src/scripts/57_hest_tables.py` |
| Compute: ResNet-50 MACs/patch | 4.112 GMACs (7.45x student) | `results/tcga_final/hest/compute.csv` | `src/scripts/57_hest_tables.py` |
| ePathST-E linear-probe (own run) | 0.3144 (IDC 0.482/COAD 0.238/CCRCC 0.223) | `results/tcga_final/hest/provenance_search.json` | `src/scripts/57_hest_tables.py` |
| BRCA cohort: candidate/eligible/survival-eligible/events | 1045/1030/994/130 | `results/tcga_final/cohort/consort_BRCA.csv` | `src/scripts/51_tcga_cohort.py` |
| Screen: BRCA genes q<0.05 (P) | 31/50 | `results/tcga_final/screen/screen_full.csv` | `src/scripts/53_tcga_screen.py` |
| Specificity: BRCA median matched rank (P) | 9.0 (null 25.5) | `results/tcga_final/screen/specificity_summary.csv` | `src/scripts/53_tcga_screen.py` |
| U4 pooled permutation, O1_T | p=0.011, q=0.050 | `results/tcga_final/programs/program_tests.csv` | `src/scripts/55_program_transfer.py` |
| Sensitivity: BRCA IDC-only, C+P vs C | 0.0087 [-0.044, 0.062], p=0.744, q=0.934 (primary stays null) | `results/tcga_final/survival_sensitivity/delta_c_sensitivity.csv` | `src/scripts/54c_sensitivity.py` |
| Sensitivity: median aggregation, KIRC C+B vs C | 0.0255 [0.003, 0.048], p=0.027, q=0.934 (matches main analysis, doesn't survive correction) | `results/tcga_final/survival_sensitivity/delta_c_sensitivity.csv` | `src/scripts/54c_sensitivity.py` |
| Sensitivity: spots-covariate, COAD C+P vs C+E | 0.0263 [0.001, 0.052], p=0.044, q=0.934 (new nominal signal, doesn't survive correction) | `results/tcga_final/survival_sensitivity/delta_c_sensitivity.csv` | `src/scripts/54c_sensitivity.py` |

## Full table/figure provenance map

Every number in every main table and every figure panel traces to one of these files. Each file's own rows/columns are the per-number index (so, e.g., 'screen_full.csv' is the row/column answer for all 450 of its own numbers, not spelled out cell by cell here).

| Table/Figure | Panel/column | File | Script |
|---|---|---|---|
| Table I | cohorts (candidate/eligible/survival-eligible/events/followup), 3 cancers x 6 cols | `figures/table1_cohorts.csv` | `51_tcga_cohort.py` -> `58_figures.py` |
| Table II | HEST same-head, 3 cancers x 3 systems (+3-cancer mean), seed0/seed1/2-seed mean, all-folds/excl-collapsed | `figures/table2_hest_same_head.csv` (= `hest/hest_main.csv`) | `57_hest_tables.py` |
| Table III | C-index, 3 cancers x 2 cohorts x up to 8 feature sets, mean+SD | `figures/table3_cindex.csv` (= `survival/cindex_by_split.csv` aggregated) | `54_tcga_survival.py` |
| Table III (delta C) | all 18 primary+secondary comparisons, corrected CI, p, q | `figures/table3b_delta_c.csv` (= `survival/delta_c.csv`) | `54_tcga_survival.py` |
| Table III (U3) | 12 bottleneck comparisons, corrected CI, p, q | `figures/table3c_u3.csv` (= `survival/bottleneck_summary.csv`) | `54_tcga_survival.py` |
| Fig 2a | HEST same-head bars, 3 cancers x 3 systems x 2 seeds | `hest/hest_main.csv` | `57_hest_tables.py` |
| Fig 2b | TCGA screen median rho, 3 cancers x 3 systems (9 numbers, derived: median of `rho` per cancer x system) | `screen/screen_full.csv` | `53_tcga_screen.py` |
| Fig 2c | U3 bottleneck, 4 comparisons x 3 cancers, corrected CI | `survival/bottleneck_summary.csv` | `54_tcga_survival.py` |
| Fig 3a | P-E per-gene PCC on HEST, 3 cancers x 2 seeds (derived: `pcc_mean_all_folds_seed{0,1}` for P minus E) | `hest/hest_main.csv` | `57_hest_tables.py` |
| Fig 3b | per-gene P-E on TCGA screen, BRCA, 50 genes with CI | `screen/system_deltas.csv` (comparison=P-E, cancer=BRCA) | `53_tcga_screen.py` |
| Fig 4a | per-gene rho + partial rho + CI, BRCA, 50 genes x 3 systems (150 numbers) | `screen/screen_full.csv` (cancer=BRCA) | `53_tcga_screen.py` |
| Fig 4b | O2 (P) by program type, 3 cancers x 3 programs (box five-number summaries) | `programs/gene_level_outcomes.csv` (column O2_P, grouped by cancer/program) | `55_program_transfer.py` |
| Fig 4c | AQP1 scatter, BRCA, 994 patients (predicted vs measured) + rho annotation | `features/BRCA_P_mean.parquet`, `features/BRCA_B.parquet`, `screen/screen_full.csv` (gene=AQP1) | `52_tcga_features.py`, `53_tcga_screen.py` |
| Fig 5a | survival forest, all 18 primary+secondary comparisons | `survival/delta_c.csv` | `54_tcga_survival.py` |
| Fig 5b | BRCA KM, C and C+P panels, repeat 0, numbers at risk in the legend | `survival_exploratory/km_summary.csv` + `survival_exploratory/km_BRCA.png` | `56_survival_exploratory.py` |

## Step 5.7 sensitivity analyses (supplement)

| Analysis | File | Script |
|---|---|---|
| BRCA IDC-only (primary + secondary set) | `survival_sensitivity/delta_c_sensitivity.csv` (variant=idc_only) | `54c_sensitivity.py` |
| Cancer-specific covariates (BRCA ER+HER2, KIRC grade) | `survival_sensitivity/delta_c_sensitivity.csv` (variant=cancer_covariates) | `54c_sensitivity.py` |
| Median instead of mean aggregation | `survival_sensitivity/delta_c_sensitivity.csv` (variant=median_aggregation) | `54c_sensitivity.py` |
| Spots-per-patient as covariate | `survival_sensitivity/delta_c_sensitivity.csv` (variant=spots_covariate) | `54c_sensitivity.py` |

**Scope note:** figure/table panels that are themselves large tables (e.g. Fig 4a's 150 numbers) are mapped to their one source file rather than enumerated cell by cell here -- that file's own rows/columns are the per-number index. Sensitivity analyses added 2026-10-10; AUC/integrated-Brier secondary survival metrics and per-gene sensitivity tables beyond this list are not yet computed.