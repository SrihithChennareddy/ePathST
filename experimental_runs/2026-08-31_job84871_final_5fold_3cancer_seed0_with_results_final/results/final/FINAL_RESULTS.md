# FINAL_RESULTS.md

Locked final scientific experiment: distilled ConvNeXt V2-Atto (from GenBio-PathFM) +
multiscale PNA graph correction, 5-fold patient-level CV x 3 HEST-1k cancers
(IDC/COAD/CCRCC) x seed 0. Source run: SLURM job 84871, completed 2026-08-31
(03:15:54 elapsed). All numbers below are read directly from
`results/cv_ablation_metrics.csv`, `results/primary_hypothesis_test.json`,
`results/secondary_hypothesis_tests.json`, `results/official_hest_benchmark/`,
`results/genbio_baseline/`, and the two parameter-count JSONs -- nothing here is
estimated or extrapolated (FINAL STOP RULE).

## Research questions, answered

**Q1/Q2 -- Does GenBio-PathFM distillation help the patch-only encoder?**
**Gate 1: Distillation (A2 > A1)** -- pooled: **NOT SUPPORTED**, delta_R=0.0115 95% CI=[-0.0036, 0.0245], Cohen's d=0.3935, p=0.0722 (n=15 cancer-fold-seed cells). Per-cancer: IDC: NOT SUPPORTED (delta=-0.0116); COAD: NOT SUPPORTED (delta=0.0150); CCRCC: SUPPORTED (delta=0.0312).
Distillation is **not supported** as a pooled effect at seed 0: A2 (distilled) is not reliably
better than A1 (ImageNet) once patient-level bootstrap uncertainty is accounted for. See the
per-cancer breakdown in `tables/gate_summary.csv` for where (if anywhere) it reverses.

**Q3 -- Does the multiscale PNA graph correction add real signal over patch-only prediction?**
**Gate 2: Spatial context (A8 > A2)** -- pooled: **SUPPORTED**, delta_R=0.0253 95% CI=[0.0110, 0.0427], Cohen's d=0.7634, p=0.0001 (n=15 cancer-fold-seed cells). Per-cancer: IDC: SUPPORTED (delta=0.0598); COAD: SUPPORTED (delta=0.0055); CCRCC: SUPPORTED (delta=0.0106).
This is the most robust positive finding in the experiment: the GNN correction improves on
patch-only prediction in **every one of the 3 cancers individually**, not just pooled.

**Q4 -- Does that benefit come from true tissue topology specifically (vs. generic neighborhood
smoothing)?**
**Gate 3: True topology (A8 > A5, decisive negative control)** -- pooled: **NOT SUPPORTED**, delta_R=-0.0017 95% CI=[-0.0044, 0.0006], Cohen's d=-0.3372, p=0.8853 (n=15 cancer-fold-seed cells). Per-cancer: IDC: NOT SUPPORTED (delta=-0.0045); COAD: NOT SUPPORTED (delta=-0.0017); CCRCC: NOT SUPPORTED (delta=0.0010).
The decisive negative control: A8 (real multiscale topology) is statistically indistinguishable
from A5 (degree-preserving shuffled graph) in every cancer. Per the FINAL STOP RULE, this is
reported as a genuine negative finding, not glossed over -- the GNN's Gate-2 benefit most likely
comes from general neighbor-averaging/regularization, not from encoding real tissue adjacency.

**Q5 -- Compression efficiency (student vs. GenBio-PathFM teacher).**
Student (distilled ConvNeXt V2-Atto): **3,387,400 parameters** (measured).
GenBio-PathFM teacher: **1,133,686,784 parameters** (measured 2026-08-31 via
`AutoModel.from_pretrained`, exact `sum(p.numel())`, not the model card's rounded "1.1B").
Compression ratio: **334.7x** fewer parameters, at the accuracy
cost quantified by Gate 1 above and by the official-protocol comparison below. All other
efficiency metrics (peak inference VRAM, patches/sec throughput, per-patient GNN wall-clock) were
**NOT MEASURED** this session -- see `tables/efficiency.csv`.

## Primary gates (pooled across IDC/COAD/CCRCC, 15 cancer-fold-seed cells each)

| label                                                      |   mean_delta_R |   ci_low |   ci_high |   p_value_one_sided | verdict       |
|:-----------------------------------------------------------|---------------:|---------:|----------:|--------------------:|:--------------|
| Gate 1: Distillation (A2 > A1)                             |         0.0115 |  -0.0036 |    0.0245 |              0.0722 | NOT SUPPORTED |
| Gate 2: Spatial context (A8 > A2)                          |         0.0253 |   0.011  |    0.0427 |              0.0001 | SUPPORTED     |
| Gate 3: True topology (A8 > A5, decisive negative control) |        -0.0017 |  -0.0044 |    0.0006 |              0.8853 | NOT SUPPORTED |

Full pooled + per-cancer breakdown, sign-flip p-values, effect sizes: `tables/gate_summary.csv`.
Secondary comparisons (A8 vs. local/union/random/matched-MLP/GATv2): `tables/secondary_comparisons.csv`.

## Macro model comparison (pooled 3-cancer x 5-fold x seed-0, bootstrap 95% CI, n_bootstrap=10000)

| model               | priority   |   hvg1000_R_mean |   hvg1000_R_ci_low |   hvg1000_R_ci_high |   hvg50_R_mean |   pathway_R_mean |
|:--------------------|:-----------|-----------------:|-------------------:|--------------------:|---------------:|-----------------:|
| A1_imagenet_patch   | primary    |           0.0612 |             0.048  |              0.0749 |         0.0844 |           0.0903 |
| A2_distilled_patch  | primary    |           0.0728 |             0.0568 |              0.0874 |         0.1064 |           0.1279 |
| A5_shuffled_graph   | primary    |           0.0998 |             0.088  |              0.1103 |         0.1461 |           0.1484 |
| A6_local_pna        | primary    |           0.0955 |             0.085  |              0.1052 |         0.1393 |           0.1416 |
| A7_multiscale_pna   | primary    |           0.1019 |             0.09   |              0.1127 |         0.1515 |           0.1482 |
| A8_scale_aware_pna  | primary    |           0.0981 |             0.0872 |              0.1078 |         0.1435 |           0.1475 |
| A0_mean_expression  | secondary  |         nan      |           nan      |            nan      |       nan      |         nan      |
| A3_matched_mlp      | secondary  |           0.0857 |             0.0759 |              0.0949 |         0.1204 |           0.1302 |
| A4_random_graph     | secondary  |           0.0997 |             0.088  |              0.1101 |         0.1451 |           0.1485 |
| A9_gatv2_multiscale | secondary  |           0.1024 |             0.0911 |              0.112  |         0.1491 |           0.1523 |

Per-cancer breakdown: `tables/model_comparison_per_cancer.csv`. Pathway/PROGENy14-only view:
`tables/pathway_progeny_results.csv`.

## Official HEST-Benchmark reproduction (SEPARATE protocol -- log1p-only targets, PCA256+Ridge,
official k-fold splits, seed=1 -- NOT comparable numerically to the custom-experiment table above,
which uses train-fold HVG selection, a trained patch head, and GNN correction)

| cancer   | encoder                   |   n_folds |   pearson_mean |   pearson_std |
|:---------|:--------------------------|----------:|---------------:|--------------:|
| IDC      | distilled_convnextv2_atto |         4 |         0.4822 |        0.0489 |
| IDC      | genbio_pathfm             |         4 |         0.5898 |        0.0861 |
| COAD     | distilled_convnextv2_atto |         2 |         0.2385 |        0.0282 |
| COAD     | genbio_pathfm             |         2 |         0.3264 |        0.0184 |
| CCRCC    | distilled_convnextv2_atto |         6 |         0.2225 |        0.0544 |
| CCRCC    | genbio_pathfm             |         6 |         0.2638 |        0.0277 |

## GenBio-PCA256-Ridge baseline (B1) -- **PARTIAL, fold0/seed0 only, not full 5-fold CV**

| cancer   | status                                                                                                                               |   HVG50_r |   HVG1000_r |   PROGENy14_r |
|:---------|:-------------------------------------------------------------------------------------------------------------------------------------|----------:|------------:|--------------:|
| IDC      | PARTIAL: fold0/seed0 only, not full 5-fold CV -- do not compare directly to the 5-fold macro table above without accounting for this |    0.1478 |      0.1053 |       -0.0157 |
| COAD     | PARTIAL: fold0/seed0 only, not full 5-fold CV -- do not compare directly to the 5-fold macro table above without accounting for this |    0.3357 |      0.1941 |       -0.0023 |
| CCRCC    | PARTIAL: fold0/seed0 only, not full 5-fold CV -- do not compare directly to the 5-fold macro table above without accounting for this |    0.2058 |      0.1227 |        0.024  |

This is preliminary (single fold per cancer); do not treat it as equivalent evidence to the
5-fold macro table above. A full 5-fold B1 rerun would be needed before drawing a firm B1
conclusion.

## Known limitations (reported honestly, not silently omitted)

1. **Bootstrap resampling unit.** All CIs above resample (cancer, fold, seed) cells (15 pooled,
   5 per cancer), not literal per-patient R_p values -- script 10 does not currently persist
   per-patient predictions to disk, only fold-level aggregates. A stricter implementation would
   plumb per-patient R_p through to script 12/20 and bootstrap over patients directly. Flagged
   since job 84871's design, not new to this aggregation step.
2. **Seeds 1/2** for the major models (A2/A8/B1) were not run -- seed 0 only, per the spec's
   "seed 0 is primary; seeds 1/2 only if compute allows" allowance.
3. **B1 (GenBio Ridge baseline)** is fold0/seed0 only for each cancer (see above), not the full
   5-fold design the rest of this table uses.
4. **Efficiency**: only parameter counts (student, teacher, compression ratio) were measured;
   VRAM/throughput/wall-clock were not.
5. The **IDC-specific distillation reversal** noted during the interactive diagnostic session
   (A1 slightly ahead of A2 in IDC specifically) has not been separately root-caused -- see
   `tables/model_comparison_per_cancer.csv` for the raw per-cancer numbers.

## Reproducibility

Exact patient-to-fold assignment used: `folds/cv_folds_used.csv`. Full config:
`experimental_runs/2026-08-31_job84871_final_5fold_3cancer_seed0/config_used.json`. Raw per-run
metrics: `results/cv_ablation_metrics.csv` (150 rows = 3 cancers x 5 folds x 10 ablations x
seed 0).
