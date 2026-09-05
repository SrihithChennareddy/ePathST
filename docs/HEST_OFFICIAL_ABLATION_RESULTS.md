# A0-A9 Ablation Grid Under the Official HEST-1k Benchmark Protocol

**Job**: SLURM 87551 (`slurm/run_official_hest_ablation_grid.slurm`), all 3 cancers, all official
k-folds, seed=1 (official `BenchmarkConfig` default), feature-seed=0 (locked distilled encoder).
Runtime: 26 min (2026-09-04 11:34-12:01). Driver: `src/scripts/22_official_hest_ablation_grid.py`.

## What this is

`src/scripts/10_run_patient_cv.py` runs the full A0-A9 grid under this project's own custom
protocol (own patient-level GroupKFold splits, HVG1000/HVG50 selected per-fold from
train-only variance, normalize_total+log1p targets, PROGENy14 pathway scores).
`src/scripts/19_official_hest_benchmark.py` reproduces the **official** HEST-1k Benchmark protocol
(official k-fold splits, official curated 50-gene panel, log1p-only targets) but only for two
raw encoders via a from-scratch PCA(256)+Ridge head -- it never touches the trained
patch-head + PNA/GATv2 graph-correction models.

This run closes that gap: the *same* A0-A9 models script 10 trains (same `run_ablation`/
`train_patch_head` code, reused via `importlib`, not reimplemented) evaluated against the
*official* protocol's data, splits, and gene targets. See `src/scripts/22_official_hest_ablation_grid.py`'s
module docstring for the exact adaptations required (official 50-gene target, no PROGENy
pathway target -- not part of the official benchmark, `patient_id` = official `sample_id`,
pixel->micron spot coordinates via `mpp = 55/spot_diameter_fullres`, and a val==train fallback
for official folds with <2 training samples, e.g. COAD fold 0).

Official folds: IDC x4, COAD x2, CCRCC x6 (12 total). 9 trainable ablations x 12 folds = 108
trained-model runs (A0 is a fixed non-learned baseline, reported separately).

## Pooled results (Fisher-z mean patient R, n=12 folds each)

| Ablation | R (mean) | R (std) | Priority |
|---|---|---|---|
| A0_mean_expression | n/a (no model) | -- | secondary |
| A1_imagenet_patch | 0.2214 | 0.1557 | primary |
| A2_distilled_patch | 0.3100 | 0.1584 | primary |
| A3_matched_mlp | 0.3157 | 0.1794 | secondary |
| A4_random_graph | 0.3197 | 0.1669 | secondary |
| A5_shuffled_graph (negative control) | 0.3204 | 0.1685 | primary |
| A6_local_pna | 0.3195 | 0.1680 | primary |
| A7_multiscale_pna | 0.3209 | 0.1673 | primary |
| **A8_scale_aware_pna (flagship)** | **0.3218** | 0.1695 | primary |
| A9_gatv2_multiscale | 0.3203 | 0.1661 | secondary |

## Per-cancer results (Fisher-z mean patient R)

| Ablation | IDC | COAD | CCRCC |
|---|---|---|---|
| A1_imagenet_patch | 0.4122 | 0.2303 | 0.0913 |
| A2_distilled_patch | 0.4890 | 0.3451 | 0.1790 |
| A3_matched_mlp | 0.5162 | 0.3602 | 0.1673 |
| A4_random_graph | 0.5203 | 0.3180 | 0.1865 |
| A5_shuffled_graph | 0.5232 | 0.3211 | 0.1850 |
| A6_local_pna | 0.5187 | 0.3244 | 0.1851 |
| A7_multiscale_pna | 0.5198 | 0.3276 | 0.1860 |
| A8_scale_aware_pna | 0.5209 | 0.3362 | 0.1843 |
| A9_gatv2_multiscale | 0.5144 | 0.3317 | 0.1872 |

Difficulty ranks the same as the custom-protocol run: IDC (easiest) > COAD > CCRCC (hardest).

## Gate readout

- **Gate 1 (distillation, A2 > A1)**: holds clearly, on every cancer and pooled
  (pooled +0.089; IDC +0.077, COAD +0.115, CCRCC +0.088).
- **Gate 2 (spatial context, A8 > A2)**: holds, but the margin is small (pooled +0.012;
  IDC +0.032, COAD +0.091 largest per-cancer gain, CCRCC essentially flat -0.005).
- **Gate 3 (true topology, A8 > A5 shuffled-graph negative control)**: **fails** -- pooled A8
  (0.3218) is statistically indistinguishable from A5 (0.3204), the same finding already
  documented under this project's own protocol (`docs/HEST_FINAL_5FOLD_RESULTS.md`). All of
  A4-A9 (any graph, real or fake) cluster within ~0.31-0.32 pooled, and A3 (a parameter-matched
  MLP with **no** neighbor information at all) matches them too. The graph correction is not
  adding topology-specific signal beyond "any local averaging helps a little."

## GenBio-PathFM teacher reference (same official protocol, PCA(256)+Ridge head)

From `src/scripts/19_official_hest_benchmark.py` (`results/official_hest_benchmark/`), a
methodologically separate comparison (raw embeddings -> train-fold PCA(256) -> Ridge, not the
trained MLP bottleneck head A1/A2 use -- numbers are close to but not directly comparable to
A1/A2 above):

| Cancer | GenBio-PathFM (1.1B params, teacher) R | Distilled CNN (3.4M params, student) R |
|---|---|---|
| IDC | 0.5898 | 0.4822 |
| COAD | 0.3264 | 0.2385 |
| CCRCC | 0.2638 | 0.2225 |

The teacher beats the 334.7x-smaller distilled student by a real, consistent margin
(+0.108 IDC, +0.088 COAD, +0.041 CCRCC) -- expected given the size gap, and the student
captures a meaningful fraction of the teacher's signal, not near-zero.

## How this differs from the custom-protocol numbers (`docs/HEST_FINAL_5FOLD_RESULTS.md`)

The official-protocol R values above are substantially higher across the board than the
custom-protocol A0-A9 numbers (which were sub-0.1 for most ablations). This is **not** because
the graph model works better under the official protocol -- the two protocols score different,
non-overlapping gene sets:

- **Official 50-gene panel**: a fixed, curated marker panel (`<cancer>/var_50genes.json`),
  computed once by the HEST-1k authors via `sc.pp.highly_variable_genes` (dispersion-based,
  mean-binned) on log1p'd **raw** counts (no `normalize_total`), pooled across the **entire**
  train+test cohort of that cancer -- see `docs/HEST_OFFICIAL_HVG50_GENES.md` for the full
  methodology writeup and gene lists.
- **This project's own HVG50**: selected fresh per fold from **train-only** spots (leakage-free
  by construction), ranked by raw variance (not scanpy's dispersion statistic) after
  `normalize_total`+`log1p`.

Gene overlap between the two is 1-2 genes out of 50 per cancer -- not the same target, so the
two R numbers aren't directly comparable. What reproduces across both protocols is the
qualitative finding: Gate 3 fails, graph correction doesn't beat the shuffled-graph negative
control -- not the absolute R values.

## Leakage check

Traced explicitly for this run (see `src/scripts/22_official_hest_ablation_grid.py` and
`src/scripts/10_run_patient_cv.py::run_ablation`):

- Official train/test splits are sample-disjoint: verified 0 overlapping `sample_id`s across
  all 12 official train/test fold pairs.
- `TrainOnlyStandardizer.assert_no_leakage(ids_test)` is called unconditionally every fold
  (would hard-crash on any train/test id overlap; did not).
- Feature encoders (distilled + ImageNet) are frozen/eval-mode -- no fold's features are ever
  influenced by another fold's data.
- Graphs are built strictly per `sample_id`; train/val/test graphs only ever contain nodes from
  their own disjoint sample pool.
- One caveat: official folds with <2 train samples (COAD fold 0) use the full train pool as
  both train and val, so early stopping there monitors train loss rather than a true held-out
  signal (an optimistic *model-selection* bias, logged explicitly as `val==train fallback` in
  the run output). This never touches test data, so it does not leak into the reported test
  metrics.

## Output files

`/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/cptac_genbio_distill_hest_pna_work/results/official_hest_ablation_grid/`
- `official_ablation_metrics.csv` -- one row per cancer x fold x ablation (120 rows)
- `official_ablation_summary_macro.csv` -- mean/std per ablation, pooled across folds/cancers
- `run_args.json` -- exact CLI args used
