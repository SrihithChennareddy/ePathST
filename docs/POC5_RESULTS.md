# PoC_5 results (CRC spatial PNA-GNN proof-of-concept)

Real, computed results from the PoC_5 project (`STS_PoC/PoC_5`,
`crcspatial` package) — the predecessor proof-of-concept this project's
Final_Project.txt explicitly says to "consider... for this project
direction." Compiled from the pipeline's own output files (per-fold/seed
`POC_REPORT_*.md` reports, `outputs/cross_fold_aggregate/*`,
`outputs/compute_accounting/*`) under
`/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/PoC_5/outputs/` —
nothing here is estimated or reconstructed from memory.

## Scientific question

Can a lightweight, spatially-structured model (MobileNetV3-Small patch
encoder + physical multiscale PNA graph) predict spatial gene expression
and pathway activity from colorectal H&E better using true tissue
topology than an identical model with destroyed/shuffled spatial
structure? Primary hypothesis: real multiscale spatial graph >
degree-preserving shuffled graph, on completely unseen patients.

## Dataset

CRC23 cohort: **23 independent colorectal-cancer patients**
(`data/manifest.csv`), one WSI + paired spatial transcriptomics section
per patient. Design: 5-fold patient-level outer CV (fold sizes 5,5,5,4,4)
x 3 seeds (0,1,2) = 15 planned fold x seed runs.

## Run completion status (honest accounting — not all planned runs finished)

| Fold | Seeds run | Status |
|---|---|---|
| 0 | 0, 1, 2 | **Completed** (real results) |
| 1 | 0, 1, 2 | **Completed** (real results) |
| 2 | 0, 1, 2 | **Completed** (real results) |
| 3 | 0, 1, 2 | **Completed** (real results) |
| 4 | 0, 1, 2 | **FAILED, all 3 seeds** — job invoked a script path (`STS/PoC_5/scripts/08_train_patch_baseline.py`) that didn't exist ("No such file or directory"); zero real compute happened, zero usable results |

**12 of the 15 planned fold x seed runs completed with real results; fold
4 (both/all 3 seeds) produced nothing.** This is a genuine, documented gap
in the 5-fold design, not a rounding choice — report 4-fold (not 5-fold)
coverage if citing this in a paper, or rerun fold 4 with the corrected
script path first.

**Further gap**: the pipeline's own cross-fold aggregation script
(`20_aggregate_cross_fold.py`) was run using only **folds 0, 1, 2** (12
test patients) — fold 3's results, though real and complete, were never
folded into the official aggregate on disk. The numbers below reflect
that as-computed 3-fold aggregate; a true 4-fold aggregate (including
fold 3) has not been (re)computed and would need `20_aggregate_cross_fold.py`
rerun with `--folds 0,1,2,3` (or equivalent) to produce a slightly
different, more complete number.

## Headline results table (official cross-fold aggregate, folds 0-2, n=12 test patients, HVG50)

Patient-level Fisher-z-aggregated Pearson r, from
`outputs/cross_fold_aggregate/headline_hvg50.csv`:

| Model | Ablation ID | Patient macro R | Median patient R |
|---|---|---|---|
| Patch CNN | `patch_only` | 0.1875 | 0.1533 |
| Patch CNN (scratch init) | `A9_random_init` | 0.0960 | 0.0601 |
| Parameter-matched MLP | `A2_param_matched` | 0.1345 | 0.1134 |
| Random graph | `A3_random_graph` | 0.1182 | 0.0658 |
| Shuffled physical graph | `A4_degree_shuffled` | 0.1188 | 0.0642 |
| Local PNA | `A5_local_pna` | 0.1229 | 0.0709 |
| Multiscale PNA | `A6_multiscale_pna` | 0.1268 | 0.0693 |
| Multiscale PNA + diffusion | `A7_multiscale_diffusion` | 0.1154 | 0.0508 |
| Multiscale + diffusion (no pathway loss) | `A8_no_pathway_loss` | 0.1317 | 0.0844 |

**The patch-only CNN baseline scores HIGHEST** of every model in this
table, including every graph variant — the graph models do not beat the
simple patch encoder on this 3-fold aggregate. This is the opposite of
what the PoC's hypothesis predicted (a monotonic CNN ≲ random ≲ shuffle <
local < multiscale < diffusion progression).

## Primary hypothesis test (official, folds 0-2, n=12 patients, HVG50)

From `outputs/cross_fold_aggregate/cross_fold_statistics_hvg50.json` —
patient-level bootstrap, sign-flip test:

| # | Comparison | mean Δz | 95% CI | p (sign-flip) | Result |
|---|---|---|---|---|---|
| 1 (**DECISIVE**) | real graph vs. degree-shuffled graph | -0.0035 | [-0.0173, 0.0099] | 0.681 | **NOT SUPPORTED** — CI includes 0, and the point estimate is slightly negative |
| 2 | multiscale vs. local | +0.0040 | [0.0002, 0.0073] | 0.036 | Supported (small positive effect, CI excludes 0) |
| 3 | diffusion vs. PNA-only | -0.0116 | [-0.0207, -0.0036] | 0.992 | Not supported (negative — diffusion made things worse) |

**1 of 3 primary hypotheses supported** (the same "1/3 PASS: partial
support" conclusion the pipeline's own per-fold reports independently
reached, e.g. fold 0/seed 1). Per this project's own PoC_5.txt: *"If
[real > shuffled] fails, the main idea is weakened regardless of how
close you get to a PFM."* **The single most important test — the one
PoC_5.txt calls decisive — did not pass**: no evidence in this run that
real spatial topology outperforms a degree-preserving shuffle.

### Secondary comparisons (same aggregate)

| Comparison | mean Δz | 95% CI | p | Result |
|---|---|---|---|---|
| multiscale vs. random graph | +0.0087 | [0.0019, 0.0163] | 0.020 | Supported |
| local vs. random graph | +0.0047 | [-0.0028, 0.0131] | 0.148 | Not supported |
| graph vs. parameter-matched MLP | -0.0079 | [-0.0241, 0.0077] | 0.816 | Not supported |
| diffusion (real biology) vs. no-pathway-loss ablation | -0.0166 | [-0.0247, -0.0094] | 1.000 | Not supported |
| ImageNet-init vs. scratch-init | +0.0934 | [0.0517, 0.1438] | 0.0004 | Supported — ImageNet pretraining clearly helps |

## Example per-fold variability (individual fold x seed runs are noisy)

Fold-level results vary considerably — illustrative examples from two
real, complete `POC_REPORT_*.md` files (HVG50 r):

**Fold 0, seed 1** (`POC_REPORT_fold0_seed1.md`):
Patch CNN 0.1129, Multiscale PNA 0.0231, Multiscale+diffusion 0.0297 —
graph models underperform patch-only badly in this fold.

**Fold 3, seed 0** (`POC_REPORT_fold3_seed0.md`):
Patch CNN 0.3236, Local PNA 0.3642, Multiscale+diffusion 0.3661 — here
graph models modestly BEAT patch-only, and absolute scores are ~3x higher
than fold 0's.

This fold-to-fold volatility (r ranging ~0.02-0.37 for the same model
across different folds) is itself a notable finding — with only 23
patients split 5 ways, individual outer folds are small and noisy; the
pooled cross-fold aggregate (above) is the statistically meaningful
number, not any single fold.

## Biological/spatial-structure analyses (fold 0, seed 1 example; per-fold, not yet cross-fold pooled)

- **Moran's I correlation** (do genes with more spatial autocorrelation
  benefit more from the graph?): `Corr(Δ_g, MoranI_g) = 0.1148, p = 0.630`
  (n=20 genes) — weak positive trend, not statistically significant.
- **Multiscale vs. local, per-gene**: 507/1000 genes gained from
  meso+macro context over local-only; 493/1000 genes gained from
  diffusion over multiscale — essentially a coin flip, no strong signal
  either way.

## Model efficiency (from `outputs/compute_accounting/compute_accounting.json`)

| Model | Trainable params | FLOPs/patch | Wall-clock (this fold/seed) |
|---|---|---|---|
| Patch CNN (`patch_only`) | 1,074,835 | 109,940,608 | 414.7s |
| Patch CNN, scratch init (`A9_random_init`) | 1,074,835 | 109,940,608 | 410.7s |
| Parameter-matched MLP | 670,634 | — | 0.44s |
| Random graph | 637,427 | — | 2.28s |
| Degree-shuffled graph | 637,427 | — | 2.16s |
| Local PNA | 637,427 | — | 1.04s |

Patch-CNN backbone comfortably meets the PoC's own `<2.5M trainable
parameters` design target (~1.07M actual). Graph-stage training is
extremely cheap (~1-2s) since it trains only the PNA/diffusion/head on
already-cached, frozen CNN embeddings — the CNN forward/backward pass
(~7 minutes) dominates total wall-clock per fold/seed.

## Bottom line

- The PoC's **decisive primary hypothesis (real graph > shuffled graph)
  did not pass** in the completed 3-fold aggregate — no evidence yet that
  true spatial topology beats a degree-preserving shuffle on this cohort.
- The **patch-only CNN baseline outperformed every graph variant**,
  contrary to the hypothesized CNN < local < multiscale < diffusion
  progression.
- ImageNet pretraining clearly helps (strongly significant), and
  multiscale beats purely-local and beats random-graph (both
  significant) — so the graph architecture isn't inert, it just doesn't
  yet beat the simplest patch-only baseline or the shuffled-topology
  control.
- Coverage is incomplete: only 4 of 5 folds ran (fold 4 failed on a
  broken script path), and the on-disk aggregate itself only pools 3 of
  those 4 completed folds. Before citing a final number, rerun fold 4 and
  regenerate the cross-fold aggregate over all 4 (or 5) completed folds.
