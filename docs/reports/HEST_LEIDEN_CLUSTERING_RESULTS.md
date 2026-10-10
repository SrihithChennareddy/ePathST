# Leiden clustering: predicted vs ground-truth spatial domains

PCC measures how well each gene is predicted. This check asks whether predicted expression recovers the same spatial domains as measured
expression: Leiden clustering is applied independently to the measured and to each model's predicted 50-gene expression, and the two partitions
are compared with the adjusted Rand index (ARI; AMI is also computed in section 2). Source for section 1: the paper, Fig. 3 and 4.

## 1. Representative sample (paper figure)

HEST sample TENX149 (COAD, 3,620 spots). Procedure: z-score, PCA (up to 30 components), kNN graph with k = 15, Leiden (igraph). The clustering
resolution was searched on the ground truth to give 5 to 9 clusters (resolution 0.3, 6 clusters) and reused for every prediction.

| Model | ARI |
|---|---|
| ePathST-E (patch-only) | 0.58 |
| ePathST (scale-aware PNA) | 0.66 |
| H-Optimus-0 (frozen, PCA-256 + Ridge) | 0.69 |

In the paper, ePathST improves ARI from 0.58 to 0.66, approaching the 0.69 of H-Optimus-0.

Per-gene prediction maps (paper Fig. 4, one gene and one sample per cancer; per-sample PCC for ePathST / H-Optimus-0): IDC GATA3 0.88 / 0.89,
COAD CEACAM6 0.79 / 0.39, ccRCC STK39 0.48 / 0.50. These are illustrations of what a prediction map looks like, not aggregate statistics.

## 2. All 32 HEST test samples (benchmark-fold training)

The same comparison on every sample of the official HEST test folds (IDC 4, COAD 4, ccRCC 24), with ePathST-E, ePathST and H-Optimus-0 trained on the
benchmark's own training slides in each official fold (benchmark-fold training), seeds 0 and 1. Leiden recipe: z-score, PCA (20), kNN k = 15,
igraph Leiden with random_state 0 at resolution 0.5 (0.3 and 1.0 as sensitivity); measured expression is clustered once per sample and the predictions
are reindexed to it by spot id. H-Optimus-0 here is the frozen teacher with PCA reduction and the same MLP patch head as ePathST-E.

| Mean ARI at resolution 0.5 | ePathST-E | ePathST | H-Optimus-0 |
|---|---|---|---|
| All 32 samples, seed 0 | 0.128 | 0.128 | 0.140 |
| All 32 samples, seed 1 | 0.127 | 0.132 | 0.139 |
| Three cancers weighted equally, seed 0 | 0.196 | 0.192 | 0.239 |
| Three cancers weighted equally, seed 1 | 0.199 | 0.198 | 0.227 |
| IDC (4), seeds 0 / 1 | 0.208 / 0.221 | 0.212 / 0.219 | 0.299 / 0.293 |
| COAD (4), seeds 0 / 1 | 0.376 / 0.371 | 0.345 / 0.362 | 0.380 / 0.356 |
| ccRCC (24), seeds 0 / 1 | 0.074 / 0.071 | 0.078 / 0.079 | 0.074 / 0.077 |

Paired Wilcoxon tests across samples between the three systems give p between 0.06 and 0.73 (none below 0.05). ccRCC contributes 24 of the 32 samples
and has low ARI for all systems. On this larger evaluation the ePathST-E to ePathST difference seen on TENX149 is not present, and repeated training
runs on TENX149 itself give ePathST-E 0.58 to 0.67 and ePathST 0.61 to 0.66, so a single-sample ordering is not stable.

**Reproducibility of the clustering (seed 0 run):** re-running the identical clustering on the saved matrices reproduces 94.5% of the saved label sets exactly; the
rest move the mean ARI by at most 0.005 (per sample at most 0.063), and the system ordering is unchanged.

Files: `results/leiden_official_hvg50{,_seed1}/` (per-sample table, summary, paired tests, ground-truth and predicted labels, models); see
`docs/reports/RESULTS_INDEX.md`.

## 3. Caveats

- Section 1 is one sample per score and one gene per map; no confidence interval applies.
- H-Optimus-0 is the full-scale reference (it is the distillation teacher) and has about 215 times the parameters of ePathST.
- Section 2 uses benchmark-fold training (1 to 3 slides per fold for IDC and COAD), so absolute values differ from the cohort-trained models of the paper tables.
