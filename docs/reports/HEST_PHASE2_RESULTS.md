# HEST Phase-2 results (CNN/GNN downstream, job 84733)

**Created**: 2026-08-30 (SLURM job 84733) · **Last updated**: 2026-09-14 (protocol note added; numbers unchanged)

> **Protocol note**: this is an early, provisional, **single-fold**
> custom-protocol run (own patient-level splits, HVG1000/HVG50/PROGENy14
> targets, patient-level Fisher-z Pearson R) — superseded for the primary
> result by the locked 5-fold run in
> [`HEST_FINAL_5FOLD_RESULTS.md`](HEST_FINAL_5FOLD_RESULTS.md) (job 84871).
> Kept as the historical record of the first clean end-to-end run,
> including a real bug (A1_imagenet_patch) found and fixed afterward. The
> published paper reports results under the separate **official HEST-1k
> Benchmark protocol** instead (see
> [`HEST_OFFICIAL_ABLATION_RESULTS.md`](HEST_OFFICIAL_ABLATION_RESULTS.md));
> nothing in this document is restated against the paper's numbers, since
> neither the protocol, the gene panel, nor the metric overlaps.

Real, computed results from the first fully clean end-to-end completion
of Steps 8-12 against the real 3-cancer HEST dataset
(`src/scripts/10_run_patient_cv.py` / `src/scripts/12_primary_hypothesis_test.py`,
job **84733**, 2026-08-30). Compiled directly from
`results/cv_ablation_metrics.csv` and `results/primary_hypothesis_test.json`
(mirrored into `experimental_runs/2026-08-30_job84733_hest_downstream_full_3cancer/`);
nothing here is estimated or reconstructed from memory. The separate,
standalone frozen-GenBio-PathFM comparison (job 84744) is documented in
`docs/RUN_STATUS.md`'s "MILESTONE 2" entry, which also has the full
operational narrative for this run (every bug found/fixed getting here).

## Scientific question

Given H&E patches from breast (IDC), colon (COAD), and kidney (CCRCC)
tumor sections, how well does spatial gene expression (HVG1000/HVG50)
and PROGENy14 pathway activity predict from: (1) a frozen,
H-Optimus-0-distilled 3.4M-param CNN patch encoder alone, versus (2)
that same encoder's output corrected by a graph neural network using the
tissue's real physical multiscale spatial structure, versus negative
controls (shuffled/random graphs, mean-expression floor)?

## Dataset & run design (honest accounting)

| | |
|---|---|
| Cancers | IDC (breast), COAD (colon), CCRCC (kidney) -- 202 total HEST samples audited |
| Samples actually used | **191** (11 excluded: 6 IDC + 5 COAD samples run on a targeted ~541-gene panel (Xenium-style) rather than whole-transcriptome ST -- see `docs/RUN_STATUS.md`'s "Step 10 bug 3" for why; COAD/CCRCC data source is a different HEST-1k download than IDC's, unaffected otherwise) |
| Split | **Single 80/20 patient-level GroupKFold split per cancer** (`--max-folds-per-cancer 1 --seeds 0`) -- NOT the full V2 Phase 3 design (5-fold x 3-seed); a deliberate time-budget tradeoff for this run, not the project's final design |
| Test-set size | IDC 23 patients, COAD 11 patients, CCRCC 5 patients |
| HVG1000 gene count | Full 1000 for every cancer (the targeted-panel exclusion above keeps this apples-to-apples across cancers -- before that fix, IDC's shared gene pool had silently collapsed to 171) |

**One fold, one seed per cancer -- real but noisy, provisional numbers
from the first successful run, not a converged final result.** Treat
differences smaller than the bootstrap CIs below (Gate section) as
noise at this sample size, especially for CCRCC (n=5 test patients).

## Ablation grid

9 ablations (A1_imagenet_patch REMOVED, see caveat below):

| ID | What it tests |
|---|---|
| A0_mean_expression | Predict the training-set mean gene vector for every spot -- sanity floor |
| A2_distilled_patch | Frozen H-Optimus-0-distilled CNN, patch-only, no graph -- **the anchor** |
| A3_matched_mlp | Parameter-matched MLP on the same features, no spatial neighbors -- tests whether extra capacity alone (no real graph) explains any gain |
| A5_shuffled_graph | Degree-preserving rewired graph -- **critical negative control** |
| A6_local_pna | Real LOCAL-only physical graph, union-PNA backend |
| A7_multiscale_pna | Real local+meso+macro union graph, one PNA sees all edges |
| A8_scale_aware_pna | **Proposed final model** -- three per-scale PNA branches + learned softmax gate |
| A4_random_graph | Same node/edge count, uniform random edges, union-PNA backend |
| A9_gatv2_multiscale | Architecture control -- same union multiscale graph, GATv2 instead of PNA |

## Headline results: HVG1000 (1000 genes, the primary target)

Patient-level Fisher-z-aggregated Pearson r, mean squared error, mean
absolute error -- per cancer and averaged across the 3 cancers (unweighted
mean of the 3 per-cancer numbers, not spot-weighted):

| Model | IDC r | COAD r | CCRCC r | **MEAN r** | IDC MSE | COAD MSE | CCRCC MSE | **MEAN MSE** | IDC MAE | COAD MAE | CCRCC MAE | **MEAN MAE** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0_mean_expression | n/a* | n/a* | n/a* | n/a* | 1.1119 | 0.8382 | 1.2545 | 1.0682 | 0.9253 | 0.7567 | 0.9364 | 0.8728 |
| **A2_distilled_patch (anchor)** | **0.0704** | **0.1159** | **0.0587** | **0.0817** | 1.0643 | 0.7597 | 1.1094 | 0.9778 | 0.8834 | 0.7019 | 0.8589 | 0.8147 |
| A3_matched_mlp | 0.0867 | 0.1071 | 0.0656 | 0.0864 | 1.0540 | 0.7658 | 1.1047 | 0.9748 | 0.8738 | 0.7040 | 0.8570 | 0.8116 |
| A5_shuffled_graph (negative control) | 0.0043 | 0.0282 | -0.0023 | 0.0101 | 1.2483 | 0.7971 | 1.2432 | 1.0962 | 0.9297 | 0.7172 | 0.9137 | 0.8535 |
| A6_local_pna | 0.0046 | 0.0292 | -0.0027 | 0.0104 | 1.2071 | 0.7839 | 1.2421 | 1.0777 | 0.9210 | 0.7095 | 0.9127 | 0.8477 |
| A7_multiscale_pna | 0.0090 | 0.0306 | -0.0024 | 0.0124 | 1.2444 | 0.8055 | 1.2666 | 1.1055 | 0.9290 | 0.7171 | 0.9218 | 0.8560 |
| **A8_scale_aware_pna (proposed)** | **0.0075** | **0.0314** | **-0.0017** | **0.0124** | 1.2477 | 0.7953 | 1.2660 | 1.1030 | 0.9305 | 0.7125 | 0.9224 | 0.8551 |
| A4_random_graph | 0.0046 | 0.0282 | -0.0025 | 0.0101 | 1.2558 | 0.7956 | 1.2465 | 1.0993 | 0.9314 | 0.7164 | 0.9135 | 0.8538 |
| A9_gatv2_multiscale | 0.0076 | 0.0271 | -0.0019 | 0.0110 | 1.2606 | 0.7964 | 1.2313 | 1.0961 | 0.9332 | 0.7219 | 0.9101 | 0.8551 |

*A0's per-patient r is undefined (`NaN`): its prediction is a constant
vector (the training mean) for every test spot, so Pearson correlation
against a zero-variance series is mathematically undefined, not a
missing-data gap.

**The patch-only models (A2, A3) score highest of every model in this
table, including every graph variant, in every cancer.** All 6
graph-corrected variants (A4-A9) score close to the shuffled-graph
negative control (A5) and, for CCRCC, slightly below zero -- worse than
the sanity floor. This is the opposite of the project's hypothesis
(graph correction should beat patch-only via real spatial structure).

## HVG50 (top-50 subset of HVG1000) and PROGENy14 pathway activity

Same pattern holds for HVG50 (mean r: A2 0.1064, A3 0.1172, A8 0.0127).
See `results/cv_ablation_metrics.csv` / `experimental_runs/
2026-08-30_job84733_hest_downstream_full_3cancer/results/cv_ablation_metrics.csv`
for the full per-cancer HVG50 table (same 9-model x 3-cancer shape as
above, all columns prefixed `hvg50_`).

**PROGENy14 pathway activity -- a notable pattern**: A5 through A9 (all
6 graph-corrected variants) produce **byte-identical** pathway r/MSE/MAE
within each cancer (e.g. IDC pathway r = 0.0022 for every one of A4/A5/
A6/A7/A8/A9). Not root-caused here, but flagged: possibly the
graph-correction models only correct the gene head's output and pass
the pathway head's prediction through unchanged regardless of graph/backend.
Pathway signal is near-zero to slightly negative for every model in
every cancer (mean r: A2 0.0261, best graph variant 0.0019) --
essentially no pathway-level predictive signal in this run.

| Cancer | A2 pathway r | A5-A9 pathway r (identical) |
|---|---|---|
| IDC | 0.0323 | 0.0022 |
| COAD | 0.0093 | 0.0061 |
| CCRCC | 0.0366 | -0.0025 |

## Primary hypothesis gates (`results/primary_hypothesis_test.json`, pooled)

Patient-level bootstrap + paired sign-flip test,
`n_bootstrap` iterations, CI as configured in `configs/base.yaml`'s
`statistics` block. **Known limitation, printed by the script itself**:
the bootstrap resampling unit is (cancer, fold, seed) cells -- 3 units
total for this 1-fold/1-seed/3-cancer run, not individual patients; a
stricter implementation would bootstrap over per-patient R_p values
directly (this project's Section 25 literal requirement). Read the
p-values/CIs below as a coarse pooled signal, not a rigorous
patient-level test.

| Gate | Comparison | Mean delta | 95% CI | Cohen's d | Result |
|---|---|---|---|---|---|
| ~~Gate 1 (distillation)~~ | ~~A2 > A1~~ | 0.0 | [0.0, 0.0] | 0.0 | **REMOVED, see caveat below -- this result is invalid, shown only for the historical record** |
| Gate 2 (spatial context) | A8_scale_aware_pna > A2_distilled_patch | **-0.0692** | [-0.0845, -0.0604] | -5.22 | **NOT SUPPORTED** -- A8 is substantially WORSE than the patch-only anchor, not better |
| Gate 3 (true topology, decisive) | A8_scale_aware_pna > A5_shuffled_graph | +0.0024 | [0.0006, 0.0033] | 1.51 | **SUPPORTED** -- but both numbers are tiny (0.0075 vs 0.0043 mean r) and close to zero; a weak signal, not a strong one |

**Overall reading**: essentially all of the (modest, r~0.06-0.12)
predictive signal for gene expression comes from the frozen patch
encoder alone (A2/A3); the GNN spatial-correction layer isn't adding
value here -- it clears the weakest bar (beats a shuffled graph, Gate 3)
but fails the bar that matters for the project's thesis (beat
patch-only, Gate 2). Plausible unexplored causes: only 1 fold/1 seed
(high variance, especially CCRCC's n=5 test patients), limited GNN
training epochs in a time-budgeted run, or a genuine architecture/
training issue in the graph-correction path worth debugging before
drawing a real scientific conclusion.

## Known caveats

- **A1_imagenet_patch was REMOVED from the ablation grid** (and
  `gate1_distillation` from `configs/base.yaml`'s `primary_gates`) after
  this run completed, once its results were found byte-identical to
  A2's across all 3 cancers -- a real feature-loading bug (A1's declared
  "ImageNet, no distillation" encoder was never actually consulted;
  `src/scripts/10_run_patient_cv.py` always loaded the distilled feature
  cache regardless of ablation spec). Every other ablation (A0, A2-A9)
  already specs the encoder that's actually loaded, so only A1's row
  and Gate 1 were invalid -- everything else here is unaffected. See
  `docs/RUN_STATUS.md` for the full writeup, including a
  `base.json`-vs-`base.yaml` sync gotcha this surfaced.
- **Single fold/seed per cancer** -- not the project's final 5-fold x
  3-seed design. Rerun with `--max-folds-per-cancer 5 --seeds 0 1 2`
  once there's a multi-job or longer-wall-clock budget, per the
  original slurm script's design note.
- **Bootstrap unit is (cancer, fold, seed), not per-patient** -- see
  Gate section above.
- **CCRCC test set is only 5 patients** -- the noisiest of the 3
  cancers by a wide margin; treat its numbers with extra caution.
