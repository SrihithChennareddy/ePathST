# Documentation index

The methods, evaluation protocol and headline results of the paper are in [`METHODS.md`](METHODS.md), which follows the final (Oct 9) version of the
paper. Everything else documents specific analyses, inputs and compute.

## Start here
| File | Content |
|---|---|
| `../README.md` | project README (abstract, installation, pipeline overview) |
| `METHODS.md` | models, data, distillation, multiscale prediction, evaluation protocol, results tables, reproducibility map |
| `reports/RESULTS_INDEX.md` | where every supplementary result lives and how to load it: official-panel Leiden, frozen models, TCGA predictions, companion data |
| `../figures/README.md` | which script and data produce each paper figure |
| `../FINAL_REFACTOR_REPORT.md` | code organization, moved and merged files, behaviour checks, open issues |

## Results
| File | Content |
|---|---|
| `reports/HEST_OFFICIAL_ABLATION_RESULTS.md` | evaluation protocol, how the paper reads Table V, additional A0-A9 controls (benchmark-fold training), Moran's I, validation checks |
| `reports/HEST_LEIDEN_CLUSTERING_RESULTS.md` | Leiden spatial-domain check: TENX149 (paper) and all 32 HEST test samples (benchmark-fold training) |
| `reports/HEST_OFFICIAL_HVG50_GENES.md` | official 50-gene panels versus the project's own HVG50, per cancer |

## Inputs and compute
| File | Content |
|---|---|
| `reports/INPUT_DATA.md` | input datasets |
| `reports/COMPUTE_RESOURCES.md` | compute used, with the accounting ledger |
| `../input_data/README.md` | input data folder note |

## Research history
`../FINAL_REFACTOR_REPORT.md` records the presentation-only refactor. `METHODS.md` is the current description of the work.

## Training and test splits
Two protocols are described in the documents; each result states which one produced it.

| Protocol | Training slides | Test slides | Where |
|---|---|---|---|
| Cohort training (paper Table III/V) | per cancer, HEST-1K cohort slides outside the held-out test partition (about 200 across the three cancers) | held-out HEST test partition (official benchmark test folds) | `METHODS.md` |
| Benchmark-fold training | the official benchmark's own training slides of each fold (IDC 3, COAD 1 to 3, ccRCC 20 slides per fold) | the fold's test slides | `reports/RESULTS_INDEX.md`, Leiden, TCGA, A0-A9 controls |

## Run snapshots (`../experimental_runs/`)
`INDEX.csv` lists every snapshot; each folder has `manifest.json`, `results/`, `figures/` and `notes.md`:

| Date | Job | Content |
|---|---|---|
| 2026-08-28 | 83825 | CPTAC distillation |
| 2026-08-30 | 84733, 84759 | HEST downstream, patch-only vs graph |
| 2026-08-30 | 84744 | frozen H-Optimus-0 patch-only baseline |
| 2026-08-31 | 84871 | custom 5-fold cross-validation (two snapshots) |
| 2026-09-05 | 88050 | official-protocol A0-A9 ablation grid |
| 2026-09-05 | 88100 | Moran's I diagnostic |

Larger outputs (Leiden runs, frozen models, TCGA predictions and companion data) are indexed in `reports/RESULTS_INDEX.md`.
