# Figure code

Scripts that generate the paper figures. Each one reads its inputs from `figures/data/` (small, committed) and the
HEST-1k benchmark folders, and writes to `figures/output/` (git-ignored) or to the folder in `EPATHST_FIG_OUT`.

```
python figures/fig_compute_vs_pcc.py          # no external data
EPATHST_FIG_OUT=/path/to/out python figures/fig_leiden_tenx149.py
```

| Paper figure | Script | Writes | Inputs |
|---|---|---|---|
| Compute versus HEST-1k PCC (OLS trend and Pareto frontier) | `fig_compute_vs_pcc.py` | `tflops_vs_hest1k_pcc_ieee_trend_pareto.png` | `data/compute_vs_pcc_points.csv` |
| Leiden clusters of TENX149 (WSI, ground truth, three systems) | `fig_leiden_tenx149.py` | `leiden_tenx149_comparison.png` | `data/leiden_tenx149/`, HEST-1k `COAD/adata`, slide scan TENX149 |
| Spatial gene predictions in IDC, COAD, ccRCC | `fig_spatial_predictions.py` | `ePathST_vs_HOptimus0_two_gene_panel{,_greens,_ylgn,_bugn,_gnbu,_summer}.png` | `data/spatial_predictions/`, HEST-1k `adata` and patches, slide scans TENX148 and INT15 |
| Gene-level context gain (dot plot, Moran's I panel) | `fig_gene_context.py` | `gene_context_dotplot.png` | `data/gene_context/` |
| Fig. 3 pretraining data of the compared models (panels a-d) | `fig_pretraining_data.py` | `fig3_pretraining_data/fig3_*.{png,pdf}`, `fig3_panels.tex` | `data/pretraining_table.csv`, `data/pretraining_sources.md` |
| Fig. 3 PCC versus pretraining slides and tiles | `fig_data_efficiency.py` | `fig3_pretraining_data/fig3_data_efficiency.{png,pdf}` | `data/data_efficiency_points.csv` |

Upstream inference (GPU) that produced the two prediction inputs:

| Input folder | Script |
|---|---|
| `data/spatial_predictions/` | `inference/infer_spatial_predictions.py` (fold 0, ePathST and H-Optimus-0 per-spot predictions of one gene per cancer) |
| `data/leiden_tenx149/` | `inference/infer_leiden_tenx149.py` (COAD fold 0, Leiden labels of TENX149) |

`data/gene_context/` holds the per-gene tables (`gene_context_{IDC,COAD,CCRCC}.csv`, Spearman summary) computed from per-spot
predictions of the ablation models; the per-spot prediction dumps they were computed from are several GB and are not stored here.

## Comparison models are data, not code

The foundation models compared in the paper (their HEST scores, GFLOPs per patch, slide and tile counts, and where each number
comes from) live only in the three CSV files and `data/pretraining_sources.md`. The scripts contain no model-specific numbers.

## Matching paper file names

The LaTeX source (`Paper_Oct9th.txt`) includes `Architecture.png`, `Methodology_Part1.png`, `ePathST_gene_panel.png`,
`leiden_tenx149_comparison.png`, `tflops_vs_hest1k_pcc_trend_pareto.png` and `gene_context.png`. The first two are schematics with
no generating script in this repository. The other four were renamed after export; the mapping in the table above is the
closest match by content and date. Check that the files uploaded to the paper are the ones these scripts write:
the file named `gene_context.png` in the local `paper_figures/` folder is a different figure (a third-party plot of
GigaTIME against GigaTIME-Flash), not the output of `fig_gene_context.py`.

## Provenance of the stored ePathST predictions

The ePathST (and ePathST-E) predictions and Leiden labels in `data/spatial_predictions/` and `data/leiden_tenx149/` were kept
from the earlier training run for exact reproducibility; only the H-Optimus-0 predictions were recomputed (2026-09-13). Both
used the locked student checkpoint `locked_encoder_seed0.pt`. See `FINAL_REFACTOR_REPORT.md`, section 9, for what is known
about how that checkpoint was distilled.

## Checking the scripts

`tests/e2e_figures.py OUT_DIR` regenerates every figure into `OUT_DIR` and compares it pixel by pixel with the reference
images (`sbatch slurm/run_figures_check.slurm`).
