# 2026-08-30_job84744_genbio_baseline_3cancer

Job 84744 (`slurm/run_genbio_hest_baseline.slurm`), first fully clean
completed run of the standalone frozen patch-only baseline (embeddings from the H-Optimus-0 teacher; the folder and file names keep the earlier
"genbio" label),
all 3 cancers (IDC/COAD/CCRCC), seed=0 fold=0. COMPLETED, exit 0:0,
7:20 elapsed (extraction was ~fully cached from earlier runs' resumable
progress, so this run is almost entirely head-training + eval time).

Ran with all 6 accumulated fixes live: manifest-wait race fix,
targeted-gene-panel-sample exclusion (11/202 samples dropped, see
docs/reports/RUN_STATUS.md), Ensembl-gene-symbol mapping, duplicate-var-names
dedup, NaN-assertion relaxation (decoupler-dropped degenerate spots are
expected-NaN, not a bug), and the 44G->160G memory bump.

No `config_used.json`/`manifest.json` (this script doesn't call
`scripts/snapshot_experiment.py` the way the main pipeline's slurm
script does) -- just the actual result files copied directly:
`genbio_summary.csv` (headline per-cancer HVG50_r/HVG1000_r/PROGENy14_r
+ mse/mae), `genbio_gene_metrics.csv`/`genbio_pathway_metrics.csv`
(per-patient-per-gene/pathway detail for finer-grained graphing), and
`per_cancer/*.json` (full per-cancer record including the exact
train/val/test patient split used).

See docs/reports/RUN_STATUS.md for the full narrative and headline numbers.
