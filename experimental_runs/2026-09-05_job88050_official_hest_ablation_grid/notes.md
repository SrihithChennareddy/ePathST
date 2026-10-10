# 2026-09-05_job88050_official_hest_ablation_grid

Stage: official A0-A9 grid, all 3 cancers, official protocol (Fisher-z + official pooled Pearson)

This snapshot's primary purpose is capturing job 88050's raw
`results/official_hest_ablation_grid/*` output (never previously
committed -- see `docs/reports/HEST_OFFICIAL_ABLATION_RESULTS.md` for the summary
that was written from it). Everything else under `results/` is whatever
else happened to be present in the shared results tree at snapshot time.

Caveat: `results/morans_i_diagnostic/*` here is the 1-fold IDC-only smoke
test (job 88082), not the full 3-cancer run -- job 88083 (full run) was
still queued when this snapshot was taken. Will re-snapshot once it
completes.

The duplicate copies of `results/final/FINAL_RESULTS.md` and `results/final/HEST_FINAL_5FOLD_RESULTS.md` that this snapshot originally carried were removed; they are identical to the copies in `2026-08-31_job84871_final_5fold_3cancer_seed0_with_results_final/` and the maintained version is `docs/reports/HEST_FINAL_5FOLD_RESULTS.md`.
