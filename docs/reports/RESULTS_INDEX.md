# Results index: official-panel Leiden, frozen models, TCGA predictions

Last updated 2026-10-09. Everything below is on `/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/` (project
space, not purged) unless marked **scratch** (purged after 30 days, re-derivable). `$WORK` =
`cptac_hoptimus0_distill_hest_pna_work`, `$RES` = `$WORK/results`.

## Map

| What | Where | Made by | Status |
|---|---|---|---|
| Official-cohort Leiden + predictions + saved models, **seed 0** | `$RES/leiden_official_hvg50/` | `35_leiden_official_hvg50.py` (now `25_leiden_official_panel.py official`), job 951095 | done |
| Same, **seed 1** (official BenchmarkConfig default) | `$RES/leiden_official_hvg50_seed1/` | same script, job 951437 | done |
| Frozen-weight checksums | `<run dir>/FROZEN_WEIGHTS_MANIFEST.json` | `42_frozen_weights_manifest.py` | see note 1 |
| HEST official-panel gene coverage per sample | `$RES/hest_panel_coverage/` | `41_hest_panel_coverage.py` | see note 1 |
| TCGA predictions, 100 um hex pseudo-spots | `tcga/predictions/hex100_leiden_official_hvg50/` | `43_tcga_worker.py` (self-scheduling GPU workers) | **done, audit COMPLETE** (2026-10-07) |
| TCGA smoke tests | `tcga/predictions/smoke_hex100/`, `smoke_tcga/` | scripts 40, 38 | done, disposable |
| TCGA embedding cache (**scratch**, 108 GB) | `/lustre/nvwulf/scratch/schennareddy/TCGA/embeddings_hex100/` | scripts 40/43 | complete; lets heads/GNN be re-run without the encoders |
| TCGA companion data (RNA-seq, TCGA-CDR, CPE purity, cBioPortal, GDC clinical) | **scratch** `/lustre/nvwulf/scratch/schennareddy/TCGA/companion/` (8.2 GB); small tables mirrored in `tcga/companion_processed/` (12 MB) | `50_tcga_companion_download.py` | done 2026-10-07; README + manifests inside |
| Old Leiden runs (project 5-fold CV) | `cptac_hoptimus0_distill_hest_pna/leiden_clustering/` | scripts 32, 33, 34 | partial, see caveats |
| TENX149 re-run (seed 1) | `STS_src/results_tenx149_recheck/seed1/` | `36_leiden_tenx149_recheck.py` | done |

Note 1: written by background jobs on 2026-10-06; if the file is missing, rerun the script (one command, a few minutes).

## 1. Official-cohort Leiden (script 35)

**Protocol.** Official HEST-1k benchmark folds (IDC 4, COAD 2, CCRCC 6) and each cancer's own official 50-gene panel
(`<cancer>/var_50genes.json`, log1p-only target). Every one of the 32 official test samples (IDC 4, COAD 4, CCRCC 24) is
predicted by a model that never trained on it. Systems: **ePathST-E** (frozen student + patch head), **ePathST** (+ scale-aware
PNA graph model), **H-Optimus-0** (frozen teacher, PCA to the student's dimension, same patch head). Leiden: z-score, PCA(20),
kNN k=15, igraph, `random_state` 0, resolution 0.5 (sensitivity 0.3, 1.0); measured expression clustered once per sample, the
predictions reindexed to it by spot ID; ARI and AMI; Wilcoxon tests between systems. Samples <50 spots skipped (none).

**Folder layout** (`$RES/leiden_official_hvg50*/`)
- `per_sample_leiden.csv` (+ `_{cancer}.csv`): one row per sample x system x resolution (ARI, AMI, cluster counts)
- `leiden_summary.csv`, `leiden_paired_tests.csv`, `run_args.json`
- `preds/{cancer}_fold{n}_{system}.npz`: held-out predictions (`spot_uid, x_um, y_um, y_true, y_pred, genes`; standardized scale)
- `labels/{cancer}_{sample}.npz`: ground-truth Leiden labels (`gt__res0.5`...), each system's labels (`{system}__res0.5`),
  `y_true`, `y_pred__{system}`, coordinates, genes
- `models/{cancer}_fold{n}/`: saved models (section 2)

**Headline numbers** (mean over the 32 samples, resolution 0.5)

| | ePathST-E | ePathST | H-Optimus-0 |
|---|---|---|---|
| ARI seed 0 | 0.128 | 0.128 | 0.140 |
| ARI seed 1 | 0.127 | 0.132 | 0.139 |
| mean of per-cancer fold PCC, seed 0 | 0.309 | 0.344 | 0.333 |
| mean of per-cancer fold PCC, seed 1 | 0.332 | 0.338 | 0.336 |

Per cancer ARI (seed 0 / seed 1): IDC E 0.208/0.221, full 0.212/0.219, H-Opt 0.299/0.293; COAD E 0.376/0.371, full 0.345/0.362,
H-Opt 0.380/0.356; CCRCC E 0.074/0.071, full 0.078/0.079, H-Opt 0.074/0.077. No paired difference is significant
(all-sample Wilcoxon p between 0.06 and 0.73). IDC and COAD have only 4 samples each.

PCC = official HEST metric: per-gene Pearson over pooled test spots, averaged over the 50 genes, per fold.


## 1b. Which HEST slides trained the heads and the graph model (script 53, verified 2026-10-09)

**Scope and vocabulary.** Two training protocols appear in this repository:

- *Cohort training* (paper Table III/V, `docs/METHODS.md` section 5): per cancer, trained on the HEST-1K cohort slides outside the held-out HEST
  test partition (about 200 slides across the three cancers), tested on the held-out HEST test partition.
- *Benchmark-fold training* (this index: scripts 22 and 35, the Leiden results, the saved frozen models, the TCGA predictions): trained only on the
  official benchmark's own training slides of each fold, tested on that fold's test slides.

Every number in this index is a benchmark-fold-training result. The checkpoints of the cohort-trained models are not stored in this repository, so
the TCGA predictions here come from benchmark-fold models; running the TCGA inference with the cohort-trained checkpoints requires those checkpoints.

`leiden_official_hvg50*/training_samples_per_fold.csv` lists train / validation / test sample ids and spot counts for every fold (seed 0 and 1).
The ePathST-E head, the H-Optimus-0 head and the ePathST graph model of a fold are trained on the same samples: only the official HEST-Benchmark
training samples of that fold, never the wider 120-IDC / 58-COAD / 24-CCRCC cohort. All folds match the official split files exactly; no sample
and no patient (HEST metadata `patient` column) is in both train/val and test, except that TENX111 (COAD) has no patient id in the metadata, so
it cannot be checked.

| Cancer | Slides per fold (train + val) | Spots per fold (train + val) | Test slides per fold |
|---|---|---|---|
| IDC | 3 (2 + 1) | 14,775 - 32,531 | 1 |
| COAD | fold 0: 1 (train = val fallback, 5,079 spots); fold 1: 3 (2 + 1) | 5,079 - 10,572 | 3 / 1 |
| CCRCC | 20 (17 + 3) | 55,849 - 67,651 | 4 |

The validation slide (early stopping) is carved from the fold's training slides, so IDC heads effectively fit on 2 slides. The seed changes which
slide is validation, not the test slides. The whole benchmark has 32 samples and 125,407 spots (IDC 4, COAD 4, CCRCC 24). TCGA-BRCA predictions
therefore come from heads fit on 3 Xenium breast slides per fold. HEST patient metadata: `misc/hest_metadata/HEST_v1_*.csv` (project space).

## 2. Frozen models (for TCGA / survival)

Frozen, not trained here: distilled student encoder
`cptac_hoptimus0_distill_hest_pna/student_checkpoints/locked_encoder_seed0.pt` (3.39M params) and H-Optimus-0
(`hf_cache/huggingface/hub/models--bioptimus--H-optimus-0`, 4.54 GB, repo `bioptimus/H-optimus-0`).
Trained and saved per fold, in `models/{cancer}_fold{n}/`:
`epathst_e_patch_head.pt`, `epathst_full.pt` (whole `ResidualGraphPredictor`, 1.86M params), `hoptimus_patch_head.pt`,
`hoptimus_pca.npz` (mean, components), `standardizer.npz` (mu, sigma, eps), `meta.json` (genes, samples, configs).
Training set sizes: CCRCC 17 samples/fold, COAD 1-2, IDC 2. Seed 0 fold models are the default for TCGA; seed 1 bundles exist too.

```python
import sys, torch, numpy as np
sys.path.insert(0, ".../STS_src/src")            # whole-module pickles import the `epathst` package
d = ".../leiden_official_hvg50/models/IDC_fold0/"
head = torch.load(d + "epathst_e_patch_head.pt", map_location="cpu", weights_only=False).eval()
full = torch.load(d + "epathst_full.pt", map_location="cpu", weights_only=False).eval()
sd = np.load(d + "standardizer.npz")             # prediction (log1p) = z * (sigma + eps) + mu
```
Loading all bundles for one cancer: `epathst.tcga.prediction.load_bundles()`.

## 3. TCGA predictions (scripts 38, 40)

Frozen ePathST-E, ePathST and H-Optimus-0 predict each cancer's official 50 genes per pseudo-spot. TCGA-BRCA uses the IDC models,
TCGA-COAD the COAD models, TCGA-KIRC the CCRCC models; fold predictions are un-standardized and averaged.

**Script 40 (use this one):** Visium-style layout. Otsu saturation tissue mask on a <=2048 px thumbnail, hexagonal lattice with
100 um pitch (area per spot (sqrt(3)/2) d^2 = 8,660 um^2, ~115 spots/mm^2), kept where the 112 um field is >=50% tissue, each
patch read at 0.5 um/px (224 px) and kept if >=50% tissue; slides <200 spots or without native mpp skipped. One patch per spot,
neighbouring patches overlap ~10%, same as Visium training data. Patches are streamed, not stored.
Smoke test (6 slides): 17.8k-38k spots, 157-327 mm^2 tissue, spacing exactly 100 um, 119-275 s per slide.

**Run result (2026-10-07, seed-0 fold models, tag `hex100_leiden_official_hvg50`):** 1,962 slides, 1,931 ok (31,054,677 spots,
7.9 GB in 1,931 files; median 3.8 MB/file), 31 skipped, 0 failed. Per cancer ok/total: BRCA 1,030/1,045, COAD 413/429, KIRC
488/488. Skipped: 21 without native mpp in the file (BRCA 7, COAD 14), 10 with <200 spots or no tissue patches (BRCA 8, COAD 2).
The final audit (`AUDIT.json`, `qc_all.csv`, `AUDIT.log` in the tag folder) opened every ok file: 3 systems present, shape
(n_spots, 50), finite, 50 unique genes, spot count equal to the QC record; 0 problems. Per-slide status: `qc/<project>/<case>.json`.

**Output** `tcga/predictions/<tag>/<project>/<case>.npz`: `x_um, y_um, genes, n_folds, native_mpp, pitch_um` and
`pred__ePathST-E`, `pred__ePathST`, `pred__H-Optimus-0` (float16, n_spots x 50, log1p scale). QC per shard:
`qc_<project>_shard<k>.csv` (spots, tissue mm^2, spacing, seconds, status). Skipped/failed slides are recorded there.

```python
z = np.load(".../hex100_leiden_official_hvg50/TCGA-BRCA/TCGA-3C-AALI.npz")
genes, P = z["genes"], z["pred__ePathST"].astype(np.float32)   # (n_spots, 50)
patient_score = P.mean(0)                                       # per-gene patient mean for correlation / survival
```

**Script 38 (older):** the thinned, randomly sampled patches already on scratch (<=2,000 per slide, spacing 114-208 um, local graph
nearly empty). Only a 9-slide smoke test was run (`smoke_tcga/`); not recommended.

## 3b. TCGA companion data (script 50)

Everything for the correlation / receptor / survival analyses, for the 1,962-patient cohort. Layout, sources and the RNA selection
rule are in `companion/README.md`; per-file sizes and checksums in `manifests/DOWNLOAD_MANIFEST.csv`; coverage in `manifests/COVERAGE.csv`.
- RNA-seq: GDC STAR-Counts, **primary tumour only**, re-selected per patient (the old cohort manifest's RNA file ids were stale or the wrong
  sample type: 992/1,962 gone from the GDC, 60 not primary tumour). 1,962/1,962 md5-verified. Panel tables
  `rnaseq/processed/<project>_panel50_{tpm,log2tpm1}.csv`; all-gene matrices `<project>_star_all_genes.npz`. All 150 panel genes found;
  one documented HGNC rename (KIRC: H3F3A -> H3-3A), recorded in `<project>_panel_gene_mapping.csv`.
- TCGA-CDR (Liu 2018) `clinical/tcga_cdr/TCGA-CDR.csv`, CPE purity (Aran 2015) `purity/aran2015_cpe/cpe_aran2015.csv` (read `sample_code` as text),
  cBioPortal (PanCanAtlas + provisional) `clinical/cbioportal/<study>/`, GDC clinical cross-check `clinical/gdc/`.
- BRCA receptors are only in the provisional study `brca_tcga` (ER_STATUS_BY_IHC, IHC_HER2, HER2_FISH_STATUS, HER2_IHC_SCORE).
  ER positive/negative 769/226; HER2 positive/negative/excluded 188/719/138 (rule: positive = IHC or FISH positive; negative = IHC or FISH
  negative; equivocal/undefined excluded).
- Coverage (BRCA / COAD / KIRC): OS+time 1,044 / 427 / 488; PFI+time 1,044 / 427 / 486; stage known 1,034 / 417 / 485; CPE 1,036 / 428 / 487;
  OS events 146 / 96 / 163; PFI events 139 / 113 / 152. One COAD patient is missing from the CDR.

## 4. Caveats to carry into the paper

- **HEST density is not one number:** IDC old-ST ~190 um, Visium ~100 um, COAD samples ~10 um (suspicious for Visium; not
  checked). 100 um is the Visium reference.
- **Official panels are built from the benchmark's own samples** (IDC 4, COAD 4, CCRCC 24). 111/114 whole-transcriptome IDC cohort
  samples and 10/53 COAD lack some panel genes (table in `hest_panel_coverage/`), so exact-panel analysis is limited to the
  benchmark samples; the project's 191-sample cohort excludes the benchmark's Xenium samples (e.g. TENX149).
- **Seed 0 IDC fold 2:** the ePathST-E patch head collapsed (PCC 0.011, prediction spread 0.085); the graph model partly rescued
  ePathST (0.463). Not present in seed 1. Head training is unstable with 1-2 training samples; H-Optimus-0 also varies a lot
  across seeds in CCRCC (fold PCC from -0.02 to 0.30).
- **ePathST ~ ePathST-E** in most folds (PCC difference <=0.003 in 8/12 seed-0 folds): the zero-initialized graph correction
  barely moves with tiny training sets.
- **Old Leiden runs are not comparable:** script 32 (fold-selected HVG1000, project CV) finished IDC only (114 samples; ARI
  ePathST-E 0.141, ePathST 0.149) before an out-of-memory kill; script 33 is zero-shot H-Optimus-0 embeddings against each
  sample's own top-variance genes (191 samples, ARI 0.171). `ari_summary_all_models.csv` there is an IDC-only test of the merge
  script. The TENX149 ARIs differ between runs (ePathST-E 0.58-0.67, ePathST 0.61-0.66 across runs) and the figure's
  `.tmp_leiden_final_out` combines two runs; treat single-sample ARI as noisy.
- **Scratch** (HEST-1k benchmark copy, TCGA slides/patches/embeddings) is purged after 30 days.
- `configs/base.yaml` / `base.json` show as modified in git; that change is not from these runs.

## 5. Code (all under `STS_src/`)

The runs above were made with numbered scripts that have since been merged into three stage drivers, with the reusable code
moved into the `epathst` package unchanged. Old name -> current command and module:

| Run with | Now | Module |
|---|---|---|
| `35_leiden_official_hvg50.py` | `25_leiden_official_panel.py official` | `epathst.evaluation.leiden` |
| `37_leiden_cohort_official50.py` | `25_leiden_official_panel.py cohort` | `epathst.evaluation.leiden_cohort` |
| `45_verify_leiden_ari.py` | `25_leiden_official_panel.py verify RUN_DIR` | `epathst.audits.leiden_ari` |
| `46_leiden_reproducibility.py` | `25_leiden_official_panel.py reproducibility RUN_DIR` | `epathst.audits.leiden_reproducibility` |
| `24_tcga_case_discovery.py` | `24_tcga_validation.py cases` | `epathst.tcga.cases` (GDC helpers in `epathst.tcga.gdc`) |
| `25_tcga_download_slides.py` | `24_tcga_validation.py download` | `epathst.tcga.download` |
| `28_tcga_extract_patches.py` | `24_tcga_validation.py patches` | `epathst.tcga.patches` |
| `38_tcga_predict_hvg50.py` | `24_tcga_validation.py predict-sparse` | `epathst.tcga.prediction` (holds `load_bundles`, `predict_from_embeddings`) |
| `40_tcga_predict_hexgrid.py` | `24_tcga_validation.py predict` | `epathst.tcga.hexgrid` |
| `43_tcga_worker.py` | `24_tcga_validation.py worker` | `epathst.tcga.worker` |
| `44_tcga_audit.py` | `24_tcga_validation.py audit` | `epathst.tcga.audit` |
| `50_tcga_companion_download.py` | `24_tcga_validation.py companion` | `epathst.tcga.companion` |
| `41_hest_panel_coverage.py` | `26_hest_audits.py panel-coverage` | `epathst.audits.panel_coverage` |
| `42_frozen_weights_manifest.py` | `26_hest_audits.py frozen-weights RUN_DIR` | `epathst.audits.frozen_weights` |
| `53_training_samples_audit.py` | `26_hest_audits.py training-samples RUN_DIR` | `epathst.audits.training_samples` |
| `32`, `33`, `34`, `36`, `39` | `src/scripts/legacy/` (unchanged earlier variants) | |
| `48_profile_hex_slide.py` | `src/scripts/tools/profile_hex_slide.py` | |
| `51`, `52` (Fig. 3) | `figures/fig_pretraining_data.py`, `figures/fig_data_efficiency.py` | |

| SLURM / tests | Purpose |
|---|---|
| `slurm/run_tcga_worker.slurm`, `slurm/launch_tcga_workers.sh` | worker (queues its own successor at start); fleet top-up across partitions |
| `slurm/run_leiden_official_hvg50.slurm` | `sbatch ... --seed N --out-name NAME` |
| `slurm/run_tcga_hexgrid_4gpu.slurm` | `sbatch --array=0-2` = 12 shards, 4 GPUs per task; resumable |
| `tests/test_tcga_queue_claims.py` | no-duplicate-slide stress test of the claim queue |
| `tests/e2e_one_slide.py`, `tests/e2e_embed_equivalence.py` | GPU checks of the hex-grid pipeline against the stored predictions and embedding cache |
