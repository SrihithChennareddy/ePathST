# ePathST — Final HPC Execution Guide

Oct 9, 2026 · @Srihith

This is the single document Claude on the HPC follows to write and run all remaining code: audit the existing TCGA predictions, run the molecular screen, the survival models, the gene-bottleneck test (U3) and the program-transfer analysis (U4), then produce the figures and numbers for the paper. Nothing on HEST is retrained; all new work is CPU analysis of outputs that already exist.

## How to use this guide

Export this doc as Markdown, copy it to `STS_src/FINAL_GUIDE.md` on the HPC, start Claude Code in `STS_src/`, and say:

> "Read FINAL\_GUIDE.md in full. Do Step 0, then stop and show me."

After each step's **CHECK**, the user replies "approved, do Step N". Never start the next step without that reply.

**Context for Claude on the HPC**

- Repo: `/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src`. Config: `configs/base.yaml`.
- SLURM header to copy: `slurm/run_official_hest_ablation_grid.slurm` (account `pn_alyu012826n`). This guide needs CPU only: request a CPU partition, 8–16 cores, 64 GB RAM, unless a step says otherwise.
- Existing scripts you will reuse, not rewrite: `src/scripts/35_leiden_official_hvg50.py` (HEST runs and heads), `src/scripts/38_tcga_predict_hvg50.py` (`predict_from_embeddings`), `src/scripts/43_tcga_worker.py` (TCGA inference driver), script 10 (`train_patch_head`).
- Existing outputs: HEST same-head runs in `results/leiden_official_hvg50/` (seed 0) and `results/leiden_official_hvg50_seed1/` (seed 1); TCGA predictions for ePathST, ePathST-E and H-Optimus-0 (locate them in Step 1).
- Python packages needed: numpy, pandas, pyarrow, scipy, statsmodels, scikit-learn, scikit-survival (≥ 0.22), lifelines, pingouin, matplotlib. Install any missing one into the project environment with `pip install --user`, and record versions.

**Rules for Claude on the HPC**

1. **Git:** create branch `final-paper`; commit after each step with message `Step N: <what>`. Never force-push.
2. **Never modify or delete existing outputs.** All new outputs go under `results/tcga_final/` (layout in Step 10).
3. **Freeze before looking.** Do not compute any correlation, hazard ratio, C-index or survival curve until Step 0's `configs/analysis_plan.json` is committed. Steps 1–3 may read RNA and clinical tables only to count and match, never to relate them to predictions.
4. **No HEST training.** Inference with saved models is allowed only where this guide says so.
5. **Smoke test first:** run each new script with `--smoke` (20 patients per cancer, 2 CV repeats) and show the output before the full run.
6. **Seeds:** use the seeds given in the plan file; never change a seed after seeing a result.
7. **Report everything:** every gene, cancer and system appears in every output table, including null, negative and unavailable results. Print n beside every statistic.
8. **Deviations:** if the data force any change to this guide, stop, explain, and on approval append a dated entry to `results/tcga_final/DEVIATIONS.md` (what changed, why, whether outcomes had been seen).
9. **Environment:** write `pip freeze` and `python --version` to `results/tcga_final/ENVIRONMENT.txt` at Step 0.
10. **Code style:** shared statistics go in one module, `src/epathst/analysis/stats_utils.py` (Appendix B), with unit tests in `tests/test_stats_utils.py`. Every script takes `--plan configs/analysis_plan.json` and `--out results/tcga_final`, logs to `results/tcga_final/logs/<script>.log`, and is deterministic.

**New scripts to write** (first confirm numbers 50–58 are unused in `src/scripts/`; if not, use the next free block and update this table in `README.md`)

| Script | Step | Reads | Writes |
| --- | --- | --- | --- |
| `50_tcga_audit.py` | 1 | predictions, slide metadata, HEST split files | `audit/` |
| `51_tcga_cohort.py` | 2 | GDC RNA, TCGA-CDR, CPE purity, receptor data | `cohort/` |
| `52_tcga_features.py` | 3 | predictions, embeddings, RNA | `features/` |
| `53_tcga_screen.py` | 4 | `features/`, `cohort/` | `screen/` |
| `54_tcga_survival.py` | 5 and 6 | `features/`, `cohort/` | `survival/` |
| `55_program_transfer.py` | 7 | `screen/`, HEST per-gene results, Appendix A labels | `programs/` |
| `56_survival_exploratory.py` | 8 | `features/`, `cohort/`, `survival/` | `survival_exploratory/` |
| `57_hest_tables.py` | 9 | HEST run outputs | `hest/` |
| `58_figures.py` | 10 | all of the above | `figures/` |

## What is already fixed

The paper answers three questions. HEST supplies existing evidence for the first two; this guide produces the TCGA evidence for the third plus two analyses (U3, U4) that explain it.

| Question | Evidence | Source |
| --- | --- | --- |
| 1. Compression: does a 3.39M student keep the 1.13B teacher's molecular signal? | Same-head HEST runs (2 seeds); TCGA screen; U3 | Existing + Steps 4, 6 |
| 2. Context: what does the GNN add, and for which genes? | Same-head HEST runs; ePathST vs ePathST-E on TCGA; U4 | Existing + Steps 4, 7 |
| 3. Translation: do frozen predictions match new patients' RNA and carry prognostic information? | TCGA screen and Cox models | Steps 4–5, 8 |

**Systems (labels used in every file)**

| Label | System | Head | Notes |
| --- | --- | --- | --- |
| `E` | ePathST-E: distilled ConvNeXt V2-Atto student (3.39M), frozen | MLP patch head (script 10 `train_patch_head`) | Student, no context |
| `P` | ePathST: student + scale-aware multiscale PNA GNN (residual correction; 5.25M total) | MLP patch head + GNN | Student with context |
| `T` | H-Optimus-0 teacher (1.13B), frozen; features PCA 1,536 → 320 fit on each fold's training spots | Same MLP patch head as E | Teacher, no context; there is no teacher + GNN system |
| `B` | Measured bulk RNA, log2(TPM + 1), same 50 genes | none | Sequencing reference |

Fair pairs: **E vs T** = same head, different encoder (the compression question). **P vs E** = context added to the same student (the context question). **P vs T** = full student system vs teacher patch-only; always label it that way.

**Confirmed inputs**

- Training: encoder distilled on all of CPTAC (363 patients); heads and GNN trained on HEST, per official fold (IDC 4 folds, COAD 2, ccRCC 6), same official 50-gene panels, same train/validation split and standardizer for E, P and T. Verified on the HPC (2026-10-09): the saved heads, the 1,931 TCGA predictions and the Leiden runs were trained on the official benchmark's own training slides only (1–20 per fold), not the \~200-slide cohort behind the paper's old numbers (those checkpoints are not on disk). The stored student was distilled from a 4,608-D teacher target, not the 1,536-D H-Optimus-0 output the paper describes: identify exactly what that target was (e.g., concatenated token summaries) before writing Methods. RNA comes from a frozen tumour piece while slides are FFPE diagnostic sections: state this as a limitation. TCGA slides, patches and the 108 GB embedding cache are on scratch and will be purged around Oct 31 – Nov 6: copy the embeddings and all prediction files to persistent storage now.
- H-Optimus-0 head files: `results/leiden_official_hvg50{,_seed1}/models/{cancer}_fold{n}/hoptimus_patch_head.pt`, with `hoptimus_pca.npz` and `standardizer.npz` in the same folder. All 12 bundles reload.
- TCGA predictions: fold-averaged ensembles from `38_tcga_predict_hvg50.py` via `43_tcga_worker.py`. BRCA has 1,030 slides predicted.
- Same-head HEST results (equal-weighted mean over cancers of per-fold PCC): seed 0 — E 0.309, P 0.343, T 0.333; seed 1 — E 0.332, P 0.338, T 0.336.

**Decisions already made**

- Cancers and heads: TCGA-BRCA → IDC head; TCGA-COAD → COAD head; TCGA-KIRC → ccRCC head.
- BRCA primary population: all histologies with a diagnostic slide. IDC-only is a sensitivity analysis.
- Endpoints (TCGA-CDR): BRCA PFI (primary), COAD PFI, KIRC OS.
- No spatial tumour–stroma contrast, no CellViT, no new HEST controls in this round (future work).

**Known caveats (must appear in the paper)**

- ePathST-E's head collapsed on IDC fold 2, seed 0 (resolved by the rule in Step 1).
- Context gain P − E is +0.034 (seed 0) vs +0.006 (seed 1); seed 0's gain may partly be the GNN correcting the collapsed head.
- The paper's old Table III/V values (0.3687, 0.3438, H-Optimus-0 0.3913 linear probe) are not traceable to a logged run; Step 9 handles them.

## Step 0 — Pre-registration

Goal: fix every analysis choice in one committed file before any outcome is computed, so the paper can state "all genes, features, endpoints, models and tests were fixed in commit X on date Y".

**0.1** Create `configs/analysis_plan.json` with exactly this structure (values shown are the frozen choices; fill the two `TO_FILL_STEP1` values after Step 1 and commit again before Step 4 — those are input facts, not outcome choices).

```json
{
  "version": "2026-10-09",
  "systems": {"E": "ePathST-E", "P": "ePathST", "T": "H-Optimus-0 same MLP head", "B": "bulk RNA"},
  "cohorts": {"BRCA": {"head": "IDC", "endpoint": "PFI", "horizon_years": 5},
              "COAD": {"head": "COAD", "endpoint": "PFI", "horizon_years": 3},
              "KIRC": {"head": "CCRCC", "endpoint": "OS", "horizon_years": 5}},
  "brca_population": {"primary": "all_histologies", "sensitivity": "Infiltrating Ductal Carcinoma"},
  "panels": "TO_FILL_STEP1: ordered 50-gene list per head, read from checkpoint/config",
  "tcga_ensemble": "TO_FILL_STEP1: seed(s) used and fold models excluded by the collapse rule",
  "collapse_rule": "fold model collapsed if validation PCC (mean over genes) < 0.05 OR median per-gene prediction SD across validation spots < 0.01 x target SD; excluded from the TCGA ensemble for every system sharing that fold head",
  "slides": {"type": "DX only", "per_patient": "DX1, else lowest DX number", "min_spots": 200},
  "rna": {"source": "GDC STAR-Counts tpm_unstranded", "transform": "log2(TPM+1)", "sample_type": "01", "aliquot_rule": "01A first, then lowest vial/portion code"},
  "clinical": {"source": "TCGA-CDR", "primary_covariates": ["age_per_10y", "stage_III_IV"],
               "sensitivity_covariates": {"BRCA": ["ER", "HER2"], "KIRC": ["grade"], "COAD": []},
               "missing": "complete case"},
  "aggregation": {"primary": "mean over all retained spots, un-standardized scale", "sensitivity": "median"},
  "screen": {"stat": "spearman", "ci": "patient bootstrap 2000, percentile", "seed": 0,
             "p": "t-approximation (n>=100) else 10000 permutations", "fdr": "BH per system over 150 gene-cancer pairs",
             "purity": "CPE partial spearman", "specificity": "matched-gene rank among 50 bulk genes"},
  "survival": {"model": "CoxnetSurvivalAnalysis", "l1_ratio": 0.5, "n_alphas": 100, "alpha_min_ratio": 0.01,
                "clinical_penalty": 0, "inner_cv": 5, "outer_cv": 5, "repeats": 5, "repeat_seeds": [0,1,2,3,4],
                "primary_metric": "Uno C at horizon", "secondary": ["Harrell C", "cumulative dynamic AUC at horizon", "integrated Brier"],
                "test": "Nadeau-Bengio corrected resampled t, df=24, two-sided"},
  "feature_sets": ["C", "C+E", "C+P", "C+T", "C+HS", "C+HT", "C+B", "C+B+P"],
  "bottleneck_U3": {"embedding": "patient mean of frozen embeddings", "pca_components": 32, "pca_sensitivity": [16, 64], "pca_seed": 0},
  "programs_U4": "Appendix A labels, frozen",
  "primary_test": {"cohort": "BRCA", "comparison": "C+P vs C", "metric": "delta Uno C", "alpha": 0.05},
  "families": "see Testing hierarchy section"
}
```

**0.2** Copy the Appendix A program labels and marker groups into `configs/gene_programs.json`. Check every gene name against the panels found in Step 1; a label for a gene not in its panel is an error to fix now.

**0.3** Write `results/tcga_final/ENVIRONMENT.txt`. Create `results/tcga_final/DEVIATIONS.md` with the header "No deviations as of \<date>".

**0.4** Ask the user, and record the answer in the JSON under `"prior_looks"`: has anyone already computed any TCGA correlation, hazard ratio, C-index or survival curve from these predictions? If yes, list exactly which; those analyses are labeled exploratory.

**0.5** `git add configs/ results/tcga_final/ && git commit -m "Step 0: pre-registration"`. Print the commit hash.

**CHECK:** show the JSON, the program-label file and the commit hash.

## Step 1 — Audit (`50_tcga_audit.py`)

Goal: prove the inputs are what the paper will say they are. This step reads predictions, embeddings, slide metadata and HEST split files only. Output: `results/tcga_final/audit/AUDIT_REPORT.md` plus one CSV per check.

| # | Check | How | Pass rule | Why |
| --- | --- | --- | --- | --- |
| A1 | Inventory | List every TCGA prediction and embedding file per system × cancer: path, format, keys/columns, array shapes, dtype, size on disk | E, P, T found for BRCA, COAD, KIRC | Paper reports real counts |
| A2 | Slide type | Parse barcodes (`TCGA-XX-XXXX-01Z-00-DX1`) | Only `-DX` slides; zero `-TS`/`-BS` | Frozen sections differ from FFPE |
| A3 | One slide per patient | Patient = first 12 characters | No duplicate patients after the DX1 rule | Patient is the unit |
| A4 | Panels | Read the ordered gene list from each head's checkpoint/config; compare with the gene axis stored in the predictions | Identical lists and order for E, P, T within each cancer | Silent column misalignment would corrupt everything |
| A5 | Scale | Per gene: mean and SD across all TCGA spots, per system; compare with the HEST standardizer means/SDs | Documented as standardized or original scale; same for all systems | Ensemble averaging and interpretation depend on it |
| A6 | Ensemble and seed | From `43_tcga_worker.py` arguments, logs and file names: which seed's fold models were averaged, and how | Documented exactly | Needed for the collapse decision |
| A7 | Collapse rule | Apply the Step 0 rule to every fold head (E, P, T; seeds 0 and 1) using HEST validation predictions or logs | Table of fold × seed × system with validation PCC, prediction SD, collapsed yes/no | Decides which fold models enter the TCGA ensemble |
| A8 | Same spots | For each slide, compare spot ID/coordinate sets across E, P, T | Identical (else use the intersection for all, documented) | Differences must come from models, not tissue |
| A9 | HEST training sets | For every cancer × fold × seed: list HEST sample IDs used for head training, validation and test; map to patients where HEST metadata allows | Counts per fold; no patient in both train and test | Methods must state training size; leakage check |
| A10 | Resolution | Slide mpp from metadata; confirm patches were resampled to 0.5 µm/px, 224 px | All slides; slides lacking mpp listed | Field of view must match HEST (112 µm) |
| A11 | Grid | Median nearest-neighbour distance between spot centres | About 100 µm | Graph scales mean the same as on HEST |
| A12 | Graph (P only) | Degree distribution per scale (0–150, 150–300, 300–600 µm); share of isolated nodes; whether inference ran on tiles and, if so, the tile halo | No NaN; isolated share reported; halo ≥ 1,200 µm if tiled | Two PNA layers × 600 µm need \~1,200 µm of context |
| A13 | Coverage and numerics | Spots per slide (median, IQR); NaN/Inf count; genes with near-zero variance across patients | Zero NaN/Inf; slides < 200 spots flagged | Noisy or broken patient means |
| A14 | Embeddings for U3 | Do per-spot student (320-D) and teacher embeddings exist for the same spots? Teacher raw 1,536-D or only PCA-320? | Documented; if absent, U3 needs embedding extraction (inference only, ask the user first) | Step 6 depends on it |
| A15 | HEST per-gene results for U4 | Do the HEST runs store per-gene PCC per fold and seed for E, P, T, or the per-spot test predictions to compute them? | Documented | Step 7 depends on it |

**After the audit:**

1. If A7 finds a collapsed fold model inside the ensemble used for TCGA, rebuild the affected TCGA ensembles without it, for every system sharing that fold (inference/averaging only, no training). Write them to `results/tcga_final/predictions_rebuilt/`, never over the originals.
2. Fill `panels` and `tcga_ensemble` in `configs/analysis_plan.json` and commit (`Step 1: input facts`).

**CHECK:** show `AUDIT_REPORT.md` with every check's result. Any failed check stops the guide until the user decides.

## Step 2 — Cohort assembly (`51_tcga_cohort.py`)

Goal: one row per patient per cancer with slide, RNA, clinical, purity and receptor fields joined on the 12-character barcode. Outcomes are joined but not analysed here. Output: `results/tcga_final/cohort/{BRCA,COAD,KIRC}_patients.csv` and `consort_{cancer}.csv`.

**2.1 Bulk RNA (GDC)**

- Files: GDC STAR-Counts `*.rna_seq.augmented_star_gene_counts.tsv`, column `tpm_unstranded`; skip the 4 summary rows starting with `N_`.
- Keep sample type code `01` (characters 14–15 of the sample barcode). One sample per patient: `01A` first, then the lowest vial letter, then the lowest portion number. Log every patient with more than one candidate and which was kept.
- Value: log2(TPM + 1), stored for the panel genes only (matching in Step 3).

**2.2 Clinical and outcomes (TCGA-CDR)**

- File: TCGA-CDR supplementary table S1 (Liu et al., Cell 2018), sheet `TCGA-CDR`. Columns used: `bcr_patient_barcode`, `type`, `age_at_initial_pathologic_diagnosis`, `ajcc_pathologic_tumor_stage`, `histological_type`, `histological_grade`, `OS`, `OS.time`, `PFI`, `PFI.time`. Verify the exact spellings in the file and log any difference.
- Age: years / 10 (so hazard ratios are per decade).
- Stage group: `Stage I`, `IA`, `IB`, `II`, `IIA`, `IIB`, `IIC` → 0 ("I–II"); `Stage III*`, `Stage IV*` → 1 ("III–IV"); `Stage X`, `[Not Available]`, `[Discrepancy]`, `[Unknown]`, empty → missing.
- KIRC grade: `G1`–`G4` → 1–4; `GX` or missing → missing.
- Times are in days; keep days in files, convert to years only for plots and horizons (365.25 days per year).

**2.3 Tumour purity:** CPE from Aran, Sirota & Butte (Nat Commun 2015), Supplementary Data 1, matched on the 15-character sample barcode of the chosen RNA sample. Missing purity excludes a patient only from purity-adjusted analyses.

**2.4 BRCA receptors (sensitivity covariates and subgroups):** ER status by IHC, HER2 by IHC and FISH, from the TCGA BRCA clinical supplement (or cBioPortal `brca_tcga_pan_can_atlas_2018`). HER2 positive = IHC 3+ or FISH positive; negative = IHC 0/1+, or IHC 2+ with FISH negative; otherwise missing. Print the raw value counts before mapping.

**2.5 Eligibility, in this order (each row of the CONSORT table counts patients removed):**

1. Candidate patients = all patients of the project with any predicted DX slide.
2. Remove patients whose slide failed Step 1 (no mpp, < 200 spots, QC failure).
3. Flag (do not remove) patients without a `01` RNA sample: they stay in survival analyses of predicted features, but not in the screen or in any set containing B.
4. Survival-eligible = endpoint and time present, time > 0, age and stage present.
5. Record histological type for the BRCA IDC-only sensitivity analysis.

**2.6 Overlap check:** confirm no TCGA patient appears among HEST-1k or CPTAC samples (match on any TCGA barcode recorded in HEST metadata). Report the count (expected 0).

**2.7 Follow-up summary per cancer:** n eligible, events, event rate, median follow-up by reverse Kaplan–Meier (censoring treated as the event), and the share of patients still at risk at 3 and 5 years.

**CHECK:** show the three CONSORT tables, event counts, median follow-up and the at-risk shares. If BRCA has fewer than 80 PFI events, stop: the user decides (counts only, no results seen) whether BRCA stays primary, and logs it in `DEVIATIONS.md`.

## Step 3 — Patient-level features (`52_tcga_features.py`)

Goal: one 50-value vector per patient per system, one embedding vector per patient per encoder, and the matching bulk RNA table. Output in `results/tcga_final/features/`.

**3.1 Predicted genes (E, P, T)**

- Input: per-spot predictions after Step 1 (rebuilt ensembles where the collapse rule applied).
- If predictions are standardized, un-standardize per fold before averaging folds: value = z × sd\_fold + mean\_fold (from each fold's `standardizer.npz`). If the stored values are already a fold average on the original scale, use them as they are and say so in the log.
- Patient value per gene = arithmetic mean over all retained spots of that patient's slide. Why the mean: bulk RNA is extracted from a homogenized tissue piece, the closest analogue of an average over tissue. Why all spots: choosing "tumour-rich" spots by predicted tumour genes would reuse the predictions to select themselves.
- Also compute the per-gene median (sensitivity) and per-gene SD across spots (supplement only).
- Files: `{cancer}_{E|P|T}_mean.parquet`, `{cancer}_{E|P|T}_median.parquet`, `{cancer}_{E|P|T}_sd.parquet`; rows = patient barcode, columns = genes in panel order.

**3.2 Embeddings for U3 (HS, HT)**

- Student: per-spot 320-D frozen embedding → patient mean → `{cancer}_HS.parquet` (320 columns).
- Teacher: per-spot raw 1,536-D H-Optimus-0 embedding → patient mean → `{cancer}_HT.parquet`. If only the PCA-320 features were saved, use those, name the file `{cancer}_HT320.parquet`, and record it as a deviation.
- Same spots as 3.1. No PCA here; PCA happens inside cross-validation in Step 5.

**3.3 Gene matching to bulk RNA**

- Map panel symbols to Ensembl gene IDs with the HGNC complete set (`hgnc_complete_set.txt`: `symbol`, `prev_symbol`, `alias_symbol`, `ensembl_gene_id`). Strip version suffixes from GDC `gene_id` (`ENSG00000141736.14` → `ENSG00000141736`). Drop `_PAR_Y` rows.
- Output `gene_matching.csv`: cancer, panel gene, Ensembl ID, GDC gene\_name, matched (yes/no), match route (symbol / previous symbol / alias).
- Unmatched genes stay in every table with status `unavailable`.
- Bulk file: `{cancer}_B.parquet` with the same gene columns and order as the predictions.

**3.4 Descriptive gene properties (needed in Step 7)**

Per cancer and gene, from bulk RNA of screen-eligible patients: mean log2(TPM + 1), SD, and the share of patients with TPM = 0. Save `gene_properties.csv`. These are covariates for "is a gene predicted better just because it is highly expressed or variable?".

**CHECK:** show the shapes of every file, the gene-matching table, and the gene properties summary. No correlations yet.

## Step 4 — Molecular transfer screen (`53_tcga_screen.py`)

Question: across patients, is higher predicted expression of gene g matched by higher measured RNA of gene g? This tests transfer to new patients, assay and hospitals; it says nothing about where in the slide a gene is expressed. Patients: screen-eligible (slide + RNA). Output in `results/tcga_final/screen/`.

**4.1 Per-gene correlation** — for each cancer × gene × system (E, P, T):

- ρ = Spearman(patient-mean prediction, log2(TPM + 1)) via `scipy.stats.spearmanr`. Why Spearman: scales differ and the relation may be non-linear; ranks only ask whether patients are ordered the same way, and resist outlier patients.
- 95% CI: patient bootstrap, 2,000 resamples, percentile interval, `np.random.default_rng(0)`. Resample patient indices once per replicate and reuse the same indices for E, P and T (needed for 4.4).
- p: two-sided from `spearmanr` when n ≥ 100; otherwise a 10,000-permutation p (shuffle RNA across patients).
- q: Benjamini–Hochberg over the 150 gene–cancer pairs, separately per system: `statsmodels.stats.multitest.multipletests(p, method="fdr_bh")`. Unavailable genes are excluded from the family and the family size is printed. P's family is the key secondary family; E and T are reference families.

**4.2 Purity-adjusted correlation**

- Partial Spearman of prediction vs RNA given CPE purity, on patients with purity: rank-transform all three, regress the prediction ranks and RNA ranks each on purity rank (OLS with intercept), Pearson r of the two residual vectors; p from t = r√((n − 3)/(1 − r²)), df = n − 3. (`pingouin.partial_corr(..., method="spearman")` gives the same; use it as a cross-check in the unit tests.)
- BH over 150 again, per system. Why: tumour-enriched genes all rise with tumour fraction; a model that only learned "how much tumour" would correlate with all of them. The partial correlation asks whether gene-specific signal survives at fixed purity.

**4.3 Specificity control**

- For each predicted gene g, compute ρ with the measured RNA of all available panel genes h in that cancer (a 50 × 50 matrix per system).
- matched\_rank(g) = rank of ρ(g, g) within the row, 1 = highest. margin(g) = ρ(g, g) − median over h ≠ g of ρ(g, h).
- Per cancer × system report: median matched rank (null ≈ 25.5), share of genes with matched rank ≤ 5 (null 10%), median margin, and a one-sample Wilcoxon of margins vs 0 over genes.
- Why: a gene-specific negative control. High ρ with the wrong genes means the model reads tissue composition, not individual genes.

**4.4 System comparison (descriptive)**

- Per gene: Δρ(P − E), Δρ(E − T), Δρ(P − T), each with a paired bootstrap CI from the shared resamples in 4.1.
- Per cancer: median Δρ over genes and a Wilcoxon signed-rank test over genes, labeled descriptive (genes are correlated, so 50 genes are not 50 independent observations).

**4.5 Output table** `screen_full.csv` — one row per cancer × gene × system, all 450 rows:

| column | meaning |
| --- | --- |
| cancer, gene, system | identifiers |
| n, n\_purity | patients used |
| rho, ci\_low, ci\_high, p, q | 4.1 |
| prho, p\_prho, q\_prho | 4.2 |
| matched\_rank, margin | 4.3 |
| status | ok / unavailable / zero-variance |

Plus `specificity_summary.csv` and `system_deltas.csv`.

**Rules for writing it up:** best-correlating genes may be highlighted only as "selected after inspection", with the full table in the supplement. Literature can explain a gene's relevance (e.g., ERBB2) but does not validate the prediction.

**CHECK:** per cancer × system: genes with q < 0.05, median ρ, median partial ρ, specificity summary, and the system-delta medians.

## Step 5 — Prognosis with penalized Cox models (`54_tcga_survival.py`)

Question: does slide-predicted expression add prognostic information beyond age and stage, and how does that compare with the teacher and with sequencing the same 50 genes? Expression models are frozen; the Cox model is trained on TCGA outcomes inside cross-validation, so never call this "zero-shot". Output in `results/tcga_final/survival/`.

**5.1 Feature sets (per cancer)**

| Set | Features | Patients |
| --- | --- | --- |
| C | age\_per\_10y, stage\_III\_IV | survival-eligible |
| C+E, C+P, C+T | C + 50 predicted gene means of that system | survival-eligible |
| C+HS, C+HT | C + 32 PCA components of patient-mean embeddings (Step 6) | survival-eligible |
| C+B | C + 50 measured genes | survival-eligible with RNA |
| C+B+P | C + measured + ePathST genes (102) | survival-eligible with RNA |

All sets are fitted on the full survival-eligible cohort, and C, C+P and C+B are also refitted on the RNA subset so that every comparison involving B is on the same patients.

**5.2 Folds** — created once and saved as `cv_folds.csv` (patient, repeat, outer\_fold), separately for the full and the RNA-subset cohort:

- `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=r)` on the event indicator, for r in 0–4 → 25 outer splits.
- Every feature set uses these same splits. Why: comparisons become paired, removing split luck from the differences.

**5.3 Fitting inside one outer split** (training part = TR, test part = TE):

1. Standardize every feature with TR's mean and SD (`StandardScaler` fit on TR only); apply to TE. For HS/HT, PCA happens here too (Step 6).
2. **C alone:** `sksurv.linear_model.CoxPHSurvivalAnalysis(alpha=1e-4, ties="efron")` (tiny ridge only for numerical stability).
3. **Penalized sets:** `CoxnetSurvivalAnalysis(l1_ratio=0.5, n_alphas=100, alpha_min_ratio=0.01, penalty_factor=pf, fit_baseline_model=True, max_iter=100000)`, where pf = 0 for the two clinical columns and 1 for all others. Why elastic net: the 50 genes are strongly correlated; pure lasso picks one of a correlated group almost at random, while l1\_ratio 0.5 keeps groups together and still removes noise. Why clinical unpenalized: the question is what genes add on top of age and stage, so those must always stay in.
4. **Choose α by inner CV:** fit once on all of TR to get the α path (`alphas_`). Make 5 inner folds of TR (`StratifiedKFold(5, shuffle=True, random_state=100 + r)` on events). For each α in the path, fit on 4 inner folds with `alphas=[α]` and score Harrell's C on the 5th; average over inner folds. Pick the α with the highest mean; ties → the larger α (simpler model).
5. Refit on all of TR with `alphas=[best α]`; predict risk for TE with `model.predict(X_TE)`.
6. If the solver fails to converge, raise `alpha_min_ratio` to 0.05 for that fit and log it.

**5.4 Metrics on each TE (25 values per set)**

- **Uno's C (primary):** `sksurv.metrics.concordance_index_ipcw(y_TR, y_TE, risk, tau=τ)[0]`. Why Uno: Harrell's C depends on each cohort's censoring pattern; Uno's C reweights pairs by the inverse probability of staying uncensored (estimated from TR), so it targets the same quantity regardless of censoring.
- **τ:** 5 years BRCA and KIRC, 3 years COAD, in days. If any TE has under 10% of patients still at risk at τ, or τ exceeds TR's largest follow-up time, lower τ for that cancer to the largest whole year that satisfies both in every split, and log it. This depends on follow-up only, never on which model wins.
- Harrell's C (`concordance_index_censored`), time-dependent AUC at τ (`cumulative_dynamic_auc`), integrated Brier score from 1 year to τ using `predict_survival_function` (supplement).

```latex
C_{\tau} = \frac{\sum_{i}\sum_{j} \Delta_i \, \hat G(T_i)^{-2} \, \mathbf{1}(T_i < T_j,\; T_i < \tau) \, \mathbf{1}(\hat\eta_i > \hat\eta_j)}{\sum_{i}\sum_{j} \Delta_i \, \hat G(T_i)^{-2} \, \mathbf{1}(T_i < T_j,\; T_i < \tau)}
```

**5.5 Paired comparisons**

- In each split: ΔC = C(set A) − C(set B). Mean and SD over the 25 splits.
- Test: Nadeau–Bengio corrected resampled t-test, df = 24, two-sided. Why: the 25 splits reuse patients, so a plain t-test would give p-values far too small.

```latex
t = \frac{\overline{\Delta C}}{\sqrt{\left(\frac{1}{J} + \frac{n_{\text{test}}}{n_{\text{train}}}\right)\hat\sigma^2_{\Delta C}}}, \qquad J = 25,\; \frac{n_{\text{test}}}{n_{\text{train}}} \approx 0.25
```

- Corrected 95% CI: mean ± t(0.975, 24) × the denominator above.
- **Primary (H1):** BRCA, C+P vs C.
- **Secondary family (BH across all of these):** C+P vs C in COAD and KIRC; C+P vs C+E; C+E vs C+T; C+P vs C+T; C+B vs C; C+B+P vs C+B (RNA subset) — each in all three cancers.
- **Complementary pooled CI:** within each TE convert risks to percentiles (rank / (n\_TE + 1)); average each patient's percentile over the 5 repeats; Uno's C of the averaged score on all patients; 1,000 patient bootstrap replicates (same draws for compared sets) for CIs of C and ΔC. Label "conditional on the fitted models".
- **Share of sequencing's value:** ΔC(C+P vs C) / ΔC(C+B vs C) on the RNA subset, only if the bulk ΔC's corrected CI excludes 0.

**5.6 Independence and proportional hazards**

- Fit `lifelines.CoxPHFitter` on all survival-eligible patients: endpoint \~ age\_per\_10y + stage\_III\_IV + averaged out-of-fold P percentile score (z-scored). Report HR per SD, 95% CI, Wald p, and the likelihood-ratio test vs age + stage.
- `proportional_hazard_test(fitted, data, time_transform="rank")` for each covariate. If p < 0.05 with a visible trend in scaled Schoenfeld residuals, report it and refit with that covariate as a stratum as a sensitivity analysis.

**5.7 Sensitivity analyses (supplement):** add the cancer-specific covariates (BRCA ER, HER2; KIRC grade); median instead of mean aggregation; BRCA IDC only; add spots-per-patient as a covariate.

**Outputs:** `cv_folds.csv`, `cindex_by_split.csv` (set × cancer × split × metric), `delta_c.csv` (comparison, mean, corrected CI, p, q), `pooled_ci.csv`, `chosen_alpha.csv`, `coefficients.csv` (every fit's non-zero coefficients), `independence.csv`, `ph_tests.csv`.

**CHECK:** C-index table (all sets × cancers, mean ± SD), ΔC table with corrected CIs, p and q, independence HRs, PH results. State plainly which are null.

## Step 6 (U3) — Interpretable-bottleneck test (inside `54_tcga_survival.py`)

Question: when the image is squeezed into 50 named genes, how much prognostic information is kept compared with the raw image embedding it came from? If the 50 genes keep most of it, the gene layer is an interpretable bottleneck that costs little; if not, the genes discard prognostic information that the image holds. Both answers are findings.

**6.1 Features**

- HS = patient-mean student embedding (320-D); HT = patient-mean teacher embedding (1,536-D, or 320-D PCA features if that is all that exists — see Step 3.2).
- Inside each outer split, on TR only: `StandardScaler` → `sklearn.decomposition.PCA(n_components=32, svd_solver="full", random_state=0)`; transform TE with the TR-fitted scaler and PCA. Never fit PCA on all patients.
- Then C+HS and C+HT go through exactly the same Coxnet procedure as Step 5.3 (clinical unpenalized; inner-CV α).
- Why 32 components: of the same order as the 50 genes, so neither side wins by having many more inputs. Sensitivity: 16 and 64 components (supplement).

**6.2 Comparisons (family U3, BH across 4 comparisons × 3 cancers)**

| Comparison | What it asks |
| --- | --- |
| C+E vs C+HS | Same student encoder: do 50 predicted genes keep the prognostic value of the student embedding? |
| C+T vs C+HT | Same question for the teacher |
| C+P vs C+HS | Does adding tissue context to the genes close any gap to the raw student embedding? |
| C+HS vs C+HT | Raw embeddings: student vs teacher prognostic value (compression at the embedding level) |

All use the Step 5 splits, the Nadeau–Bengio test and corrected CIs.

**6.3 Retention ratio** (reported only when the denominator's corrected CI excludes 0):

```latex
R_{\text{student}} = \frac{\Delta C(\text{C+E vs C})}{\Delta C(\text{C+HS vs C})}, \qquad R_{\text{teacher}} = \frac{\Delta C(\text{C+T vs C})}{\Delta C(\text{C+HT vs C})}
```

CI by the pooled patient bootstrap of Step 5.5 (same draws for numerator and denominator).

**6.4 How to read the results**

| Result | Meaning for the paper |
| --- | --- |
| Genes ≈ embedding (ΔC CI within about ±0.02 and R near 1) | The 50-gene layer is an interpretable summary that keeps the image's prognostic signal |
| Embedding > genes | The image holds prognostic information beyond these 50 genes; the gene panel is the bottleneck, not the encoder |
| Genes > embedding | Supervision on expression concentrated useful signal (possible with few events, where 32 unlabeled components are noisier) |
| Both ≈ C | Neither adds beyond age and stage in this cohort; report as null |

A confidence interval that includes 0 does not prove equality; say "no detectable difference, CI \[a, b\]".

**Outputs:** rows for C+HS and C+HT in `cindex_by_split.csv` and `delta_c.csv`; `bottleneck_summary.csv` (cancer, comparison, ΔC, CI, p, q, R, R CI); PCA explained-variance per split in `pca_variance.csv`.

**CHECK:** show the U3 table and the explained variance of the 32 components.

## Step 7 (U4) — Which biology transfers (`55_program_transfer.py`)

Question: which kinds of genes can be read from H&E, which survive compression, and which gain from tissue context? Hypothesis to test, not assume: cell-identity and lineage genes (immune, stromal, epithelial markers) are predicted and transferred better than cell-state genes (interferon, hypoxia, proliferation, stress). Exploratory throughout. Output in `results/tcga_final/programs/`.

**7.1 Inputs**

- Labels: `configs/gene_programs.json` (Appendix A): program type (state / identity / other) and domain-marker groups. Frozen at Step 0.
- Gene-level outcomes, one value per cancer × gene:

| Outcome | Definition | Source |
| --- | --- | --- |
| O1 HEST accuracy | Per-gene test PCC, mean over folds, then over seeds 0 and 1, for E, P, T | HEST same-head runs (A15) |
| O2 TCGA transfer | Spearman ρ (Step 4.1) for E, P, T | `screen_full.csv` |
| O3 Purity-adjusted transfer | partial ρ (Step 4.2) | `screen_full.csv` |
| O4 Compression cost | T − E, on O1 and on O2 | derived |
| O5 Context gain | P − E, on O1 and on O2 | derived |

- If HEST per-gene PCC is not stored, compute it from the saved per-spot test predictions (same metric as the paper defines). If neither exists, run U4 on O2–O5 (TCGA) only and log it.
- O1 is reported twice: all folds, and excluding IDC fold 2 seed 0 (the collapsed head).
- Gene covariates: mean log2(TPM + 1), SD, zero-share (Step 3.4) for TCGA outcomes; for HEST outcomes, the same three computed from measured test-spot expression if available, else the TCGA values.

**7.2 Tests per outcome and system**

1. **Per cancer, program type:** Kruskal–Wallis across state / identity / other (`scipy.stats.kruskal`). Then identity vs state: Mann–Whitney U (two-sided) with rank-biserial effect size and a 2,000-replicate gene bootstrap 95% CI (seed 0). If either group has fewer than 8 genes (e.g., ccRCC identity has 6), report medians and the effect size only, no p-value.
2. **Pooled over cancers:** statistic = mean over cancers of \[median(identity) − median(state)\]; null by permuting labels within each cancer, 10,000 permutations, seed 0; two-sided p = (1 + #|null| ≥ |observed|) / 10,001. Why within-cancer permutation: cancers differ in overall accuracy, so pooling raw values would compare cancers, not programs.
3. **Is it just expression level?** OLS with HC3 standard errors (`statsmodels.formula.api.ols(...).fit(cov_type="HC3")`): `outcome ~ C(program, Treatment("identity")) + mean_expr + sd_expr + zero_share + C(cancer)`. Report the state coefficient and its CI. Why: highly expressed, variable genes are easier to predict; this separates biology from detectability.
4. **Domain-marker groups:** per cancer, median outcome per group and Kruskal–Wallis across groups (genes in no group are excluded; a gene listed in two groups is excluded from the test and noted).
5. **Does HEST accuracy predict TCGA transfer?** Spearman across the 50 genes between O1 and O2, per cancer and system. Why: if genes predicted well on HEST transfer well to TCGA, the benchmark is informative about real-world behaviour.

**7.3 Multiplicity:** BH across every p-value produced in 7.2 (family U4). All results are labeled exploratory.

**7.4 Outputs:** `gene_level_outcomes.csv` (cancer, gene, program, domain group, O1–O5 per system, covariates), `program_tests.csv` (test, cancer, outcome, system, statistic, effect size, CI, p, q), `hest_vs_tcga.csv`.

**7.5 Figure (for Fig. 4b):** per cancer, a strip-and-box plot of O2 (P) by program type with the identity–state effect size and CI annotated; a second panel with O4 and O5 by program type.

**CHECK:** show `program_tests.csv` sorted by q, the OLS table, and the HEST-vs-TCGA correlations.

## Step 8 — Exploratory survival (`56_survival_exploratory.py`)

These analyses explain the Step 5 result; they never replace or rescue it. All are labeled exploratory. Output in `results/tcga_final/survival_exploratory/`.

**8.1 Selection stability** — from `coefficients.csv` (Step 5): for C+P, C+T and C+B, each gene's share of the 25 outer fits with a non-zero coefficient and the sign of its mean coefficient. Stable driver = selected in ≥ 80% of fits with one sign. Why: one penalized fit's chosen genes are unstable; frequency across refits shows what is robust.

**8.2 Per-gene adjusted hazard ratios** — for each cancer × system (P, T, B) × gene: `CoxPHFitter` with age\_per\_10y + stage\_III\_IV + gene (z-scored over the cohort). Report HR per SD, 95% CI, Wald p; BH across the 50 genes within each cancer × system. Why separate small models: p-values read off a penalized model after selection are invalid.

**8.3 Agreement with sequencing** — per cancer: Spearman across genes between log-HR from P and log-HR from B; overlap of the top-10 genes by selection frequency (P vs B), with the hypergeometric expectation for random overlap.

**8.4 Biology of drivers** — tabulate stable drivers by program type and domain group (Appendix A). Correlated genes share credit, so describe programs, not single genes.

**8.5 Kaplan–Meier illustration** — repeat 0 only: in each outer split, threshold = median risk on TR; apply it to TE; pool the five TE groups. Plot KM curves for high vs low risk per system and cancer (`lifelines.KaplanMeierFitter`), numbers at risk at 0, 1, 2, 3, 4, 5 years under the axis, log-rank p (`logrank_test`), events / n per group. Why training-fold thresholds: a threshold chosen on all patients leaks test information and flatters the curves.

**8.6 Subgroups (pre-specified)**

| Cancer | Subgroup | Why | Clinical covariate inside it |
| --- | --- | --- | --- |
| BRCA | ER+ / HER2− | Genomic risk tests guide chemotherapy here | age |
| COAD | Stage II | Benefit of adjuvant chemotherapy is uncertain | age |
| KIRC | Stage III | Candidates for adjuvant therapy | age (+ grade in sensitivity) |

Run the Step 5 procedure for C vs C+P inside each subgroup. If a subgroup has fewer than 20 events, report only KM curves and event counts. Wording: "could help identify patients whose treatment decision is uncertain", never "changes treatment".

**8.7 Never do:** correlate any feature with raw survival time (it is censored); choose genes, cut-offs or endpoints by which curve separates most.

**CHECK:** stability table, per-gene HR forest data, log-HR agreement, KM figures, subgroup table with events.

## Step 9 — HEST reporting without rerunning (`57_hest_tables.py`)

Goal: every HEST number in the paper traces to a file, uses one metric definition, and states its sample size honestly. No training; reading outputs only. Output in `results/tcga_final/hest/`.

**9.1 Provenance search** — grep logs, results, SLURM outputs and `git log -p` for the old paper values: ePathST 0.5384 / 0.3302 / 0.2375; patch-only 0.4890 / 0.3151 / 0.2275; local 0.5161 / 0.3238 / 0.2316; meso 0.5237 / 0.3261 / 0.2328; macro 0.5194 / 0.3249 / 0.2341; multiscale 0.5272 / 0.3256 / 0.2337. If found, write `docs/reports/PAPER_RUN_PROVENANCE.md` (machine, commit, config, job ID, files) and fix the three repo passages that call the run undocumented (`DESIGN.md` §9.2, §14; `HEST_OFFICIAL_ABLATION_RESULTS.md`). If not found, the paper does not use them.

**9.2 Main HEST table (same-head runs)** — per cancer and three-cancer mean: E, P, T for seed 0, seed 1 and the two-seed mean; plus per-fold values in the supplement. A second version excludes IDC fold 2 seed 0. State the metric once: "per-gene Pearson correlation on each held-out fold, averaged over genes, then over folds; cancers weighted equally" (adjust the wording to what the code actually computes, and use only that metric everywhere).

**9.3 Scale ablation (if logged for the same runs)** — local / meso / macro / union / scale-aware per cancer, with seeds. If only the old untraceable table exists, drop the scale ablation from the main text and describe the GNN as one system (P).

**9.4 Secondary table** — the nine foundation models from HEST's published linear-probe protocol (PCA-256 + Ridge), labeled as such, with ePathST-E's own linear-probe score (0.3144 in the repo) beside them. Never put linear-probe and MLP-head numbers in the same column.

**9.5 Sample sizes** — "IDC 4 test sections (n spots), COAD 4 (n; 2 folds), ccRCC 24 (n; 6 folds); training slides per fold from check A9; 2 seeds." Limitations: IDC and COAD rest on four patients each.

**9.6 Compute** — recompute per-patch MACs/FLOPs for the student and teacher with one tool (`fvcore` counts MACs; say so) and one patches-per-WSI assumption; recheck the old ResNet-50 ratio (3.69× vs about 7.5× expected from 4.1 vs 0.55 GMACs). Efficiency sentences use one comparator each.

**9.7 Wording rules for HEST:** "comparable" for differences under about 0.01 with two seeds; context gain reported as the seed range (+0.006 to +0.034); no claims that tissue topology causes the gain (an earlier shuffled-graph control matched the real graph and a Moran's I test was null — report both); "94%" never appears unless recomputed from the same-head table and labeled as a score ratio.

**CHECK:** provenance result, the main and secondary tables, the compute table.

## Step 10 — Figures, tables, numbers sheet (`58_figures.py`)

**Five main figures** (each caption states n patients/sections, the metric and the test):

| Fig. | Panels | Source |
| --- | --- | --- |
| 1 Study design | CPTAC distillation → frozen encoder → HEST heads (E, P, T) → frozen TCGA inference → screen + Cox (trained on TCGA); radii labels 0–150 / 150–300 / 300–600 µm; 3.39M encoder vs 5.25M system | redrawn |
| 2 Compression | (a) same-head HEST bars E, P, T per cancer, seed points shown; (b) TCGA screen: median ρ per system per cancer; (c) U3: ΔC of genes vs embeddings | Steps 9, 4, 6 |
| 3 Context | (a) P − E on HEST per cancer with both seeds; (b) per-gene P − E on TCGA ρ, sorted, CIs | Steps 9, 4 |
| 4 Transfer and biology | (a) per-gene ρ with 95% CI, all 50 genes per cancer, E/P/T side by side, q < 0.05 marked, partial ρ as open marks; (b) U4 program-type panel; (c) one scatter (gene chosen by rule: median ρ among q < 0.05 genes of P in BRCA) | Steps 4, 7 |
| 5 Prognosis | (a) forest plot of Δ Uno's C vs C per system and cancer, corrected CIs, primary test highlighted; (b) BRCA KM curves with numbers at risk | Steps 5, 8 |

**Main tables:** (I) cohorts: CPTAC, HEST (sections, spots, folds, training slides per fold), TCGA CONSORT final n, events, median follow-up; (II) HEST same-head table; (III) C-index and ΔC table including U3 rows.

**Supplement:** full screen (450 rows), specificity, U4 tables, per-gene HRs, stability, PH tests, sensitivity analyses, Harrell's C, AUC, Brier, subgroups, linear-probe table, `analysis_plan.json`, `DEVIATIONS.md`.

**Plot rules:** one fixed colour per system across all figures (E, P, T, B); error bars always labeled (95% CI unless stated); vector PDF plus 300-dpi PNG; fonts ≥ 7 pt at print size; no figure shows a number that is not in a results file.

**Numbers sheet:** `docs/reports/NUMBERS_SHEET.md` — every number in the paper → file → row/column → script → commit. Update `README.md` with a figure/table → script → output map and a code-availability statement.

**Release (after the paper numbers are frozen):** tag the commit `v1.0-paper`; publish code; publish the TCGA patient-level predicted gene tables (derived from open-access data) with a manifest. Check the H-Optimus-0 licence before releasing any weights distilled from it.

**Folder layout**

```
results/tcga_final/
  ENVIRONMENT.txt  DEVIATIONS.md  logs/
  audit/                  AUDIT_REPORT.md, A1..A15 csv
  predictions_rebuilt/    only if the collapse rule changed an ensemble
  cohort/                 {cancer}_patients.csv, consort_{cancer}.csv, followup.csv
  features/               {cancer}_{E,P,T}_{mean,median,sd}.parquet, {cancer}_{HS,HT,B}.parquet,
                          gene_matching.csv, gene_properties.csv
  screen/                 screen_full.csv, specificity_summary.csv, system_deltas.csv
  survival/               cv_folds.csv, cindex_by_split.csv, delta_c.csv, pooled_ci.csv,
                          chosen_alpha.csv, coefficients.csv, independence.csv, ph_tests.csv,
                          bottleneck_summary.csv, pca_variance.csv
  programs/               gene_level_outcomes.csv, program_tests.csv, hest_vs_tcga.csv
  survival_exploratory/   stability.csv, per_gene_hr.csv, loghr_agreement.csv, km_*.pdf, subgroups.csv
  hest/                   hest_main.csv, hest_linear_probe.csv, compute.csv
  figures/                fig1..fig5 .pdf/.png, supp_*.pdf
```

**CHECK:** all figures, the numbers sheet, then `git push origin final-paper`.

## Testing hierarchy and power

One primary test is named in advance, and every other family is corrected within itself, so the headline cannot be the luckiest of hundreds of tries.

| Level | Analysis | Tests | Control |
| --- | --- | --- | --- |
| Primary | BRCA: C+P vs C, Δ Uno's C | 1 | α = 0.05, two-sided |
| Key secondary | ePathST (P) molecular screen | up to 150 | BH q = 0.05 |
| Secondary | Step 5.5 comparisons in all cancers | about 20 | BH within family |
| Secondary | U3 bottleneck comparisons | 12 | BH within family |
| Reference | E and T screens; purity-adjusted screens | 150 each | BH within each; reported, not claimed |
| Exploratory | U4, per-gene HRs, stability, KM, subgroups, highlighted genes | many | BH within each table; labeled exploratory |

**Power, honestly:** event counts come from Step 2. With about 100 events a single C-index has a standard error near 0.03; the paired ΔC is tighter, but gains under about 0.02 are unlikely to reach significance. Fifty genes break the 10-events-per-variable rule for ordinary Cox, which is why the gene models are penalized and judged out of fold. A null primary result is realistic and is reported as such.

## What may be claimed for each outcome

Decided now, so results cannot pull the wording. Every row is publishable.

| Result | Allowed claim | Not allowed |
| --- | --- | --- |
| Many P genes q < 0.05, partial ρ stays positive, matched rank well below 25.5 | "Frozen predictions tracked gene-specific expression in an independent cohort and assay, beyond tumour purity" | "Validated spatial expression"; anything about location within the slide |
| Screen positive but purity or specificity removes it | "Predictions mainly tracked tissue composition" | Calling it gene-level transfer |
| E ≈ T on HEST and TCGA | "Under an identical head, the 3.39M student showed no detectable difference from the 1.13B teacher (CI \[a, b\])" | "Equivalent" or "as good as" without a CI |
| T > E | "Compression cost x (CI) at y× less compute" | Hiding the gap |
| H1 significant | "Slide-predicted expression improved discrimination beyond age and stage (ΔC, CI) in a retrospective cohort; prospective validation needed" | "Predicts survival", "clinically validated" |
| H1 null | "No detectable improvement beyond age and stage in BRCA (ΔC, CI); molecular transfer results stand separately" | Promoting a secondary cancer to the headline without saying H1 was null |
| U3 genes ≈ embeddings | "The 50-gene layer kept the image's prognostic signal while making it interpretable" | "No information lost" |
| U3 embeddings > genes | "The image carries prognostic information beyond this 50-gene panel" | — |
| U4 identity > state | "In this panel, lineage and composition genes transferred better than cell-state genes (exploratory)" | Causal or general claims about all genes |

**Use:** estimate, consistent with, suggests, retrospective, prognostic stratification, therapy-relevant gene, ERBB2 mRNA. **Avoid:** diagnoses, replaces IHC or sequencing, clinically validated, discovers, causes, physically grounded, best model, zero-shot (for the survival model).

## Schedule to Nov 5 and final checklist

| Dates | Work |
| --- | --- |
| Oct 10 | Step 0 pre-registration; Step 1 audit |
| Oct 11 | Collapse decision and any ensemble rebuild; Steps 2–3 |
| Oct 12–13 | Step 4 screen |
| Oct 14–16 | Steps 5–6 (survival + U3; the longest compute: 9 sets × 3 cancers × 25 splits × inner CV) |
| Oct 17 | Step 7 (U4) |
| Oct 18 | Step 8 exploratory survival |
| Oct 19 | Step 9 HEST tables and provenance |
| Oct 20–22 | Step 10 figures, numbers sheet |
| Oct 23 – Nov 1 | Write the report yourself, in first person, in your own words; mentor and one outside reader review |
| Nov 2–3 | Numbers audit against the sheet; format check; submit at least 24 hours early |

If time runs short, cut in this order: subgroups (8.6), integrated Brier score, PCA sensitivity (16/64), domain-group tests (7.2 item 4). Never cut Step 0, Step 1, the primary test or the screen.

**Final checklist**

- [ ] Step 0 committed before any outcome was computed
- [ ] Audit passed; collapse rule applied; ensemble documented
- [ ] CONSORT tables and event counts
- [ ] Screen with purity adjustment and specificity control
- [ ] Survival: primary test, secondary family, independence, PH checks
- [ ] U3 bottleneck table
- [ ] U4 program tables
- [ ] Exploratory survival
- [ ] HEST same-head table, provenance result, compute recheck
- [ ] Five figures, three tables, numbers sheet
- [ ] Paper fixes: Fig. 1 radii, Fig. 2 loss/GELU (or remove), 3.39M vs 5.25M, "Yurovsky", "bluk", MERGE described accurately, TICON and PEaRL cited, mentor role and AI assistance disclosed
- [ ] Every number in the paper found in `NUMBERS_SHEET.md`

## Appendix A — Gene program labels and marker groups

Copy this into `configs/gene_programs.json` at Step 0. Rule: **state** = member of an MSigDB Hallmark response/process set (interferon α/γ response, inflammatory response, TNFα via NF-κB, hypoxia, unfolded protein response, G2M checkpoint, E2F targets, MYC targets, mTORC1, fatty-acid/cholesterol metabolism, heme metabolism) or a stress/acute-phase gene; **identity** = canonical cell-type or lineage marker (PanglaoDB / CellMarker) not classed as state; **other** = neither. Each cancer's three lists sum to its 50 panel genes; at Step 0 confirm every gene is in its panel and fix any mismatch before committing. These labels were drafted on 2026-10-06; any change after Step 0 goes in `DEVIATIONS.md`.

```json
{
  "program_type": {
    "IDC": {
      "state": ["FASN","SCD","TOP2A","MMP1","MMP12","SERPINA3"],
      "identity": ["ABCC11","ADH1B","ADIPOQ","ANKRD30A","AQP1","AQP3","CCR7","CD3E","CEACAM6","CEACAM8","CLIC6","CYTIP","ESR1","GATA3","IL2RG","IL7R","KIT","KLF5","KRT14","KRT5","KRT6B","MS4A1","MUC6","MYH11","MYLK","OXTR","PIGR","PTPRC","SCGB2A1","SFRP1","SLAMF7","TACSTD2","TCL1A","TENT5C","TPSAB1","TRAC","VWF"],
      "other": ["DST","ERBB2","MYBPC1","OPRPN","PTGDS","PTN","SERHL2"]
    },
    "COAD": {
      "state": ["AREG","CCL20","CXCL10","DUOX2","IL1B","LCN2","MMP1","MMP3","PI3","TYMS"],
      "identity": ["AQP8","BANK1","CA1","CA4","CCR7","CD177","CEACAM6","CEACAM7","CHGA","CHGB","CLCA1","CLCA4","CTSE","FCRL1","GATA2","GCG","GUCA2A","GUCA2B","INSL5","ITLN1","MS4A1","MS4A12","NXPE4","OLFM4","OTOP2","PRPH","RETNLB","SCGN","SLC26A3","SPIB","TCL1A","TMIGD1"],
      "other": ["CALB2","CHI3L2","COL11A1","CXCL13","DMBT1","HEPACAM2","SCG2","UGT2B17"]
    },
    "CCRCC": {
      "state": ["DERL1","GADD45B","HMOX1","HSPA1A","IFI27","IFI6","IRF7","ISG15","LY6E","NAMPT","PARP10","TFRC","TP53INP1","VMP1"],
      "identity": ["CD14","EGFL7","GPNMB","NOTCH3","S100A9","TRBC1"],
      "other": ["AKAP8","BLVRB","BTF3","CD109","CHST15","CIB1","CKB","CPD","CPE","DNAJB12","FLNB","H3F3A","HNRNPK","HTRA1","KRTCAP2","LAPTM4B","LRRC41","NOL3","NT5DC2","PTGS1","PXN","RXRA","SCARB1","SEMA4B","SLC15A4","SLC39A14","STK39","TM2D3","TMEM91","YWHAE"]
    }
  },
  "domain_markers": {
    "IDC": {
      "tumor_epithelium": ["ESR1","GATA3","ERBB2","TACSTD2","SCGB2A1","ANKRD30A","ABCC11","CEACAM6"],
      "immune": ["PTPRC","CD3E","TRAC","IL7R","MS4A1","TCL1A","CCR7","CYTIP","IL2RG","SLAMF7"],
      "stroma_vascular": ["MYH11","MYLK","VWF","AQP1","PTGDS","PTN"],
      "adipose": ["ADH1B","ADIPOQ"],
      "basal_myoepithelial": ["KRT5","KRT14","KRT6B","OXTR","SFRP1"]
    },
    "COAD": {
      "tumor_epithelium": ["CEACAM6","CEACAM7","OLFM4","LCN2","DMBT1","CTSE"],
      "normal_mucosa": ["CA1","CA4","GUCA2A","GUCA2B","AQP8","SLC26A3","MS4A12","CLCA4","CLCA1"],
      "immune": ["MS4A1","BANK1","CXCL13","CCR7","TCL1A","FCRL1","SPIB"],
      "stroma": ["COL11A1","MMP1","MMP3","CALB2","PRPH"]
    },
    "CCRCC": {
      "interferon_immune": ["IFI27","ISG15","IRF7","IFI6","LY6E","TRBC1"],
      "myeloid": ["CD14","GPNMB","S100A9","HMOX1"],
      "vascular": ["EGFL7","NOTCH3"]
    }
  }
}
```

Note for interpretation: ccRCC has only 6 identity genes, so its identity-vs-state comparison is descriptive (Step 7.2, item 1). H3F3A is listed under its old symbol; Step 3.3's HGNC matching handles the current name (H3-3A).

## Appendix B — Shared statistics functions

Implement these once in `src/epathst/analysis/stats_utils.py`, with the unit tests listed, before Step 4. Every script imports them; none re-implements them.

| Function | Signature | Behaviour | Unit test |
| --- | --- | --- | --- |
| Bootstrap Spearman | `spearman_boot(x, y, n_boot=2000, seed=0, idx=None) -> (rho, lo, hi, p)` | NaN-pairs dropped; `idx` = pre-drawn (n\_boot × n) index matrix so several systems share draws; percentile CI; p from `spearmanr` (n ≥ 100) else permutation | Perfectly monotone data → ρ = 1, CI = \[1, 1\]; independent normals n = 500 → CI covers 0 in \~95% of 200 simulations |
| Partial Spearman | `partial_spearman(x, y, z) -> (r, p, n)` | Rank x, y, z; residualize x and y on z (OLS with intercept); Pearson of residuals; t-test df = n − 3 | Matches `pingouin.partial_corr(method="spearman")` to 1e-8 |
| Permutation p | `perm_p(stat_fn, labels, groups, n_perm=10000, seed=0) -> p` | Permutes labels within each group; two-sided (1 + count) / (n\_perm + 1) | Under shuffled labels, p uniform (KS test over 200 runs) |
| BH | `bh(pvals) -> q` | Wraps `multipletests(method="fdr_bh")`; ignores NaN and keeps their positions | Matches statsmodels on a fixed vector |
| Nadeau–Bengio | `nb_ttest(deltas, n_train, n_test) -> (mean, lo, hi, t, p)` | Corrected variance (1/J + n\_test/n\_train)·s², df = J − 1, two-sided | Hand-computed example: deltas = \[0.01\]\*25 + noise → compare with a manual calculation |
| Uno's C | `uno_c(y_tr, y_te, risk, tau) -> float` | Wraps `concordance_index_ipcw`; raises if tau exceeds training follow-up | Known toy data from the scikit-survival docs |
| Coxnet fit | `fit_coxnet(X_tr, y_tr, n_clin, seed) -> model, alpha` | Step 5.3 items 3–6 exactly | On simulated data with 2 true genes, both selected in most of 20 repeats |
| Within-fold percentile | `to_percentile(risk) -> array` | rank / (n + 1) | Monotone, values in (0, 1) |
| Rank-biserial | `rank_biserial(a, b) -> r` | 2U/(n\_a n\_b) − 1 | All a > b → r = 1 |

All functions accept and return NumPy arrays or pandas Series and never use global random state.

## Methods references

Cited from memory, not re-fetched; verify each before it goes in the paper.

- Liu J, et al. An integrated TCGA pan-cancer clinical data resource to drive high-quality survival outcome analytics. Cell 2018.
- Aran D, Sirota M, Butte AJ. Systematic pan-cancer analysis of tumour purity. Nat Commun 2015.
- Uno H, et al. On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data. Stat Med 2011.
- Nadeau C, Bengio Y. Inference for the generalization error. Machine Learning 2003.
- Benjamini Y, Hochberg Y. Controlling the false discovery rate. J R Stat Soc B 1995.
- Simon N, Friedman J, Hastie T, Tibshirani R. Regularization paths for Cox's proportional hazards model via coordinate descent. J Stat Softw 2011.
- Grambsch PM, Therneau TM. Proportional hazards tests and diagnostics based on weighted residuals. Biometrika 1994.
- Liberzon A, et al. The Molecular Signatures Database Hallmark gene set collection. Cell Syst 2015.
- Collins GS, et al. TRIPOD+AI statement. BMJ 2024; McShane LM, et al. REMARK. J Natl Cancer Inst 2005.
- Pölsterl S. scikit-survival. J Mach Learn Res 2020.
- Kapse S, et al. TICON: a slide-level tile contextualizer for histopathology representation learning. CVPR 2026 (findings).
- Jaume G, et al. HEST-1k. NeurIPS 2024.
