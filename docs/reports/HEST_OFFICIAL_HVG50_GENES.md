# HVG50 Gene Lists: Official HEST-1k Panel vs. This Project's Own Selection

Two different "HVG50" gene sets exist across this codebase's runs, and they are **not the same
genes**. This doc lists both, per cancer, and explains why they diverge (see also
`docs/HEST_OFFICIAL_ABLATION_RESULTS.md` for the results that use the official panel).

## How each is computed

**Official panel** (`<cancer>/var_50genes.json`, from the HEST-1k benchmark release,
`mahmoodlab/HEST`, `src/hest/utils.py::get_k_genes(criteria='var')`):
1. Take every sample belonging to the cancer (train **and** test pooled together -- computed
   once, offline, before any fold split exists).
2. Restrict to genes present in >=10% of spots in every sample, then intersect across samples
   to get one common gene set for the cancer.
3. Stack every sample's raw-count spots into one pooled expression matrix.
4. `sc.pp.log1p` on the pooled **raw** counts -- no `sc.pp.normalize_total` step.
5. `sc.pp.highly_variable_genes(n_top_genes=50)` (scanpy's default `flavor='seurat'`:
   genes are binned by mean expression, then ranked by **normalized dispersion within each
   bin** -- not raw variance).
6. This one list is then used, unchanged, for every official fold and every model evaluated
   against that cancer.

**This project's own HVG50** (`pathology_st.expression.select_hvg_for_fold`, used by
`src/scripts/10_run_patient_cv.py`'s custom protocol):
1. Per fold, using **train patients only** (a fresh selection every fold -- leakage-free by
   construction, but not the same list across folds or comparable to a fixed benchmark panel).
2. Per-slide `normalize_total(target_sum=1e4)` + `log1p` (official has no `normalize_total`).
3. Rank `common_genes` by plain `.var(axis=0)` (raw variance) across training spots, drop
   near-zero-variance genes, take the top 50 (official uses scanpy's dispersion-based HVG
   statistic instead of raw variance; raw variance is dominated by highly-expressed genes,
   while the dispersion statistic controls for the mean-variance relationship).

Neither is "wrong" in isolation -- the project's per-fold, train-only selection is the more
conservative choice to avoid gene-selection leakage in the project's own from-scratch
evaluation. But statistic, normalization, and data scope (train-only-per-fold vs.
pooled-whole-cohort) all differ, so the two procedures land on almost entirely different genes.

The project's list below is job 84871's fold-0 selection (`experimental_runs/
2026-08-31_job84871_final_5fold_3cancer_seed0_with_results_final/results/genbio_baseline/
hvg_lists/<cancer>_hvg50.json`) -- shown as one representative fold; the actual selection is
re-run fresh every fold.

## IDC

Overlap: 2/50 (`ERBB2`, `FASN`)

**Official** (`IDC/var_50genes.json`):
```
ABCC11, ADH1B, ADIPOQ, ANKRD30A, AQP1, AQP3, CCR7, CD3E, CEACAM6, CEACAM8,
CLIC6, CYTIP, DST, ERBB2, ESR1, FASN, GATA3, IL2RG, IL7R, KIT,
KLF5, KRT14, KRT5, KRT6B, MMP1, MMP12, MS4A1, MUC6, MYBPC1, MYH11,
MYLK, OPRPN, OXTR, PIGR, PTGDS, PTN, PTPRC, SCD, SCGB2A1, SERHL2,
SERPINA3, SFRP1, SLAMF7, TACSTD2, TCL1A, TENT5C, TOP2A, TPSAB1, TRAC, VWF
```

**Project (fold 0, train-only variance)**:
```
COX6C, TFF3, S100A9, ERBB2, BEST1, COL3A1, MGP, MUC1, IGFBP5, TMSB4X,
KRT8, COL1A2, COL1A1, AZGP1, IFI27, FN1, B2M, MIEN1, S100A8, ISG15,
PSMB3, TAGLN, CISD3, IFI6, C3, KRT7, POSTN, AEBP1, APOC1, FASN,
ADIRF, DCN, SPARC, APOD, IFITM3, HSP90AA1, CCND1, VIM, IGFBP2, SSR4,
S100A11, COL6A3, COL6A2, BGN, CTSD, FTH1, NBEAL1, UBC, C1QA, PSMD3
```

## COAD

Overlap: 2/50 (`LCN2`, `OLFM4`)

**Official** (`COAD/var_50genes.json`):
```
AQP8, AREG, BANK1, CA1, CA4, CALB2, CCL20, CCR7, CD177, CEACAM6,
CEACAM7, CHGA, CHGB, CHI3L2, CLCA1, CLCA4, COL11A1, CTSE, CXCL10, CXCL13,
DMBT1, DUOX2, FCRL1, GATA2, GCG, GUCA2A, GUCA2B, HEPACAM2, IL1B, INSL5,
ITLN1, LCN2, MMP1, MMP3, MS4A1, MS4A12, NXPE4, OLFM4, OTOP2, PI3,
PRPH, RETNLB, SCG2, SCGN, SLC26A3, SPIB, TCL1A, TMIGD1, TYMS, UGT2B17
```

**Project (fold 0, train-only variance)**:
```
IGKC, COL1A1, TAGLN, IGHG1, UBC, COL3A1, FN1, COL1A2, OLFM4, PIGR,
SPARC, DES, MYL9, CD24, VIM, TPM2, KRT8, TMSB4X, ACTA2, THBS1,
ACTB, B2M, FTL, TFF3, LGALS4, PHGR1, CEACAM5, DCN, MMP2, COL6A3,
FLNA, COL6A2, FTH1, CD74, FABP1, EPCAM, PPDPF, LCN2, IGFBP7, MUC2,
HSPB1, SERPINE1, JCHAIN, EEF1G, PTGES3, S100A14, TSPAN8, TIMP1, MT2A, EEF2
```

## CCRCC

Overlap: 1/50 (`CPE`)

**Official** (`CCRCC/var_50genes.json`):
```
AKAP8, BLVRB, BTF3, CD109, CD14, CHST15, CIB1, CKB, CPD, CPE,
DERL1, DNAJB12, EGFL7, FLNB, GADD45B, GPNMB, H3F3A, HMOX1, HNRNPK, HSPA1A,
HTRA1, IFI27, IFI6, IRF7, ISG15, KRTCAP2, LAPTM4B, LRRC41, LY6E, NAMPT,
NOL3, NOTCH3, NT5DC2, PARP10, PTGS1, PXN, RXRA, S100A9, SCARB1, SEMA4B,
SLC15A4, SLC39A14, STK39, TFRC, TM2D3, TMEM91, TP53INP1, TRBC1, VMP1, YWHAE
```

**Project (fold 0, train-only variance)**:
```
IGKC, IGHG1, MT1E, IGHG3, HLA-C, IGHA1, IGLC1, IGHG2, SERPINA1, IGHG4,
HLA-A, TGFBI, CRYAB, COL1A1, CPE, HLA-DRB1, COL3A1, SERPINE1, FXYD2, UBC,
COL1A2, SPP1, NDUFA4L2, MT2A, IGFBP3, FGB, CP, CD24, C3, NDRG1,
GPX3, RGS5, ACTA2, APOE, PDZK1IP1, JCHAIN, PDK4, CLU, FN1, PGF,
PTMA, IGHM, MMP9, HIST1H4C, CCN2, FABP7, SPINK13, APOL1, HSPA1B, DCN
```

## Takeaway

The project's train-only variance selection is dominated by immunoglobulin genes (`IGKC`,
`IGHG1`, ...) and generic stromal/housekeeping genes (`COL1A1`, `COL3A1`, `UBC`, `FTH1`,
`B2M`, ...) -- high raw-variance genes across whatever patients happen to be in that fold's
train split. The official panel is a curated, cancer-relevant marker set (e.g. IDC: `ESR1`,
`GATA3`, `ERBB2`, `KRT5/14`, immune markers `CD3E`/`PTPRC`/`MS4A1`; COAD: `GUCA2A/B`, `CA1/4`,
enteroendocrine markers `CHGA/CHGB/GCG`; CCRCC: hypoxia/stress-response genes `HMOX1`,
`GADD45B`, `NAMPT`). Read any R-value comparison across the two protocols as "same qualitative
finding, different genes," not a like-for-like score.
