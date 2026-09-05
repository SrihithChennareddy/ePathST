# Input data

This project consumes two independent real-world datasets. Neither is
committed to this repo (both are large, DICOM/TIFF/HDF5 binary data) — this
document is the reference for what they are, where they live on disk, and
their exact schema, so the pipeline can be reasoned about without needing
filesystem access. See `input_data/README.md` at the repo root for the
short pointer version, and `configs/base.yaml`'s `paths:` block for the
config keys that resolve to these locations at run time.

## 1. CPTAC (Phase 1 — teacher/student distillation)

**What**: Whole-slide H&E histology images (WSIs) from the CPTAC-3 cohort,
covering three cancer types — breast (BRCA), colon (COAD), clear-cell
kidney (ccRCC). Used to train and evaluate the distilled student encoder;
no molecular/expression data involved in this phase.

**Source / provenance**: Imaging Data Commons (IDC), `idc-open-data` public
S3 bucket, collections `cptac_brca`, `cptac_coad`, `cptac_ccrcc`. One WSI
per patient was pre-selected (from all available candidate slides per
patient) using a metadata-based quality proxy — tumor-tagged specimen
preferred over ambiguous/normal-tagged, then ranked by tissue-quantity/
resolution proxy (`series_size_MB`, pyramid instance count) — with an
additional pixel-level QC pass (tissue fraction, focus/Laplacian variance,
pen-mark/artifact detection, stain-saturation outliers) used to re-verify
or override that pick where multiple candidates existed. See
`src/pathology_st/cptac_select.py` for the QC/selection logic as implemented in this
repo, and `src/scripts/01_select_cptac_wsi.py` for how it runs against the
already-downloaded data below.

**On-disk layout** (`paths.cptac_root` / `paths.cptac_metadata` in
`configs/base.yaml`):

```
CPTAC_METADATA/
  BRCA_selected_WSI_per_patient.txt   \  TSV, one row per patient
  COAD_selected_WSI_per_patient.txt    > (already deduplicated to
  ccRCC_selected_WSI_per_patient.txt  /   1 WSI/patient upstream)

CPTAC_ROOT/{BRCA,COAD,ccRCC}/{patient_id}/*.dcm
  one DICOM whole-slide-image SERIES per patient: multiple .dcm SOP
  instances forming a resolution pyramid (VOLUME levels + LABEL/OVERVIEW/
  THUMBNAIL auxiliary images), SOPClassUID 1.2.840.10008.5.1.4.1.1.77.1.6
  ("VL Whole Slide Microscopy Image"). Read via pathology_st.wsi_patches
  .WsiDicomReader (the `wsidicom` package) — plain openslide-python cannot
  read this without a libdicom-enabled build.
```

Note the cancer-type directory casing on disk (`BRCA`, `COAD`, `ccRCC`) —
`ccRCC` does not match `configs/base.yaml`'s `cancers.cptac: [BRCA, COAD,
CCRCC]` casing; `src/scripts/01_select_cptac_wsi.py`'s `CANCER_DIR_ALIASES`
handles the mismatch.

**`{CANCER}_selected_WSI_per_patient.txt` columns**:

| column | meaning |
|---|---|
| `patient_id` | CPTAC patient identifier, matches the `{patient_id}` directory name |
| `wsi_series_uid` | DICOM SeriesInstanceUID of the selected slide |
| `study_uid` | DICOM StudyInstanceUID |
| `specimen_label` | e.g. `HE tumor_tissue`, `HE`, `HE normal_tissue` — tumor vs. ambiguous vs. normal tissue tag |
| `specimen_rank` | 1 = tumor, 2 = ambiguous, 3 = normal (selection priority tier) |
| `series_size_MB` | total size of the DICOM series, used as a tissue-quantity/resolution proxy |
| `instanceCount` | number of DICOM instances (pyramid levels) in the series |
| `aws_bucket` | source S3 bucket (`idc-open-data`) |
| `series_aws_url` | `s3://idc-open-data/<uuid>/*` — the exact path used to fetch the slide |

**Scale**: 198 BRCA + 178 COAD + 222 ccRCC = 598 patients, ~129GB total
across all three cancer types (598 DICOM series, one per patient).

**Known data gaps**: COAD has 43 patients where the *only* available slide
in IDC is normal-tissue-labeled (no tumor slide exists for them at all) —
included by necessity, not preference, flagged via `specimen_rank == 3`.
One BRCA patient (`18BR003`) has a single candidate slide that fails to
decode (OpenJPEG `SOC marker` error) — included with `qc_status == error`
since it's the only candidate.

### Real post-QC selection funnel (job 83825, first clean run, 2026-08-28)

Section 4's hard filters (human / H&E / tumor specimen / readable / mpp
known / adequate tissue) were applied to all 598 candidates via
`src/pathology_st/cptac_select.py`. Full results, for the paper's Data/Methods section:

| Stage | BRCA | COAD | CCRCC | Total |
|---|---|---|---|---|
| Initial candidates (metadata) | 198 | 178 | 222 | 598 |
| Excluded: `not_tumor_specimen` | 90 | 131 | 0 | 221 |
| Excluded: `insufficient_tissue` | 2 | 0 | 6 | 8 |
| Excluded: `unreadable_or_corrupt` | 0 | 0 | 6 | 6 |
| **Selected (passed all hard filters)** | **106** | **47** | **210** | **363** |

**Why COAD's yield is disproportionately low (74% excluded vs. BRCA's 45%
and CCRCC's 0%) — a documented limitation, not a pipeline bug.** The
`not_tumor_specimen` filter checks whether `specimen_label` contains the
substring "tumor". The raw label distributions differ sharply by cancer
type in CPTAC/IDC's own metadata:

| Cancer | `"HE tumor_tissue"` (passes) | `"HE"` bare (excluded) | `"HE normal_tissue"` (excluded) |
|---|---|---|---|
| CCRCC | 222 (100%, labeled `"HE tumor"`) | 0 | 0 |
| BRCA | 108 | 90 | 0 |
| COAD | 47 | 88 | 43 |

Two distinct effects combine for COAD: (1) 43 patients' only available
slide is genuinely matched-normal tissue (correctly excluded from a
tumor-focused study — this is real, expected, and already noted above),
and (2) a much larger fraction than BRCA (88/178 = 49% vs. 90/198 = 45%,
comparable rate, but combined with (1) it compounds) carry only the bare,
unqualified `"HE"` label with no explicit tumor/normal designation in this
specific metadata field. **The current filter treats "no explicit 'tumor'
substring" as "not a tumor specimen," which is a conservative
approximation, not a verified fact** — some of these 88 (COAD) + 90 (BRCA)
bare-`"HE"` specimens may well be genuine tumor sections that simply
weren't tagged with that word in `specimen_label`. This has not been
cross-checked against a more authoritative CPTAC diagnosis/pathology field
and should be flagged as a limitation (or resolved, if a better field is
found) before the final cohort composition is reported in the paper.

### Patient-level train/val/test split (seed 0, from the selected 363)

Split is **patient-level, 80/10/10, stratified per cancer type**
(`configs/base.yaml`: `train_frac/val_frac/test_frac`), computed by
`src/pathology_st/patient_split.py` *before* any patch is extracted — no patient's
patches ever cross a split boundary (enforced by
`assert_no_patch_leakage`). Real counts:

| Cancer | Train | Val | Test | Total |
|---|---|---|---|---|
| BRCA | 85 | 11 | 10 | 106 |
| CCRCC | 168 | 21 | 21 | 210 |
| COAD | 38 | 5 | 4 | 47 |
| **Total** | **291** | **37** | **35** | **363** |

Three independent patient-level splits (seeds 0, 1, 2) are generated the
same way for later multi-seed robustness reporting; only seed 0 has been
trained on as of this writing (see `docs/RUN_STATUS.md` /
`docs/COMPUTE_RESOURCES.md` for training progress).

## 2. HEST (Phase 2 — downstream spatial transcriptomics)

**What**: Paired H&E whole-slide images and spatial transcriptomics (ST)
expression data (Visium-style spot arrays) from the HEST benchmark, across
all three of this project's cancer types (colon/COAD, breast/IDC,
kidney/CCRCC — all now confirmed present on disk, see below). Used to
train/evaluate the graph-based molecular expression prediction — the
locked CPTAC-distilled student encoder is applied here as a frozen feature
extractor; no further distillation happens in this phase.

**Source / provenance** (confirmed on-disk 2026-08-29 — all three cancers
now available, two different provenances):

| Cancer | Location | Provenance | Scale |
|---|---|---|---|
| Breast (IDC) | `/lustre/nvwulf/projects/YurovskyGroup-nvwulf/simons_2026/breast/{wsis,st}/` | Yurovsky-lab HEST cohort, used **in place** (not copied locally) | 127 WSI (~62GB) / 125 ST (~7.4GB) |
| Colon (COAD) | `/lustre/nvwulf/scratch/schennareddy/HEST-1k/data/colon/{wsis,st}/` | Downloaded from HuggingFace `MahmoodLab/hest` | 58 WSI / 58 ST (~184GB total) — supersedes the earlier 23-slide `simons_2026/colon` CRC23 subset used for PoC_5/early validation |
| Kidney (CCRCC) | `/lustre/nvwulf/scratch/schennareddy/HEST-1k/data/kidney/{wsis,st}/` | Downloaded from HuggingFace `MahmoodLab/hest` | 25 WSI / 25 ST (~18GB total) |

Each cancer's metadata is a standard HEST-1k per-organ CSV under
`/lustre/nvwulf/scratch/schennareddy/HEST-1k/metadata/{breast,colon,kidney}_metadata.csv`
(downloaded alongside the colon/kidney WSI+ST data; also covers breast even
though breast's WSI/ST files themselves live at the `simons_2026` path
above). `configs/base.yaml`'s `paths.hest_by_cancer` records all three
locations (`wsi_dir`, `st_dir`, `metadata_csv`) per cancer; the pipeline no
longer assumes a single shared `hest_root` layout — see
`src/scripts/08_prepare_hest_metadata.py`'s `load_hest_raw_metadata`, which
reads each cancer's config entry independently and skips-and-reports any
cancer with no configured/found source, same fallback behavior as before.

**On-disk layout** (`paths.hest_by_cancer.{IDC,COAD,CCRCC}` in
`configs/base.yaml`):

```
{wsi_dir}/{id}.tif    pyramidal generic TIFF, openslide-readable
{st_dir}/{id}.h5ad     standard scanpy/Visium AnnData
```

**`{organ}_metadata.csv` columns** (only the ones this pipeline uses; the
breast CSV additionally carries stale `wsi_path`/`h5ad_path` columns
pointing at a different group's directory — **ignored**, since `id` +
`hest_by_cancer.IDC.{wsi_dir,st_dir}` is the source of truth for where the
real files actually are):

| column | meaning |
|---|---|
| `id` | sample identifier — doubles as `patient_id` (one section per patient; no separate patient_id column exists) |
| `organ` | `Breast`/`Bowel`/`Kidney` |
| `disease_state` | filtered to `"cancer"` (case-insensitive) only |
| `oncotree_code` | mapped onto this project's three cancer buckets via `ONCOTREE_TO_CANCER`: `IDC`→IDC (excludes 4/125 `ILC` breast rows — a biologically distinct histology); `COAD`/`COADREAD`/`READ`→COAD (all colorectal); `SCCRCC`→CCRCC (excludes 1/25 `PRCC` kidney row — papillary RCC, a distinct RCC subtype) |
| `st_technology` | ST platform (Visium, Visium HD, Xenium, etc.) |
| `species` | present for colon/kidney (`Homo sapiens`); absent from the breast CSV — every present breast sample is confirmed human, so `pathology_st.hest_prep` defaults to `species: human` when the column is missing |

**`{id}.h5ad` structure** (standard scanpy/Visium AnnData):

- `X` — raw gene expression counts, spots × genes
- `obs` — `array_row`/`array_col` (Visium grid coords), `pxl_row_in_fullres`/`pxl_col_in_fullres` (pixel coords in the full-resolution WSI)
- `obsm['spatial']` — `(x_px, y_px)` = `(pxl_col_in_fullres, pxl_row_in_fullres)`, already filtered to `in_tissue == 1`
- `uns['spatial']['ST']['scalefactors']['spot_diameter_fullres']` — the Visium 55µm spot's diameter in pixels; used as an independent cross-check of microns-per-pixel against `openslide.mpp-x` (confirmed to agree to 4 decimal places, e.g. 0.7257 vs. 0.7258 µm/px on sample `TENX28`)

**Alignment QC**: before any patch extraction, `pathology_st.hest_prep
.alignment_qc_sample` samples ST spot coordinates and confirms they
actually land on real tissue in the paired WSI (`pct_spots_with_tissue`);
a sample is excluded as `likely_misaligned` if this check fails, since a
coordinate-system mismatch would otherwise silently corrupt every
downstream patch/graph.

**Scale (current, post metadata-audit, before file-existence/alignment
filtering)**: 202 real samples across all three cancers — IDC 120, COAD 58,
CCRCC 24 (verified live against the actual metadata CSVs + on-disk
WSI/ST files, 2026-08-29; IDC's 120 vs. 121 oncotree-matched rows reflects
one sample missing its paired WSI/ST file on disk). 0 excluded by
`audit_hest_samples`'s mouse/cancer-bucket/duplicate-id filters. Full
alignment QC + patch extraction across all 202 has not yet been run at
scale — `src/scripts/08_prepare_hest_metadata.py` has only been exercised on a
1-2 sample subset for real-data validation so far (see
`docs/V1_TO_V2_MIGRATION.md`); the full run is future work.

## 3. Derived artifacts (not raw input, generated by the pipeline)

Everything under `runs/<experiment_name>/` (patch shards, manifests, QC
reports, teacher embeddings, student checkpoints, graphs, predictions,
results, figures) is *generated* from the two datasets above, not raw
input — see `design/DESIGN.md` §3 for the full data-flow table. These are
excluded from git (`.gitignore`: `runs/`, `logs/`, `*.h5`, `*.h5ad`,
`*.pt`, `*.pth`) and, per `docs/RUN_STATUS.md`, are kept off `/home` and
`/scratch` entirely (20GB per-user quota) in favor of the
`YurovskyGroup-nvwulf` project space.
