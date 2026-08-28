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
`sts/cptac_select.py` for the QC/selection logic as implemented in this
repo, and `scripts/01_select_cptac_wsi.py` for how it runs against the
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
  ("VL Whole Slide Microscopy Image"). Read via sts.wsi_patches
  .WsiDicomReader (the `wsidicom` package) — plain openslide-python cannot
  read this without a libdicom-enabled build.
```

Note the cancer-type directory casing on disk (`BRCA`, `COAD`, `ccRCC`) —
`ccRCC` does not match `configs/base.yaml`'s `cancers.cptac: [BRCA, COAD,
CCRCC]` casing; `scripts/01_select_cptac_wsi.py`'s `CANCER_DIR_ALIASES`
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

## 2. HEST (Phase 2 — downstream spatial transcriptomics)

**What**: Paired H&E whole-slide images and spatial transcriptomics (ST)
expression data (Visium-style spot arrays) from the HEST benchmark,
restricted to the same three cancer types (colon/COAD, breast/IDC,
kidney/CCRCC) where available. Used to train/evaluate the graph-based
molecular expression prediction — the locked CPTAC-distilled student
encoder is applied here as a frozen feature extractor; no further
distillation happens in this phase.

**Source / provenance**: Yurovsky-lab HEST cohort (`simons_2026`), the same
CRC23 colon/COAD cohort referenced as "23 slides = 23 unique patients" in
the project's PoC_5 work. As of the current run, only the `colon` (COAD)
directory has a populated `copy_log_*.csv`; `breast` (IDC) and `kidney`
(CCRCC) directories exist on disk but have no metadata log yet —
`scripts/08_prepare_hest_metadata.py` skips-and-reports missing cancers
rather than failing when this is the case.

**On-disk layout** (`paths.hest_root` in `configs/base.yaml`):

```
HEST_ROOT/
  copy_log_{colon,breast,kidney}_cancer.csv   metadata index, one row/sample
  {colon,breast,kidney}/
    wsis/{id}.tif    pyramidal generic TIFF, openslide-readable
    st/{id}.h5ad      standard scanpy/Visium AnnData
```

**`copy_log_{organ}_cancer.csv` columns** (only the ones this pipeline
uses):

| column | meaning |
|---|---|
| `id` | sample identifier — doubles as `patient_id` (one section per patient; no separate patient_id column exists) |
| `metadata_organ` / `destination_organ` | organ subdirectory name (`colon`/`breast`/`kidney`) |
| `disease_state` | filtered to `"cancer"` only |
| `oncotree_code` | `COAD`, `COADREAD`, or `READ` for the colon cohort — all three map to this project's single `"COAD"` cancer bucket (`ONCOTREE_TO_CANCER`); analogous mappings apply once breast/kidney logs land |
| `st_technology` | ST platform (Visium, etc.) |
| `image_filename` | filename under `{organ}/wsis/` |
| `dest_h5ad` / the derived `{id}.h5ad` | expression file under `{organ}/st/` |

There is no `species` column — every present sample is confirmed human, so
`sts.hest_prep` hardcodes `species: human` rather than reading it.

**`{id}.h5ad` structure** (standard scanpy/Visium AnnData):

- `X` — raw gene expression counts, spots × genes
- `obs` — `array_row`/`array_col` (Visium grid coords), `pxl_row_in_fullres`/`pxl_col_in_fullres` (pixel coords in the full-resolution WSI)
- `obsm['spatial']` — `(x_px, y_px)` = `(pxl_col_in_fullres, pxl_row_in_fullres)`, already filtered to `in_tissue == 1`
- `uns['spatial']['ST']['scalefactors']['spot_diameter_fullres']` — the Visium 55µm spot's diameter in pixels; used as an independent cross-check of microns-per-pixel against `openslide.mpp-x` (confirmed to agree to 4 decimal places, e.g. 0.7257 vs. 0.7258 µm/px on sample `TENX28`)

**Alignment QC**: before any patch extraction, `sts.hest_prep
.alignment_qc_sample` samples ST spot coordinates and confirms they
actually land on real tissue in the paired WSI (`pct_spots_with_tissue`);
a sample is excluded as `likely_misaligned` if this check fails, since a
coordinate-system mismatch would otherwise silently corrupt every
downstream patch/graph.

**Scale (current)**: 23 real colon/COAD WSIs (the confirmed CRC23 cohort),
0 excluded by the metadata audit. Breast/kidney pending metadata arrival.

## 3. Derived artifacts (not raw input, generated by the pipeline)

Everything under `runs/<experiment_name>/` (patch shards, manifests, QC
reports, teacher embeddings, student checkpoints, graphs, predictions,
results, figures) is *generated* from the two datasets above, not raw
input — see `design/DESIGN.md` §3 for the full data-flow table. These are
excluded from git (`.gitignore`: `runs/`, `logs/`, `*.h5`, `*.h5ad`,
`*.pt`, `*.pth`) and, per `docs/RUN_STATUS.md`, are kept off `/home` and
`/scratch` entirely (20GB per-user quota) in favor of the
`YurovskyGroup-nvwulf` project space.
