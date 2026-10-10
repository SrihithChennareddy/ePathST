# ePathST: methods, data and results (aligned with the paper, Oct 9 version)

This document follows the paper *Efficient Histopathology Representation Learning for Spatial Gene Expression Prediction*
(Chennareddy and Yurovsky). Where it adds implementation detail it names the script or module that implements it. It replaces the earlier
`DESIGN.md`, which described superseded intermediate protocols.

## 1. Models

| Name | What it is | Parameters |
|---|---|---|
| **ePathST-E** | frozen distilled ConvNeXt V2-Atto encoder (320-d) + nonlinear patch-level predictor, no graph (A2 in `src/epathst/ablations.py`) | 3.39 M |
| **ePathST** | ePathST-E + scale-aware multiscale PNA graph network predicting a residual correction (A8) | 5.25 M |
| H-Optimus-0 | frozen 1.13 B-parameter ViT-g/14 teacher | 1,134 M |

## 2. Data

**Corpus:** over 953,000 H&E patches from two public datasets, used for pretraining (distillation), downstream training and evaluation.

**CPTAC (distillation).** H&E WSIs from CPTAC-BRCA, -COAD and -ccRCC via the Imaging Data Commons. After quality control on specimen type,
tissue composition and physical resolution of 598 candidate slides, 363 WSIs from 363 patients were kept. They were split 80/10/10
(train/validation/test) at the patient level before patch extraction. 564,312 patches of 224 x 224 pixels at about 0.5 um/pixel.

| Cohort | WSIs | Patients | Patches |
|---|---|---|---|
| CPTAC-BRCA | 106 | 106 | 167,825 |
| CPTAC-COAD | 47 | 47 | 80,776 |
| CPTAC-ccRCC | 210 | 210 | 315,711 |
| **Total** | **363** | **363** | **564,312** |

**HEST-1K (spatial gene-expression prediction).** 202 WSIs with paired spatial transcriptomics after the final cohort selection: 120 IDC,
58 COAD, 24 ccRCC. ST coordinates were verified against the matching WSI before extracting a 224 x 224 patch centred on every spot.

| Cohort | Samples / WSIs | Spot-patch pairs |
|---|---|---|
| HEST-IDC | 120 | 129,188 |
| HEST-COAD | 58 | 186,213 |
| HEST-ccRCC | 24 | 74,220 |
| **Total** | **202** | **389,621** |

Implementation: `src/scripts/01`-`03` (CPTAC selection, patches, patient split), `08` (HEST metadata audit), `09` (student features on HEST).

## 3. Pathology representation distillation

The frozen H-Optimus-0 teacher is distilled into an ImageNet-initialised ConvNeXt V2-Atto student on the patient-disjoint CPTAC partitions.
For a patch x, teacher t = f_T(x) in R^1536 and student h = f_S(x) in R^320. A training-only projection head maps the student into the teacher
space (320 -> 1024 -> 1536, GELU between the linear layers). Both embeddings are L2-normalised and the loss is
L = L_cos + 0.1 L_feature, where L_cos is the mean of 1 - cosine similarity and L_feature is Smooth-L1 between the normalised embeddings.
Optimiser: AdamW, learning rates 1e-4 (student) and 3e-4 (projection head), weight decay 0.05, batch size 256, cosine decay with a
three-epoch warm-up, up to 100 epochs, checkpoint chosen by validation cosine similarity. The selected checkpoint (epoch 55) reaches a mean
teacher-student cosine similarity of 0.9060 on the CPTAC test partition. The projection head is then discarded and the frozen 320-d
student is used for all HEST-1K experiments. Implementation: `src/scripts/04`-`07`, `src/epathst/distillation/`, `models/student.py`, `models/teacher.py`.

## 4. Multiscale spatial gene-expression prediction

Each transcriptomic spot i has a 224 x 224 patch encoded as h_i = f_S(x_i) in R^320. Coordinates are not node features; they only define the graph.

- **Patch predictor** f_patch: 320 -> 128 -> 64 with LayerNorm, GELU and dropout, then the gene-expression output (`models/patch_head.py`).
- **Graph:** spots are nodes. Three non-overlapping edge sets by physical distance d_ij: local 0 < d <= 150 um, mesoscopic 150 < d <= 300 um,
  macroscopic 300 < d <= 600 um, keeping at most 8, 12 and 16 nearest eligible neighbours per node. Edge distances are normalised and encoded with
  16 radial-basis-function features (`graph/builder.py`). Candidate radii considered: 50, 100, 150, 200, 300, 400, 500, 600 um.
- **Scale-aware PNA:** each scale has an independent two-layer PNA branch (hidden 128; mean, max, min, std aggregators; identity, amplification,
  attenuation scalers) producing a 64-d context vector. A learned softmax gate combines the patch representation with the three contexts, so each
  scale's contribution varies by location (`models/graph_model.py`, `models/graph_layers.py`).
- **Residual formulation:** y_hat = y_patch + delta_graph. During graph training the patch predictor is frozen and the final graph-correction
  layer is zero-initialised, so optimisation starts from the patch-only solution.

## 5. Evaluation protocol

- **Metric and genes:** Pearson correlation (PCC), computed per gene on held-out samples and averaged over genes within each cancer, on the official
  HEST-1K 50-gene highly variable panel (HVG50) of each cancer (`<cancer>/var_50genes.json`).
- **Pathology foundation-model baselines:** the official HEST-1K frozen-encoder protocol: embeddings reduced to 256 dimensions by PCA and fed to
  Ridge regression (`src/scripts/19_official_hest_benchmark.py`).
- **ePathST-E, ePathST and the ablation models:** evaluated with identical patient-level partitions and gene targets. All evaluation uses
  patient-disjoint partitions, so no patient appears in both training and evaluation. Model selection uses validation data only; the
  held-out HEST test partitions are used only for the final evaluation.
- **Training and test slides (Table III/V results):** per cancer, the models are trained on the HEST-1K cohort slides outside the held-out test
  partition (about 200 slides across the three cancers) and evaluated on the held-out HEST test partition (the official HEST-Benchmark test folds).
- **Hardware:** the full ePathST system and the ablations were evaluated on NVIDIA H200 GPUs.
- **Efficiency:** deployed parameter counts and encoder FLOPs measured for a single 224 x 224 patch under one protocol for student, teacher and
  foundation models; TFLOPs per WSI scales that by the patch count of one reference WSI.

## 6. Results

**Comparison with pathology foundation models (HEST-1K PCC; relative size normalised to ePathST, 5.25 M).**

| Method | Architecture | Params (M) | Rel. size | IDC | COAD | ccRCC | Average |
|---|---|---|---|---|---|---|---|
| GigaPath | ViT-G/14 | 1135 | 216.19x | 0.5515 | 0.2992 | 0.2430 | 0.3646 |
| H-Optimus-0 | ViT-g/14 | 1134 | 216.00x | **0.5976** | 0.3086 | 0.2676 | 0.3913 |
| GenBio-PathFM | ViT-G | 1133 | 215.94x | 0.5872 | 0.3284 | 0.2615 | **0.3924** |
| UNI2-h | ViT-H/14 | 681 | 129.71x | 0.5898 | 0.3015 | 0.2640 | 0.3851 |
| Virchow2 | ViT-H/14 | 632 | 120.38x | 0.5971 | 0.2581 | **0.2719** | 0.3757 |
| CONCH v1.5 | ViT-L/16 | 307 | 58.48x | 0.5440 | 0.2802 | 0.2176 | 0.3473 |
| Kaiko ViT-B/16 | ViT-B/16 | 86 | 16.38x | 0.5352 | 0.2812 | 0.2291 | 0.3485 |
| CTransPath | Swin-Tiny | 27.5 | 5.24x | 0.4993 | 0.2382 | 0.2362 | 0.3246 |
| ResNet50 | ResNet-50 | 25.5 | 4.86x | 0.4739 | 0.2500 | 0.2252 | 0.3164 |
| **ePathST-E** | frozen CNN | 3.39 | 0.65x | 0.4890 | 0.3151 | 0.2275 | 0.3438 |
| **ePathST** | frozen CNN + GNN | 5.25 | 1.00x | 0.5384 | **0.3302** | 0.2375 | 0.3687 |

ePathST keeps about 94% of H-Optimus-0's average PCC (0.3687 / 0.3913) with 215x fewer parameters and 1,231x less compute per WSI, and is the
best model on COAD.

**Ablation of spatial context (frozen ePathST-E encoder fixed).**

| Variant | Local | Meso | Macro | Scale-aware | IDC | COAD | ccRCC | Average |
|---|---|---|---|---|---|---|---|---|
| ePathST-E (patch only) | - | - | - | - | 0.4890 | 0.3151 | 0.2275 | 0.3438 |
| Local-only PNA | yes | - | - | - | 0.5161 | 0.3238 | 0.2316 | 0.3572 |
| Meso-only PNA | - | yes | - | - | 0.5237 | 0.3261 | 0.2328 | 0.3609 |
| Macro-only PNA | - | - | yes | - | 0.5194 | 0.3249 | 0.2341 | 0.3595 |
| Multiscale PNA | yes | yes | yes | - | 0.5272 | 0.3256 | 0.2337 | 0.3622 |
| **ePathST (scale-aware)** | yes | yes | yes | yes | **0.5384** | **0.3302** | **0.2375** | **0.3687** |

Scale-aware fusion gains 0.0249 average PCC over ePathST-E and 0.0065 over conventional multiscale aggregation.

**Compute-performance.** TFLOPs per WSI: ePathST-E 13.07 (0.99x), ePathST 13.15 (1.00x), ResNet50 48.56, CTransPath 53.30, Kaiko 399.53,
Virchow2 4,076.58, UNI2-h 4,393.78, Prov-GigaPath 5,408.90, CONCH v1.5 6,380.90, H-Optimus-0 7,011.44, GenBio-PathFM 16,191.64 (1,231.30x).
A log-linear OLS fit over the prior pathology models gives PCC = 0.0262 log10(TFLOPs) + 0.2767, R^2 = 0.78; at 13.15 TFLOPs the fit predicts 0.306,
against ePathST's 0.3687. No evaluated model has both higher PCC and lower compute than ePathST.

**Spatial-domain check (single sample).** On HEST sample TENX149, Leiden clustering of predicted expression against ground-truth clusters
gives ARI 0.58 (ePathST-E), 0.66 (ePathST) and 0.69 (H-Optimus-0). The evaluation of all test samples is in `docs/reports/HEST_LEIDEN_CLUSTERING_RESULTS.md`.

## 7. Supplementary analyses (documented in `docs/reports/RESULTS_INDEX.md`)

- **Leiden across all 32 HEST test samples**, with the three systems trained on the benchmark's own training slides in each official fold
  (benchmark-fold training); H-Optimus-0 uses the same MLP patch head on PCA-reduced features.
- **TCGA external validation** (BRCA, COAD, KIRC): predictions on 100 um hexagonal pseudo-spots from the same benchmark-fold models, with bulk RNA-seq,
  TCGA-CDR clinical data and CPE purity assembled in `companion/`.

## 8. Reproducibility map

| Stage | Scripts |
|---|---|
| Distillation | `01`-`07` |
| HEST features, custom 5-fold patient-level cross-validation | `08`-`18` |
| Official HEST protocol, ablation grid, Moran's I | `19`-`23` |
| TCGA validation, official-panel Leiden, audits | `24_tcga_validation.py`, `25_leiden_official_panel.py`, `26_hest_audits.py` |
| Paper figures | `figures/` |

Hyperparameters live in `configs/base.yaml`. Dated result snapshots are in `experimental_runs/`; larger outputs are indexed in `docs/reports/RESULTS_INDEX.md`.
