# PathDistill-ST: Lightweight Spatial Transcriptomics Prediction via Pathology Foundation Model Distillation

Abstract — Spatial transcriptomics enables spatially resolved molecular profiling of tissue but remains costly and difficult to scale, motivating methods that infer molecular states directly from routine histopathology. We propose PathDistill-ST, a lightweight framework that compresses pathology foundation-model knowledge into a compact vision encoder and combines it with explicit multiscale spatial graph reasoning. A ~3.7M-parameter ConvNeXt V2-Atto student is distilled from the ~1.1B-parameter GenBio-PathFM using patient-unique CPTAC whole-slide images (WSIs) from breast, colon, and kidney cancers. The distilled encoder is then applied to paired H&E–spatial transcriptomics data from HEST, where each spatial location is represented by a compact morphology embedding and connected to neighboring regions through a physically defined multiscale graph. A lightweight PNA-based graph neural network predicts spatial gene expression and pathway activity from these representations. To isolate the contribution of spatial organization, we compare against patch-only models, parameter-matched nonspatial baselines, random graphs, degree-preserving shuffled graphs, and progressive spatial-edge destruction. PathDistill-ST tests whether foundation-model knowledge can be compressed by nearly 300× while preserving useful pathology representations and whether explicit tissue topology contributes molecular information beyond local morphology. Overall, PathDistill-ST provides a computationally efficient framework for scalable molecular prediction from routine histology, targeting competitive predictive accuracy with substantially reduced model size, inference time, and computational resources.

## Pipeline overview

The `scripts/` are numbered end-to-end and split into two phases: distilling the student encoder on CPTAC, then applying it downstream on HEST spatial transcriptomics.

**Phase 1 — CPTAC distillation**

1. `01_select_cptac_wsi.py` — select one WSI per patient from the CPTAC root/metadata (breast, colon, kidney)
2. `02_extract_cptac_patches.py` — tissue-masked, spatially sampled 224×224 patch extraction
3. `03_cptac_patient_split.py` — patient-level 80/10/10 split, per seed
4. `04_run_teacher_genbio.py` — run the frozen GenBio-PathFM teacher once over every selected patch
5. `05_train_student_distill.py` — train the ConvNeXt V2-Atto student(s) distilled from the teacher
6. `06_eval_distillation.py` — evaluate the student(s) on held-out CPTAC test patients
7. `07_lock_student.py` — lock the final student checkpoint at the chosen hyperparameters

**Phase 2 — HEST downstream**

8. `08_prepare_hest_metadata.py` — audit HEST metadata across the IDC/COAD/CCRCC cohorts
9. `09_extract_student_features_hest.py` — run the locked student encoder over HEST patches
10. `10_run_patient_cv.py` — main patient-level cross-validation: patch-only vs. spatial PNA-GNN models
11. `11_spatial_destruction_doseresponse.py` — progressive spatial-edge destruction dose-response
12. `12_primary_hypothesis_test.py` — aggregate per-patient Fisher-z Pearson correlations for the primary hypothesis test
13. `13_additional_analyses.py` — multiscale contribution and Moran's I vs. graph structure
14. `14_model_efficiency.py` — parameter counts, compression ratio, FLOPs/MACs
15. `15_genbio_fair_comparison.py` — fair comparison against GenBio-PathFM itself on the same HEST cohort
16. `16_sanity_checks.py` — sanity checks that must pass before launching full runs
17. `17_make_results_tables_and_figures.py` — final results tables and figures across HVG settings
18. `18_final_interpretation.py` — final scientific interpretation of the results
