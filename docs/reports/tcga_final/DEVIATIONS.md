# Deviations

## 2026-10-09 — BRCA ensemble rebuilt without IDC fold 2 (seed 0)

- **What changed:** the TCGA-BRCA predictions for E, P and T are rebuilt as the average of IDC folds 0, 1 and 3; fold 2 is excluded for all three systems so they average the same folds. Frozen inference from the cached embeddings with the saved bundles, no training. Files in `predictions_rebuilt/TCGA-BRCA/`; the original 4-fold files are unchanged.
- **Why:** Step 1 check A7. The ePathST-E head of IDC fold 2, seed 0 collapsed (held-out test PCC 0.011 averaged over genes; median prediction SD 0.09 x target SD). The frozen rule refers to validation predictions, which the HEST runs did not store, so the decision uses the held-out test diagnostics; the user decided on that basis.
- **Outcomes seen:** no. Decided by the user on 2026-10-09, before any TCGA correlation, hazard ratio, C-index or survival curve was computed.
- **Also noted (not acted on):** H-Optimus-0 for ccRCC fold 4, seed 1 has test PCC -0.024 (seed 1 is not used for the TCGA ensemble).
- **Label-free half of the rule checked too:** the median-prediction-SD criterion (`< 0.01 x target SD`, needs no RNA/outcome labels) was computed for all 72 fold x seed x system models; none trip it (`flag_sd_lt_0.01` all False in `audit/A7_collapse.csv`). So no additional collapse is hiding behind a constant-output model, and nothing else needs rebuilding.
- **Step 9 plan (not yet executed):** the HEST same-head table will report two versions — all folds, and with IDC fold 2/seed 0 (E) and ccRCC fold 4/seed 1 (T) excluded — so the seed-1 collapse is visible without silently dropping it from the headline number.

## 2026-10-09 — A7 evaluated on test predictions, not validation

- **What changed:** none; recorded for the record. The pre-registered rule says "validation PCC"; validation-split predictions were never saved for the HEST same-head runs, so `50_tcga_audit.py` evaluated the PCC half of the rule on held-out test predictions instead.
- **Why:** data availability, not a result-driven choice — the SD half of the rule (label-free) was computed exactly as specified and found nothing; only the PCC half substitutes test for validation, and it only confirms the single already-known IDC fold 2/seed 0 collapse (A6: TCGA ensembles use seed 0) plus the seed-1-only ccRCC fold 4 case above.
- **Outcomes seen:** no.

## 2026-10-09 — A8 (same spots across E/P/T) verified by code inspection, not a separate audit check

- **What changed:** `50_tcga_audit.py` does not emit a standalone A8 check. Inspection of `epathst/tcga/prediction.py::predict_from_embeddings` shows all three systems are scored against the same `coords_um` array within one function call, so their spot sets cannot differ by construction — there is no code path that produces different spots per system.
- **Fix:** added `assert all(p.shape[0] == n for p in preds.values())` at the end of `predict_from_embeddings` so a future refactor that broke this guarantee would fail loudly instead of silently.
- **Outcomes seen:** no.

## 2026-10-10 — Step 8 KIRC Stage III subgroup: tau lowered by Step 5.4's rule

- **What changed:** the KIRC Stage III subgroup analysis (8.6) ran at tau=4.00 years (1461.0 days) instead of the nominal 5-year horizon.
- **Why:** this subgroup has only 117 patients / 47 events; Step 5.4's rule (>=10% TE at risk, tau<=TR max follow-up, >=1 TE patient with time<tau in every outer split) failed at 5 years and was satisfied at 4. This is `56_survival_exploratory.py`'s reuse of `54_tcga_survival.py`'s `pick_tau`/`run_cohort`, which lowers tau automatically; this entry was added after the run completed because the subgroup path didn't wire its deviation list back to this file (a gap in `56_survival_exploratory.py`, noted for completeness, not re-run for it alone since the result itself is already correct).
- **Outcomes seen:** logged after the run. Tau was lowered automatically by the pre-registered Step 5.4 rule, based on follow-up only; no choice was made after seeing outcomes.

## 2026-10-14 — ΔC corrected t-test: J=25, not 50, for C vs C+P comparisons

- **What changed:** `delta_table` (54) and the sensitivity deltas (54c) paired splits on (cohort, repeat, fold) across both the full and RNA cohorts. Every survival-eligible patient has RNA, so the two cohorts are identical, and C+P vs C was tested on 50 "splits" (each counted twice). Comparisons are now paired within one cohort (full when both sets are fit there, else RNA), J=25, df=24, as pre-registered. Rebuilt from the saved per-split C-indices (`--deltas-only`), no refitting.
- **Effect:** point estimates unchanged; CIs wider, p higher for C+P vs C. BRCA primary 0.0163 [-0.031, 0.063], p=0.484 (was p=0.460); KIRC 0.0197 [0.003, 0.036], p=0.022 (was 0.014). Other comparisons and U3 unchanged. Old tables in `survival/_superseded_J50_20261014/` and `survival_sensitivity/_superseded_J50_20261014/`.
- **Outcomes seen:** yes — a coding error against the pre-registered test, fixed regardless of direction; the primary test is null either way.

## 2026-10-15 — Post-hoc null controls (not pre-registered)

- **What:** `60_null_controls.py`, added after external review flagged the lack of a negative control. (a) Transfer: 1,000 shuffled-patient permutations of the predicted matrix (one permutation across all genes), same Spearman + BH over 150 pairs as Step 4; null of #genes q<0.05. (b) KIRC: 200 permutations of the predicted-gene rows across patients with age, stage and outcome fixed, rerunning the full Step 5 C/C+P nested CV and the Step 5.6 independence fit; null of the P-score HR, LR statistic and ΔC.
- **Why:** the Step 5.6 score is the out-of-fold C+P risk, which contains age and stage, so HR 2.54 could partly reflect the clinical terms; (b) tests that directly.
- **Outcomes seen:** yes — added after all primary and secondary results were known. Reported as a supplementary control; it does not change any pre-registered result.
