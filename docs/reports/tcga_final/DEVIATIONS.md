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
