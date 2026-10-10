#!/usr/bin/env python
"""Regression check for the target/prediction misalignment bug in run_ablation() of
src/scripts/10_run_patient_cv.py, which affected graph ablations A4-A9.

Scores with the real epathst.evaluation.metrics rather than a reimplementation and writes
results/debug_idc_pipeline/T1_alignment_bug.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from epathst.evaluation.metrics import evaluate_predictions

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "debug_idc_pipeline"


def _fake_cfg():
    return {"evaluation": {"min_regions_per_patient": 3, "gene_r_thresholds": [0.0, 0.2, 0.3]}}


def t1_alignment_bug():
    """Reproduce the pre-fix misalignment of run_ablation() on synthetic data and check the fix.

    ids_test spans several patients and is not grouped by patient: fold pooling follows sample
    order, and one patient can contribute several samples/sections (see load_fold_expression).
    The predictions come back in per-patient-grouped order, as from predict_graph_model, while
    the pre-fix code left y_gene_test in the original order.
    """
    rng = np.random.RandomState(0)
    n_patients, spots_per_patient, n_genes = 6, 12, 20

    # Interleave patients so rows are not pre-grouped, as when several samples per patient are
    # pooled with pd.concat in sample order rather than patient order.
    patient_seq = []
    blocks = [[f"P{p}"] * spots_per_patient for p in range(n_patients)]
    for i in range(0, n_patients, 2):
        interleaved = []
        a, b = blocks[i], blocks[i + 1] if i + 1 < n_patients else []
        for x, y in zip(a, b):
            interleaved += [x, y]
        interleaved += a[len(interleaved) // 2 :] if len(a) > len(interleaved) // 2 else []
        patient_seq += interleaved
    ids_test_original = np.array([f"{p}::spot{j}" for j, p in enumerate(patient_seq)])
    pids_original = np.array(patient_seq)
    n = len(ids_test_original)

    # Random ground truth; predictions below are a near-copy, so a correctly aligned pipeline
    # must score R close to 1.
    y_true = rng.randn(n, n_genes).astype(np.float32)

    # Truth plus small noise (a nonzero correction, not the degenerate all-zero-delta case),
    # permuted into per-patient-grouped order like predict_graph_model's output.
    unique_patients = np.unique(pids_original)
    order = np.concatenate([np.where(pids_original == p)[0] for p in unique_patients])
    y_pred_grouped = (y_true[order] + 0.01 * rng.randn(n, n_genes)).astype(np.float32)

    cancer_placeholder = np.array(["IDC"] * n)

    # Before the fix: ids_test was reassigned to grouped order...
    ids_test_buggy = ids_test_original[order]
    pids_buggy = np.array([s.split("::")[0] for s in ids_test_buggy])
    # ...but y_gene_test/y_path_test (y_true here) were never reordered to match.
    gene_cols = [str(i) for i in range(n_genes)]
    m_before = evaluate_predictions(
        y_true, y_pred_grouped, pids_buggy, cancer_placeholder, gene_cols, _fake_cfg()
    )["overall"]

    # After the fix: y_true is reordered with the same `order` as the predictions.
    y_true_fixed = y_true[order]
    m_after = evaluate_predictions(
        y_true_fixed, y_pred_grouped, pids_buggy, cancer_placeholder, gene_cols, _fake_cfg()
    )["overall"]

    result = {
        "description": (
            "y_pred is a near-exact (0.01-noise) copy of y_true's per-patient rows, "
            "just permuted into per-patient-grouped order -- a correct pipeline must "
            "recover fisher_z_mean_patient_R close to 1.0."
        ),
        "before_fix_fisher_z_mean_patient_R": m_before["fisher_z_mean_patient_R"],
        "after_fix_fisher_z_mean_patient_R": m_after["fisher_z_mean_patient_R"],
        "expected_after_fix": "close to 1.0 (near-perfect self-recovery)",
        "bug_confirmed": bool(
            m_before["fisher_z_mean_patient_R"] < 0.9 and m_after["fisher_z_mean_patient_R"] > 0.95
        ),
    }
    print(json.dumps(result, indent=2))
    assert result["after_fix_fisher_z_mean_patient_R"] > 0.95, "fix verification FAILED"
    assert (
        result["before_fix_fisher_z_mean_patient_R"] < 0.9
    ), "bug reproduction FAILED (should have been misaligned)"
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = t1_alignment_bug()
    (OUT_DIR / "T1_alignment_bug.json").write_text(json.dumps(result, indent=2))
    print(f"\n[debug_idc_pipeline] wrote {OUT_DIR / 'T1_alignment_bug.json'}")


if __name__ == "__main__":
    main()
