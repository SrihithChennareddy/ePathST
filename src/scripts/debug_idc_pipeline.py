#!/usr/bin/env python
"""Standalone diagnostic for the HEST pipeline. Imports the same project
utilities the real experiment uses (sts.metrics) so this tests the actual
implementation, not a reimplementation of it.

T1 demonstrates and verifies the fix for a target/prediction misalignment
bug in src/scripts/10_run_patient_cv.py's run_ablation() that affected every
graph-based ablation (A4-A9): predict_graph_model() concatenates
predictions in per-patient-grouped order; ids_test was reassigned to match
that order, but y_gene_test/y_path_test (the ground-truth arrays passed
into evaluate_predictions) were not reordered to match -- silently
misaligning every graph ablation's reported metric.

Output: results/debug_idc_pipeline/T1_alignment_bug.json (before/after
demonstration of the bug and fix)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts.metrics import evaluate_predictions

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "debug_idc_pipeline"


def _fake_cfg():
    return {"evaluation": {"min_regions_per_patient": 3, "gene_r_thresholds": [0.0, 0.2, 0.3]}}


def t1_alignment_bug():
    """Reproduces the exact shape of src/scripts/10_run_patient_cv.py's
    run_ablation(): ids_test spans multiple patients NOT already grouped by
    patient (matching the real pipeline's fold-pooling order, which is
    grouped by SAMPLE, not necessarily by PATIENT when the same patient can
    contribute multiple samples/sections -- see load_fold_expression's
    docstring), gene_pred is produced in per-patient-GROUPED order (exactly
    what predict_graph_model does), and y_gene_test stays in the original
    order (exactly what the pre-fix code did)."""
    rng = np.random.RandomState(0)
    n_patients, spots_per_patient, n_genes = 6, 12, 20

    # Build a patient order that is NOT already grouped (interleaved),
    # matching the real risk: multiple samples/sections per patient pooled
    # via pd.concat in sample order, not patient order.
    patient_seq = []
    blocks = [[f"P{p}"] * spots_per_patient for p in range(n_patients)]
    # interleave two patients' spots to break any accidental pre-grouping
    for i in range(0, n_patients, 2):
        interleaved = []
        a, b = blocks[i], blocks[i + 1] if i + 1 < n_patients else []
        for x, y in zip(a, b):
            interleaved += [x, y]
        interleaved += a[len(interleaved) // 2:] if len(a) > len(interleaved) // 2 else []
        patient_seq += interleaved
    ids_test_original = np.array([f"{p}::spot{j}" for j, p in enumerate(patient_seq)])
    pids_original = np.array(patient_seq)
    n = len(ids_test_original)

    # Ground truth: a smooth per-patient signal so a real model should
    # recover it near-perfectly if predictions == truth.
    y_true = rng.randn(n, n_genes).astype(np.float32)

    # Simulate a residual graph model with a REAL (nonzero) but topology-
    # agnostic learned correction so this isn't the degenerate all-zero-delta
    # case -- predictions are a slightly noisy copy of truth, in
    # PER-PATIENT-GROUPED order (matching predict_graph_model's actual
    # concatenation order).
    unique_patients = np.unique(pids_original)
    order = np.concatenate([np.where(pids_original == p)[0] for p in unique_patients])
    y_pred_grouped = (y_true[order] + 0.01 * rng.randn(n, n_genes)).astype(np.float32)

    cancer_placeholder = np.array(["IDC"] * n)

    # --- BEFORE FIX: exact pre-fix src/scripts/10_run_patient_cv.py behavior --
    # ids_test reassigned to grouped order (as the pre-fix code did)...
    ids_test_buggy = ids_test_original[order]
    pids_buggy = np.array([s.split("::")[0] for s in ids_test_buggy])
    # ...but y_gene_test/y_path_test (here: y_true) were NEVER reordered.
    m_before = evaluate_predictions(y_true, y_pred_grouped, pids_buggy, cancer_placeholder,
                                     [str(i) for i in range(n_genes)], _fake_cfg())["overall"]

    # --- AFTER FIX: reorder y_true with the SAME `order` used for predictions ---
    y_true_fixed = y_true[order]
    m_after = evaluate_predictions(y_true_fixed, y_pred_grouped, pids_buggy, cancer_placeholder,
                                    [str(i) for i in range(n_genes)], _fake_cfg())["overall"]

    result = {
        "description": "y_pred is a near-exact (0.01-noise) copy of y_true's per-patient rows, "
                        "just permuted into per-patient-grouped order -- a correct pipeline must "
                        "recover fisher_z_mean_patient_R close to 1.0.",
        "before_fix_fisher_z_mean_patient_R": m_before["fisher_z_mean_patient_R"],
        "after_fix_fisher_z_mean_patient_R": m_after["fisher_z_mean_patient_R"],
        "expected_after_fix": "close to 1.0 (near-perfect self-recovery)",
        "bug_confirmed": bool(m_before["fisher_z_mean_patient_R"] < 0.9 and m_after["fisher_z_mean_patient_R"] > 0.95),
    }
    print(json.dumps(result, indent=2))
    assert result["after_fix_fisher_z_mean_patient_R"] > 0.95, "fix verification FAILED"
    assert result["before_fix_fisher_z_mean_patient_R"] < 0.9, "bug reproduction FAILED (should have been misaligned)"
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = t1_alignment_bug()
    (OUT_DIR / "T1_alignment_bug.json").write_text(json.dumps(result, indent=2))
    print(f"\n[debug_idc_pipeline] wrote {OUT_DIR / 'T1_alignment_bug.json'}")


if __name__ == "__main__":
    main()
