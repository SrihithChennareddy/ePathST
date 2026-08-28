"""Step 19 (Section 23): patient-level cross-validation, 5 folds x 3 seeds,
grouped by patient (never by section/spot) and never randomly splitting
spots within a patient. Falls back to fewer folds, with an explicit
documented reason, when a cohort has too few independent patients."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


def patient_level_kfold(patient_ids: list[str], n_splits_requested: int, min_folds: int) -> tuple[list[dict], int, str | None]:
    """Returns (folds, n_splits_used, adaptation_reason). Each fold dict:
    {fold, train_patients, test_patients}. Uses sklearn's GroupKFold with
    each patient as its own group (one row per patient here, so grouping is
    trivially by identity) -- this guarantees every patient's data lands in
    exactly one fold's test set."""
    unique_patients = sorted(set(patient_ids))
    n = len(unique_patients)
    reason = None
    n_splits = n_splits_requested
    if n < n_splits_requested:
        n_splits = max(min_folds, min(n, min_folds))
        reason = (f"cohort has only {n} independent patients (< requested {n_splits_requested} folds); "
                  f"reduced to {n_splits} folds to keep every test fold non-trivial")
    if n < n_splits:
        n_splits = max(2, n)
        reason = (reason or "") + f"; further reduced to {n_splits} because n_patients={n}"

    gkf = GroupKFold(n_splits=n_splits)
    X = np.arange(n).reshape(-1, 1)
    folds = []
    for k, (train_idx, test_idx) in enumerate(gkf.split(X, groups=unique_patients)):
        folds.append({
            "fold": k,
            "train_patients": sorted([unique_patients[i] for i in train_idx]),
            "test_patients": sorted([unique_patients[i] for i in test_idx]),
        })
    all_test = [p for f in folds for p in f["test_patients"]]
    assert sorted(all_test) == unique_patients, "folds do not partition all patients exactly once"
    return folds, n_splits, reason


def carve_validation_from_train(train_patients: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Splits a fold's train pool into (train, val) patient sets, patient-
    level, for early stopping / checkpoint selection -- never touches the
    held-out test patients."""
    rng = np.random.RandomState(seed)
    pids = sorted(train_patients)
    rng.shuffle(pids)
    n_val = max(1, int(round(len(pids) * val_frac)))
    return pids[n_val:], pids[:n_val]


def build_cv_grid(patient_ids_by_cancer: dict[str, list[str]], cfg: dict) -> dict:
    """Section 23: report pooled-across-cancers AND per-cancer folds.
    Builds folds independently per cancer type (so cohort-size adaptation
    is per-cancer, as required), then also returns the pooled patient list
    for a pooled analysis pass."""
    cv = cfg["cross_validation"]
    per_cancer = {}
    for cancer, pids in patient_ids_by_cancer.items():
        folds, n_used, reason = patient_level_kfold(pids, cv["n_outer_folds"], cv["min_folds_if_too_few_patients"])
        per_cancer[cancer] = {"folds": folds, "n_folds_used": n_used, "adaptation_reason": reason,
                               "n_patients": len(set(pids))}
    pooled_pids = sorted({p for pids in patient_ids_by_cancer.values() for p in pids})
    pooled_folds, pooled_n, pooled_reason = patient_level_kfold(pooled_pids, cv["n_outer_folds"],
                                                                 cv["min_folds_if_too_few_patients"])
    return {
        "per_cancer": per_cancer,
        "pooled": {"folds": pooled_folds, "n_folds_used": pooled_n, "adaptation_reason": pooled_reason,
                   "n_patients": len(pooled_pids)},
        "seeds": cv["seeds"],
    }
