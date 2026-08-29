"""Step 19 (Section 23), revised per Final_Project_V2.txt Phase 3: patient-
level cross-validation, 5 folds x 3 seeds, grouped by patient (never by
section/spot) and never randomly splitting spots within a patient. Falls
back to fewer folds, with an explicit documented reason, when a cohort has
too few independent patients.

V2 Phase 3 adds a hard completeness requirement, quoted directly because
it names the exact failure this project already hit once (PoC_5's
cross-fold aggregate silently pooling only folds 0-2 out of 4 completed
folds -- see docs/POC5_RESULTS.md): "Before final aggregation, your code
should assert something equivalent to: expected = n_folds * n_seeds;
assert completed_runs == expected, and print every included cancer/fold/
seed/model before calculating the final statistics." `assert_run_completeness`
below is that check -- every aggregation script MUST call it before
computing any cross-fold statistic."""
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
        # V2 Phase 1 / "Make the three cancers separate primary experiments":
        # per-cancer results are the primary analysis unit; `pooled` is
        # reported secondarily. Aggregation/reporting code should key off
        # this flag rather than hard-coding the assumption.
        "primary_analysis_unit": "per_cancer",
    }


def expected_run_keys(cv_grid: dict, ablation_names: list[str], unit: str = "per_cancer") -> set[tuple]:
    """Enumerates every (cancer, fold, seed, ablation) -- or, for
    `unit='pooled'`, every (fold, seed, ablation) -- combination the CV
    design calls for, given the ACTUAL (possibly cohort-size-adapted)
    n_folds_used per cancer. This is the "expected" side of the V2 Phase 3
    completeness assertion."""
    seeds = cv_grid["seeds"]
    keys = set()
    if unit == "per_cancer":
        for cancer, info in cv_grid["per_cancer"].items():
            for fold in info["folds"]:
                for seed in seeds:
                    for ab in ablation_names:
                        keys.add((cancer, fold["fold"], seed, ab))
    else:
        for fold in cv_grid["pooled"]["folds"]:
            for seed in seeds:
                for ab in ablation_names:
                    keys.add((fold["fold"], seed, ab))
    return keys


def assert_run_completeness(expected: set[tuple], completed: set[tuple], context: str = "") -> None:
    """V2 Phase 3's required hard gate: `assert completed_runs == expected`,
    printing every included (cancer, fold, seed, model) row first so a
    partial run is loud and visible, never silently pooled as if it were
    complete (this is the exact PoC_5 bug -- see module docstring).
    Raises AssertionError (not a soft warning) on any mismatch, by design:
    V2 explicitly wants aggregation to FAIL rather than quietly proceed."""
    print(f"[assert_run_completeness{f' ({context})' if context else ''}] "
          f"{len(completed)}/{len(expected)} expected runs present")
    for key in sorted(completed):
        print(f"  included: {key}")
    missing = expected - completed
    extra = completed - expected
    if missing:
        print(f"[assert_run_completeness] MISSING {len(missing)} run(s): {sorted(missing)}")
    if extra:
        print(f"[assert_run_completeness] UNEXPECTED {len(extra)} extra run(s) not in the CV design: {sorted(extra)}")
    assert completed == expected, (
        f"run completeness check FAILED -- {len(missing)} missing, {len(extra)} unexpected "
        f"(see printed lists above). Refusing to aggregate a partial/mismatched set of runs."
    )
