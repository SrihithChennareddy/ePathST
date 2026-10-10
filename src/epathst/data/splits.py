"""Patient-level partitioning: the CPTAC train/val/test split and cross-validation folds.

Both are grouped by patient so no patient contributes patches or spots to more than one side of a
split. Also holds the expected-run bookkeeping and completeness gate for aggregating CV results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def cptac_patient_split(manifest: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """Split patients into train/val/test, stratified per cancer_type.

    `manifest` has one row per selected WSI (== per patient). Train and val use the configured
    fractions; test takes the remainder. Returns [patient_id, cancer_type, split].
    """
    sc = cfg["cptac_split"]
    rng = np.random.RandomState(seed)
    rows = []
    for cancer, sub in manifest.groupby("cancer_type"):
        pids = sub["patient_id"].tolist()
        rng.shuffle(pids)
        n = len(pids)
        n_train = int(round(n * sc["train_frac"]))
        n_val = int(round(n * sc["val_frac"]))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        train, val, test = pids[:n_train], pids[n_train : n_train + n_val], pids[n_train + n_val :]

        for pid in train:
            rows.append({"patient_id": pid, "cancer_type": cancer, "split": "train"})
        for pid in val:
            rows.append({"patient_id": pid, "cancer_type": cancer, "split": "val"})
        for pid in test:
            rows.append({"patient_id": pid, "cancer_type": cancer, "split": "test"})

    df = pd.DataFrame(rows)
    assert df["patient_id"].is_unique, "a patient appears more than once across splits"
    assert set(df["patient_id"]) == set(
        manifest["patient_id"]
    ), "split does not cover every selected patient"
    return df


def assert_no_patch_leakage(patch_manifest: pd.DataFrame, split_df: pd.DataFrame) -> None:
    """Every patch's patient_id must map to exactly one split; no patient spans two splits."""
    merged = patch_manifest.merge(
        split_df[["patient_id", "split"]],
        on="patient_id",
        how="left",
        validate="m:1",
    )
    assert merged["split"].notna().all(), "patches exist for patients missing from the split file"
    per_patient_splits = merged.groupby("patient_id")["split"].nunique()
    assert (per_patient_splits == 1).all(), "some patient's patches span more than one split"


def exclude_targeted_panel_samples(
    hest_samples: pd.DataFrame,
    min_genes: int = 2000,
) -> pd.DataFrame:
    """Drop samples on a targeted gene panel (e.g. Xenium, ~541 genes); keep whole-transcriptome ST.

    A targeted panel would collapse the cross-sample HVG1000 gene intersection. `min_genes` sits
    between the targeted (541) and whole-transcriptome (14700+) panels seen in this cohort. The
    gene count is read with h5py instead of a full anndata load to stay cheap over hundreds of
    samples.
    """
    import h5py

    def n_genes(path: str) -> int:
        with h5py.File(path, "r") as f:
            return f["var"]["_index"].shape[0]

    counts = hest_samples["expression_path"].map(n_genes)
    keep = counts >= min_genes
    dropped = hest_samples.loc[~keep]
    if len(dropped):
        print(
            f"[exclude_targeted_panel_samples] dropping {len(dropped)} "
            f"targeted-gene-panel sample(s) (<{min_genes} genes) from the CV split: "
            f"{sorted(dropped['sample_id'].tolist())}"
        )
    return hest_samples.loc[keep].reset_index(drop=True)


def patient_level_kfold(
    patient_ids: list[str],
    n_splits_requested: int,
    min_folds: int,
) -> tuple[list[dict], int, str | None]:
    """Returns (folds, n_splits_used, adaptation_reason).

    Each fold is {fold, train_patients, test_patients}. GroupKFold with each patient as its own
    group guarantees every patient lands in exactly one fold's test set. When the cohort has fewer
    patients than requested folds, the fold count is reduced and the reason is returned.
    """
    unique_patients = sorted(set(patient_ids))
    n = len(unique_patients)
    reason = None
    n_splits = n_splits_requested
    if n < n_splits_requested:
        n_splits = max(min_folds, min(n, min_folds))
        reason = (
            f"cohort has only {n} independent patients "
            f"(< requested {n_splits_requested} folds); reduced to {n_splits} folds "
            f"to keep every test fold non-trivial"
        )
    if n < n_splits:
        n_splits = max(2, n)
        reason = (reason or "") + f"; further reduced to {n_splits} because n_patients={n}"

    gkf = GroupKFold(n_splits=n_splits)
    X = np.arange(n).reshape(-1, 1)
    folds = []
    for k, (train_idx, test_idx) in enumerate(gkf.split(X, groups=unique_patients)):
        folds.append(
            {
                "fold": k,
                "train_patients": sorted([unique_patients[i] for i in train_idx]),
                "test_patients": sorted([unique_patients[i] for i in test_idx]),
            }
        )
    all_test = [p for f in folds for p in f["test_patients"]]
    assert sorted(all_test) == unique_patients, "folds do not partition all patients exactly once"
    return folds, n_splits, reason


def carve_validation_from_train(
    train_patients: list[str],
    val_frac: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split a fold's train pool into (train, val) patients for early stopping/checkpoint choice.

    Never touches the held-out test patients.
    """
    rng = np.random.RandomState(seed)
    pids = sorted(train_patients)
    rng.shuffle(pids)
    n_val = max(1, int(round(len(pids) * val_frac)))
    return pids[n_val:], pids[:n_val]


def build_cv_grid(patient_ids_by_cancer: dict[str, list[str]], cfg: dict) -> dict:
    """Build folds independently per cancer type, plus a pooled-across-cancers set.

    Per-cancer folds let cohort-size adaptation happen per cancer; the pooled set is for a
    secondary analysis pass.
    """
    cv = cfg["cross_validation"]
    per_cancer = {}
    for cancer, pids in patient_ids_by_cancer.items():
        folds, n_used, reason = patient_level_kfold(
            pids,
            cv["n_outer_folds"],
            cv["min_folds_if_too_few_patients"],
        )
        per_cancer[cancer] = {
            "folds": folds,
            "n_folds_used": n_used,
            "adaptation_reason": reason,
            "n_patients": len(set(pids)),
        }
    pooled_pids = sorted({p for pids in patient_ids_by_cancer.values() for p in pids})
    pooled_folds, pooled_n, pooled_reason = patient_level_kfold(
        pooled_pids,
        cv["n_outer_folds"],
        cv["min_folds_if_too_few_patients"],
    )
    return {
        "per_cancer": per_cancer,
        "pooled": {
            "folds": pooled_folds,
            "n_folds_used": pooled_n,
            "adaptation_reason": pooled_reason,
            "n_patients": len(pooled_pids),
        },
        "seeds": cv["seeds"],
        # `per_cancer` is primary, `pooled` secondary; aggregation should key off this flag.
        "primary_analysis_unit": "per_cancer",
    }


def expected_run_keys(
    cv_grid: dict,
    ablation_names: list[str],
    unit: str = "per_cancer",
) -> set[tuple]:
    """Every (cancer, fold, seed, ablation) key the CV design calls for.

    For `unit='pooled'` the keys are (fold, seed, ablation). Uses the actual n_folds_used per
    cancer; this is the "expected" side of assert_run_completeness.
    """
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
    """Hard completeness gate: print the included runs, then assert completed == expected.

    Raises instead of warning: aggregation must fail loudly rather than silently proceed on a
    partial run, as it once did (see docs/reports/POC5_RESULTS.md).
    """
    print(
        f"[assert_run_completeness{f' ({context})' if context else ''}] "
        f"{len(completed)}/{len(expected)} expected runs present"
    )
    for key in sorted(completed):
        print(f"  included: {key}")
    missing = expected - completed
    extra = completed - expected
    if missing:
        print(f"[assert_run_completeness] MISSING {len(missing)} run(s): {sorted(missing)}")
    if extra:
        print(
            f"[assert_run_completeness] UNEXPECTED {len(extra)} extra run(s) "
            f"not in the CV design: {sorted(extra)}"
        )
    assert completed == expected, (
        f"run completeness check FAILED -- {len(missing)} missing, {len(extra)} unexpected "
        f"(see printed lists above). Refusing to aggregate a partial/mismatched set of runs."
    )
