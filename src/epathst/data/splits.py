"""Patient-level partitioning used throughout ePathST: the initial CPTAC train/val/test
split plus cross-validation fold construction, both grouped by patient to prevent leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def cptac_patient_split(manifest: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """`manifest` has one row per selected WSI (== per patient); returns [patient_id, cancer_type, split] stratified per cancer_type using the configured fractions."""
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
        train, val, test = pids[:n_train], pids[n_train:n_train + n_val], pids[n_train + n_val:]

        for pid in train:
            rows.append({"patient_id": pid, "cancer_type": cancer, "split": "train"})
        for pid in val:
            rows.append({"patient_id": pid, "cancer_type": cancer, "split": "val"})
        for pid in test:
            rows.append({"patient_id": pid, "cancer_type": cancer, "split": "test"})

    df = pd.DataFrame(rows)
    assert df["patient_id"].is_unique, "a patient appears more than once across splits"
    assert set(df["patient_id"]) == set(manifest["patient_id"]), (
        "split does not cover every selected patient"
    )
    return df


def assert_no_patch_leakage(patch_manifest: pd.DataFrame, split_df: pd.DataFrame) -> None:
    """Every patch's patient_id must map to exactly one split, and no patient may contribute patches to more than one split."""
    merged = patch_manifest.merge(
        split_df[["patient_id", "split"]], on="patient_id", how="left", validate="m:1",
    )
    assert merged["split"].notna().all(), (
        "patches exist for patients missing from the split file"
    )
    per_patient_splits = merged.groupby("patient_id")["split"].nunique()
    assert (per_patient_splits == 1).all(), "some patient's patches span more than one split"


def exclude_targeted_panel_samples(
    hest_samples: pd.DataFrame, min_genes: int = 2000,
) -> pd.DataFrame:
    """Drops samples on a targeted gene panel (e.g. Xenium, ~541 genes) rather than whole-transcriptome ST, since outliers would collapse the cross-sample HVG1000 intersection; `min_genes` sits between real targeted (541) and whole-transcriptome (14700+) panels seen in this cohort; reads gene count via h5py, not a full anndata load, to stay cheap over hundreds of samples."""
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
    patient_ids: list[str], n_splits_requested: int, min_folds: int,
) -> tuple[list[dict], int, str | None]:
    """Returns (folds, n_splits_used, adaptation_reason); each fold is {fold, train_patients, test_patients}. Uses GroupKFold with each patient as its own group, guaranteeing every patient lands in exactly one fold's test set."""
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
        folds.append({
            "fold": k,
            "train_patients": sorted([unique_patients[i] for i in train_idx]),
            "test_patients": sorted([unique_patients[i] for i in test_idx]),
        })
    all_test = [p for f in folds for p in f["test_patients"]]
    assert sorted(all_test) == unique_patients, "folds do not partition all patients exactly once"
    return folds, n_splits, reason


def carve_validation_from_train(
    train_patients: list[str], val_frac: float, seed: int,
) -> tuple[list[str], list[str]]:
    """Splits a fold's train pool into (train, val) patient sets for early stopping/checkpoint selection -- never touches the held-out test patients."""
    rng = np.random.RandomState(seed)
    pids = sorted(train_patients)
    rng.shuffle(pids)
    n_val = max(1, int(round(len(pids) * val_frac)))
    return pids[n_val:], pids[:n_val]


def build_cv_grid(patient_ids_by_cancer: dict[str, list[str]], cfg: dict) -> dict:
    """Builds folds independently per cancer type (so cohort-size adaptation is per-cancer), plus a pooled-across-cancers fold set for a secondary analysis pass."""
    cv = cfg["cross_validation"]
    per_cancer = {}
    for cancer, pids in patient_ids_by_cancer.items():
        folds, n_used, reason = patient_level_kfold(
            pids, cv["n_outer_folds"], cv["min_folds_if_too_few_patients"],
        )
        per_cancer[cancer] = {
            "folds": folds, "n_folds_used": n_used, "adaptation_reason": reason,
            "n_patients": len(set(pids)),
        }
    pooled_pids = sorted({p for pids in patient_ids_by_cancer.values() for p in pids})
    pooled_folds, pooled_n, pooled_reason = patient_level_kfold(
        pooled_pids, cv["n_outer_folds"], cv["min_folds_if_too_few_patients"],
    )
    return {
        "per_cancer": per_cancer,
        "pooled": {
            "folds": pooled_folds, "n_folds_used": pooled_n, "adaptation_reason": pooled_reason,
            "n_patients": len(pooled_pids),
        },
        "seeds": cv["seeds"],
        # `per_cancer` is the primary analysis unit, `pooled` secondary; aggregation code should key off this flag, not assume.
        "primary_analysis_unit": "per_cancer",
    }


def expected_run_keys(
    cv_grid: dict, ablation_names: list[str], unit: str = "per_cancer",
) -> set[tuple]:
    """Enumerates every (cancer, fold, seed, ablation) -- or, for `unit='pooled'`, (fold, seed, ablation) -- combination the CV design calls for given the actual n_folds_used per cancer; the "expected" side of the completeness assertion below."""
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
    """Hard completeness gate: prints included rows then asserts completed == expected, raising (not warning) on mismatch -- aggregation must fail loudly, not silently proceed on a partial run like it once did (see docs/POC5_RESULTS.md)."""
    print(f"[assert_run_completeness{f' ({context})' if context else ''}] "
          f"{len(completed)}/{len(expected)} expected runs present")
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
