"""Patient-level 80/10/10 CPTAC split, stratified by cancer type, computed
before any patch is touched. No patient's patches may appear in more than
one split."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cptac_patient_split(manifest: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """`manifest` must have one row per selected WSI (== one row per
    patient). Returns a DataFrame with columns
    [patient_id, cancer_type, split] where split in {train, val, test},
    stratified per cancer_type using the configured fractions."""
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
    """Every patch's patient_id must map to exactly one split, and no
    patient may contribute patches to more than one split."""
    merged = patch_manifest.merge(
        split_df[["patient_id", "split"]], on="patient_id", how="left", validate="m:1",
    )
    assert merged["split"].notna().all(), (
        "patches exist for patients missing from the split file"
    )
    per_patient_splits = merged.groupby("patient_id")["split"].nunique()
    assert (per_patient_splits == 1).all(), "some patient's patches span more than one split"
