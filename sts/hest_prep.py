"""Steps 10-12 (Sections 13-15): HEST metadata audit, patient-level fold-
safety bookkeeping, and ST-spot-centered H&E patch extraction with
mandatory visual alignment QC."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HEST_SAMPLE_COLUMNS = ["sample_id", "patient_id", "study", "cancer", "organ", "species",
                        "st_technology", "he_path", "expression_path", "mpp"]


def audit_hest_samples(raw_meta: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Section 13: keep only human cancer H&E+ST samples for
    {IDC/breast, COAD/colon, CCRCC/kidney}; drop mouse, incompatible
    non-cancer tissue, and duplicates. Returns (clean_df, report)."""
    df = raw_meta.copy()
    n0 = len(df)
    reasons = {}

    mouse_mask = df["species"].str.lower().eq("mouse") if "species" in df else pd.Series(False, index=df.index)
    reasons["mouse"] = int(mouse_mask.sum())
    df = df[~mouse_mask]

    target_cancers = set(cfg["cancers"]["hest"])
    cancer_mask = df["cancer"].isin(target_cancers)
    reasons["cancer_not_in_target_set"] = int((~cancer_mask).sum())
    df = df[cancer_mask]

    dup_mask = df.duplicated(subset=["sample_id"], keep="first")
    reasons["duplicate_sample_id"] = int(dup_mask.sum())
    df = df[~dup_mask]

    report = {"n_initial_samples": n0, "n_final_samples": len(df), "exclusion_reasons": reasons,
              "by_cancer": df["cancer"].value_counts().to_dict() if len(df) else {}}
    return df.reset_index(drop=True), report


def enforce_patient_section_grouping(df: pd.DataFrame) -> pd.DataFrame:
    """Section 13: 'If multiple sections originate from one patient, they
    MUST stay in the same train/test fold.' No-op transform here -- this
    just asserts the grouping key exists; sts.cv_folds groups by
    `patient_id`, never `sample_id`, enforcing the constraint at split time."""
    assert "patient_id" in df.columns and df["patient_id"].notna().all()
    return df


def alignment_qc_sample(he_reader, spot_coords_px: np.ndarray, spot_ids: list[str], mpp: float, cfg: dict,
                         seed: int = 0) -> dict:
    """Section 15: mandatory alignment check -- for a random subsample of
    spots, extract the ST-centered patch and confirm it overlaps tissue
    (not background), i.e. registration is plausible. Returns per-spot
    tissue-fraction so a human/CI check can flag misalignment (near-zero
    tissue fraction across many spots implies systematic misregistration)."""
    from sts.cptac_select import tissue_mask_from_thumbnail

    n = cfg["hest"]["n_alignment_qc_samples"]
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(spot_ids), size=min(n, len(spot_ids)), replace=False)

    rows = []
    for i in idx:
        patch = he_reader.read_region_at_mpp(
            spot_coords_px[i, 0], spot_coords_px[i, 1],
            physical_size_um=cfg["hest"]["patch_um"], out_px=cfg["hest"]["patch_px"], source_mpp=mpp,
        )
        tf = float(tissue_mask_from_thumbnail(patch).mean())
        rows.append({"spot_id": spot_ids[i], "tissue_fraction": tf, "x_px": spot_coords_px[i, 0],
                     "y_px": spot_coords_px[i, 1]})
    df = pd.DataFrame(rows)
    return {
        "n_sampled": len(df),
        "mean_tissue_fraction": float(df["tissue_fraction"].mean()) if len(df) else np.nan,
        "pct_spots_with_tissue": float((df["tissue_fraction"] > 0.05).mean()) if len(df) else np.nan,
        "likely_misaligned": bool(len(df) and (df["tissue_fraction"] > 0.05).mean() < 0.5),
        "per_spot": df.to_dict(orient="records"),
    }


def extract_hest_patches_for_sample(he_reader, spot_ids: list[str], coords_px: np.ndarray, mpp: float,
                                     cfg: dict, n_threads: int = 8) -> np.ndarray:
    """Step 12: 224x224 patch centered on every ST spot at ~0.5 um/px.
    Reads spots concurrently via a thread pool on the one shared WSI
    handle -- openslide's C `read_region` releases the GIL during the
    actual tile decode, so this is genuine I/O-bound parallelism (each
    thread mostly waits on disk/decoded-tile fetches), not just Python-
    level interleaving. Each thread writes to its own disjoint index of
    `patches`, so no locking is needed."""
    from concurrent.futures import ThreadPoolExecutor

    patches = np.empty((len(spot_ids), cfg["hest"]["patch_px"], cfg["hest"]["patch_px"], 3), dtype=np.uint8)

    def _read_one(i: int) -> None:
        patches[i] = he_reader.read_region_at_mpp(
            coords_px[i, 0], coords_px[i, 1], physical_size_um=cfg["hest"]["patch_um"],
            out_px=cfg["hest"]["patch_px"], source_mpp=mpp,
        )

    with ThreadPoolExecutor(max_workers=max(1, n_threads)) as pool:
        list(pool.map(_read_one, range(len(spot_ids))))
    return patches


def integrity_check_spot_alignment(spot_ids: list[str], patch_ids: list[str], coords: np.ndarray,
                                    expr_index: pd.Index) -> dict:
    """Mandatory integrity assertion (Section 16): spot_id <-> patch <->
    coordinate <-> expression vector must remain exactly aligned."""
    ok_patch = list(spot_ids) == list(patch_ids)
    ok_expr = list(spot_ids) == list(expr_index)
    ok_coords = coords.shape[0] == len(spot_ids)
    result = {"patch_alignment_ok": ok_patch, "expression_alignment_ok": ok_expr, "coord_count_ok": ok_coords}
    result["all_ok"] = all(result.values())
    return result
