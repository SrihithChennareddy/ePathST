"""Step 1 (Final_Project.txt Section 4): select one high-quality CPTAC WSI
per patient.

Design: first apply hard exclusions, then rank remaining candidates per
patient by the deterministic composite score

    Q = 0.35*usable_tissue + 0.25*focus + 0.20*artifact_free
        + 0.10*stain_quality + 0.10*representativeness

computed from a low-res thumbnail (global tissue/stain QC) plus a sample of
50-100 ~20x tissue tiles (focus/artifact QC). Everything here is
torch-free/openslide-only so it can run on a CPU QC node ahead of any GPU
distillation work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class WSIQualityRecord:
    wsi_id: str
    patient_id: str
    cancer_type: str
    wsi_path: str
    specimen_type: str | None
    mpp: float | None
    width: int | None
    height: int | None
    tissue_area_mm2: float = np.nan
    tissue_fraction: float = np.nan
    focus_score: float = np.nan
    blurry_tissue_fraction: float = np.nan
    artifact_fraction: float = np.nan
    stain_score: float = np.nan
    representativeness_score: float = np.nan
    qc_score: float = np.nan
    excluded: bool = False
    exclusion_reason: str | None = None
    selection_reason: str | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- tissue --
def tissue_mask_from_thumbnail(thumb_rgb: np.ndarray, sat_thresh: int = 15, val_thresh: int = 235) -> np.ndarray:
    """Simple, dependency-light tissue mask: HSV saturation/value
    thresholding (glass background is low-saturation, near-white; tissue is
    saturated and darker). Good enough for global tissue-fraction QC; the
    same routine is reused for tile sampling and for Step 2 patch masking."""
    import colorsys
    arr = thumb_rgb.astype(np.float32) / 255.0
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    sat = np.where(maxc > 0, (maxc - minc) / np.clip(maxc, 1e-6, None), 0.0) * 255.0
    val = maxc * 255.0
    mask = (sat > sat_thresh) & (val < val_thresh)
    return mask


def focus_score_laplacian(tile_gray: np.ndarray) -> float:
    """Variance of Laplacian: standard, cheap focus/blur metric. Higher =
    sharper. Implemented with a 3x3 kernel convolution to avoid a hard
    cv2/scipy dependency; scipy is already a required package (Section 3),
    so we use scipy.ndimage directly."""
    from scipy.ndimage import laplace
    lap = laplace(tile_gray.astype(np.float32))
    return float(lap.var())


def artifact_fraction_tile(tile_rgb: np.ndarray) -> float:
    """Fraction of a tile flagged as artifact: pen marks (highly saturated,
    non-tissue hue), folds/tears (near-uniform very-dark or very-bright
    patches), air bubbles. Heuristic, deterministic, and documented as such
    -- Section 4 asks for an artifact-fraction QC signal, not a learned
    artifact detector."""
    hsv_sat = _rgb_to_sat(tile_rgb)
    pen_like = hsv_sat > 200  # oversaturated ink
    very_dark = tile_rgb.mean(axis=-1) < 20
    very_bright = tile_rgb.mean(axis=-1) > 250
    artifact = pen_like | very_dark | very_bright
    return float(artifact.mean())


def _rgb_to_sat(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    maxc = arr.max(axis=-1)
    minc = arr.min(axis=-1)
    return np.where(maxc > 0, (maxc - minc) / np.clip(maxc, 1e-6, None), 0.0) * 255.0


def stain_outlier_score(tile_rgbs: list[np.ndarray], reference_mean: np.ndarray | None = None) -> float:
    """Mean-color distance from a reference/global stain centroid, in
    normalized [0,1] where 0 = typical, 1 = extreme outlier. If no
    reference is given, uses this WSI's own tile-mean color (so the score
    reflects intra-slide stain *consistency*, not absolute H&E hue)."""
    means = np.stack([t.reshape(-1, 3).mean(axis=0) for t in tile_rgbs])
    ref = reference_mean if reference_mean is not None else means.mean(axis=0)
    dist = np.linalg.norm(means - ref[None, :], axis=1)
    scale = 60.0  # empirical normalization: ~60 RGB-units treated as a strong outlier
    return float(np.clip(dist.mean() / scale, 0.0, 1.0))


def compute_wsi_qc(wsi_path: str, cfg: dict, reader) -> dict:
    """`reader` is a duck-typed slide-reader object exposing
    `get_thumbnail(max_dim)`, `read_region_20x(x, y, size)`, `mpp`,
    `dimensions`, `tissue_bbox()`. Kept as a parameter (rather than importing
    openslide here) so this module stays importable without openslide for
    unit tests. See sts/wsi_patches.OpenSlideReader for the real
    implementation used by scripts/01_select_cptac_wsi.py."""
    thumb = reader.get_thumbnail(cfg["wsi_selection"]["thumbnail_max_dim"])
    tissue_mask = tissue_mask_from_thumbnail(thumb)
    tissue_fraction = float(tissue_mask.mean())

    mpp = reader.mpp
    width, height = reader.dimensions
    px_area_mm2 = (mpp or 0.25) ** 2 * 1e-6  # um^2 -> mm^2 per level-0 pixel
    tissue_area_mm2 = tissue_fraction * width * height * px_area_mm2

    tile_px = cfg["wsi_selection"]["qc_tile_px"]
    n_tiles = cfg["wsi_selection"]["n_qc_tiles"]
    coords = reader.sample_tissue_tile_coords(tissue_mask, n_tiles, tile_px)
    tiles = [reader.read_region_20x(x, y, tile_px) for (x, y) in coords]

    if tiles:
        focus_vals = np.array([focus_score_laplacian(t.mean(axis=-1)) for t in tiles])
        artifact_vals = np.array([artifact_fraction_tile(t) for t in tiles])
        focus_score = float(np.median(focus_vals))
        blurry_fraction = float((focus_vals < np.percentile(focus_vals, 20)).mean()) if len(focus_vals) > 4 else 0.0
        artifact_fraction = float(artifact_vals.mean())
        stain_score = stain_outlier_score(tiles)
    else:
        focus_score = np.nan
        blurry_fraction = np.nan
        artifact_fraction = np.nan
        stain_score = np.nan

    return {
        "tissue_area_mm2": tissue_area_mm2,
        "tissue_fraction": tissue_fraction,
        "focus_score": focus_score,
        "blurry_tissue_fraction": blurry_fraction,
        "artifact_fraction": artifact_fraction,
        "stain_score": stain_score,
        "mpp": mpp,
        "width": width,
        "height": height,
        "n_qc_tiles_sampled": len(tiles),
    }


def apply_hard_filters(rec: dict, cfg: dict) -> tuple[bool, str | None]:
    """Section 4 hard exclusions 1-7. Returns (excluded, reason)."""
    hf = cfg["wsi_selection"]["hard_filters"]
    if rec.get("is_he") is False:
        return True, "not_HE"
    if rec.get("is_human") is False:
        return True, "not_human"
    if rec.get("is_tumor_specimen") is False:
        return True, "not_tumor_specimen"
    if rec.get("readable") is False:
        return True, "unreadable_or_corrupt"
    mpp = rec.get("mpp")
    if mpp is None:
        return True, "mpp_unknown_and_uninferable"
    if not (hf["min_mpp"] <= mpp <= hf["max_mpp"]):
        return True, f"mpp_out_of_range({mpp:.3f})"
    if rec.get("tissue_fraction", 0.0) < hf["min_tissue_fraction"]:
        return True, "insufficient_tissue"
    if rec.get("blurry_tissue_fraction", 0.0) > hf["max_blurry_tissue_fraction"]:
        return True, "excessive_blur"
    if rec.get("artifact_fraction", 0.0) > hf["max_artifact_fraction"]:
        return True, "excessive_artifact"
    return False, None


def compute_q_score(rec: dict, cfg: dict, cohort_stats: dict) -> float:
    """Deterministic ranking, Section 4:
    Q = 0.35*usable_tissue + 0.25*focus + 0.20*artifact_free
        + 0.10*stain_quality + 0.10*representativeness
    Each component is min-max normalized against the (post-hard-filter)
    cohort so Q is comparable across WSIs of very different raw scales.
    `representativeness_score` = 1 - |tissue_fraction - cohort_median|,
    i.e. WSIs with a typical tissue fraction for this cancer type score
    higher than extreme outliers on either side."""
    w = cfg["wsi_selection"]["quality_weights"]

    def norm(key, value, higher_is_better=True):
        lo, hi = cohort_stats[key]
        if hi <= lo:
            return 0.5
        z = (value - lo) / (hi - lo)
        z = float(np.clip(z, 0.0, 1.0))
        return z if higher_is_better else 1.0 - z

    usable_tissue = norm("tissue_area_mm2", rec["tissue_area_mm2"])
    focus = norm("focus_score", rec["focus_score"])
    artifact_free = norm("artifact_fraction", rec["artifact_fraction"], higher_is_better=False)
    stain_quality = norm("stain_score", rec["stain_score"], higher_is_better=False)

    median_tf = cohort_stats["tissue_fraction_median"]
    representativeness = float(np.clip(1.0 - abs(rec["tissue_fraction"] - median_tf), 0.0, 1.0))

    q = (w["usable_tissue"] * usable_tissue + w["focus"] * focus + w["artifact_free"] * artifact_free
         + w["stain_quality"] * stain_quality + w["representativeness"] * representativeness)
    return float(q)


def compute_cohort_stats(records: list[dict]) -> dict:
    def rng(key):
        vals = np.array([r[key] for r in records if np.isfinite(r.get(key, np.nan))])
        return (float(vals.min()), float(vals.max())) if len(vals) else (0.0, 1.0)

    tf = np.array([r["tissue_fraction"] for r in records if np.isfinite(r.get("tissue_fraction", np.nan))])
    return {
        "tissue_area_mm2": rng("tissue_area_mm2"),
        "focus_score": rng("focus_score"),
        "artifact_fraction": rng("artifact_fraction"),
        "stain_score": rng("stain_score"),
        "tissue_fraction_median": float(np.median(tf)) if len(tf) else 0.5,
    }


def select_one_wsi_per_patient(all_records: list[dict], cfg: dict) -> tuple[list[dict], dict]:
    """Applies hard filters, computes cohort-relative Q per cancer type,
    then selects the max-Q WSI per patient. Returns (selected_records,
    report) where report has the Section-4-required exclusion/count
    summary."""
    for r in all_records:
        excluded, reason = apply_hard_filters(r, cfg)
        r["excluded"], r["exclusion_reason"] = excluded, reason

    kept = [r for r in all_records if not r["excluded"]]
    by_cancer: dict[str, list[dict]] = {}
    for r in kept:
        by_cancer.setdefault(r["cancer_type"], []).append(r)

    for cancer, recs in by_cancer.items():
        stats = compute_cohort_stats(recs)
        for r in recs:
            r["qc_score"] = compute_q_score(r, cfg, stats)

    selected = []
    by_patient: dict[str, list[dict]] = {}
    for r in kept:
        by_patient.setdefault(r["patient_id"], []).append(r)
    for pid, cands in by_patient.items():
        cands_sorted = sorted(cands, key=lambda r: r["qc_score"], reverse=True)
        best = cands_sorted[0]
        best["selection_reason"] = (
            "highest_qc_score" if len(cands_sorted) > 1 else "only_candidate"
        )
        selected.append(best)

    n_before_cap = len(selected)
    target_n = cfg["wsi_selection"].get("target_n_patients")
    if target_n and n_before_cap > target_n:
        # Section 2: "Target approximately ~500 patient-unique WSIs total."
        # When more patients pass QC than the target, keep the highest-Q
        # patients, capped proportionally per cancer so no single cancer
        # type is squeezed out by another's higher average Q.
        by_cancer_selected: dict[str, list[dict]] = {}
        for r in selected:
            by_cancer_selected.setdefault(r["cancer_type"], []).append(r)
        capped = []
        for cancer, recs in by_cancer_selected.items():
            cancer_target = max(1, round(target_n * len(recs) / n_before_cap))
            recs_sorted = sorted(recs, key=lambda r: r["qc_score"], reverse=True)
            for r in recs_sorted[:cancer_target]:
                r["selection_reason"] += "+top_qc_capped_to_target_n"
            capped.extend(recs_sorted[:cancer_target])
        selected = capped

    exclusion_counts: dict[str, int] = {}
    for r in all_records:
        if r["excluded"]:
            exclusion_counts[r["exclusion_reason"]] = exclusion_counts.get(r["exclusion_reason"], 0) + 1

    report = {
        "n_initial_wsis": len(all_records),
        "n_initial_patients": len(set(r["patient_id"] for r in all_records)),
        "n_excluded_wsis": len(all_records) - len(kept),
        "exclusion_reasons": exclusion_counts,
        "n_patients_passed_qc_before_target_cap": n_before_cap,
        "n_final_patients": len(selected),
        "n_final_wsis": len(selected),
        "by_cancer": {
            c: {
                "n_wsis_considered": len(recs),
                "n_patients_selected": len(set(r["patient_id"] for r in recs)),
            }
            for c, recs in by_cancer.items()
        },
    }
    return selected, report


def to_manifest_rows(selected: list[dict]) -> list[dict]:
    """Maps a selected-record dict to exactly the manifests/cptac_selected_wsi.csv
    schema in Section 4."""
    cols = ["patient_id", "cancer_type", "wsi_id", "wsi_path", "specimen_type", "mpp",
            "width", "height", "tissue_area_mm2", "tissue_fraction", "focus_score",
            "artifact_fraction", "stain_score", "qc_score", "selection_reason"]
    return [{c: r.get(c) for c in cols} for r in selected]
