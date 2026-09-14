"""Evaluation metrics: patient-level Fisher-z aggregated Pearson (the primary metric -- patients, not
spots, are the independent statistical unit), plus Spearman, MSE/MAE, threshold fractions, Moran's I."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 3 or np.nanstd(y_true[valid]) < 1e-12 or np.nanstd(y_pred[valid]) < 1e-12:
        return np.nan
    return float(np.corrcoef(y_true[valid], y_pred[valid])[0, 1])


def per_patient_per_column_pearson(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
    col_names: list[str],
    min_regions: int,
) -> pd.DataFrame:
    rows = []
    for pid in sorted(set(patient_ids)):
        mask = patient_ids == pid
        if mask.sum() < min_regions:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        for j, col in enumerate(col_names):
            r = _safe_pearson(yt[:, j], yp[:, j])
            mse = float(np.nanmean((yt[:, j] - yp[:, j]) ** 2))
            mae = float(np.nanmean(np.abs(yt[:, j] - yp[:, j])))
            rows.append({
                "patient_id": pid,
                "column": col,
                "r": r,
                "mse": mse,
                "mae": mae,
                "n": int(mask.sum()),
            })
    return pd.DataFrame(rows)


def patient_macro_fisher_z(per_patient_per_col: pd.DataFrame) -> pd.DataFrame:
    """R_p = tanh(mean_g(atanh(r_pg))) -- the fundamental independent observations for every downstream statistical test."""
    df = per_patient_per_col.dropna(subset=["r"]).copy()
    df["r_clipped"] = df["r"].clip(-0.999999, 0.999999)
    df["z"] = np.arctanh(df["r_clipped"])
    agg = df.groupby("patient_id")["z"].mean().reset_index()
    agg["R_patient"] = np.tanh(agg["z"])
    return agg[["patient_id", "z", "R_patient"]]


def target_summary(per_patient_per_col: pd.DataFrame, thresholds=(0.0, 0.2, 0.3)) -> dict:
    """Fisher-z mean + median patient-level correlation, plus %{gene: r > threshold} on the per-gene pooled-across-patients mean r."""
    per_col = per_patient_per_col.dropna(subset=["r"]).groupby("column")["r"].mean()
    macro = patient_macro_fisher_z(per_patient_per_col)
    out = {
        "fisher_z_mean_patient_R": float(macro["R_patient"].mean()) if len(macro) else np.nan,
        "median_patient_R": float(macro["R_patient"].median()) if len(macro) else np.nan,
        "n_columns": int(per_col.shape[0]),
        "n_patients": int(macro.shape[0]),
    }
    for t in thresholds:
        out[f"pct_columns_r_gt_{t}"] = float((per_col > t).mean()) if len(per_col) else np.nan
    return out


def spearman_per_column(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
    col_names: list[str],
    min_regions: int,
) -> pd.DataFrame:
    rows = []
    for pid in sorted(set(patient_ids)):
        mask = patient_ids == pid
        if mask.sum() < min_regions:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        for j, col in enumerate(col_names):
            valid = ~(np.isnan(yt[:, j]) | np.isnan(yp[:, j]))
            rho = np.nan
            if valid.sum() >= 3 and np.nanstd(yt[valid, j]) > 1e-12:
                rho = spearmanr(yt[valid, j], yp[valid, j]).correlation
            rows.append({"patient_id": pid, "column": col, "spearman": rho})
    return pd.DataFrame(rows)


def morans_i(values: np.ndarray, coords_um: np.ndarray, k: int = 6) -> float:
    """Moran's I of a single spatial field via a k-NN binary weight matrix."""
    from scipy.spatial import cKDTree

    n = len(values)
    valid = ~np.isnan(values)
    if valid.sum() < k + 1:
        return np.nan
    v = values.copy()
    v[~valid] = np.nanmean(values)

    # k-NN neighbor graph
    tree = cKDTree(coords_um)
    _, idx = tree.query(coords_um, k=k + 1)
    neighbors = idx[:, 1:]

    # accumulate weighted cross-products
    x = v - v.mean()
    num, w_sum = 0.0, 0.0
    for i in range(n):
        for j in neighbors[i]:
            num += x[i] * x[j]
            w_sum += 1.0
    denom = (x ** 2).sum()
    if denom < 1e-12 or w_sum < 1e-12:
        return np.nan
    return float((n / w_sum) * (num / denom))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
    cancer_types: np.ndarray,
    col_names: list[str],
    cfg: dict,
) -> dict:
    """Overall + per-cancer patient-level Pearson/Spearman/MSE/MAE summary for one target block (HVG50, HVG1000, or pathways)."""
    min_regions = cfg["evaluation"]["min_regions_per_patient"]
    thresholds = tuple(cfg["evaluation"]["gene_r_thresholds"])
    out = {}
    # overall + per-cancer subgroups
    groups = {"overall": np.ones(len(patient_ids), dtype=bool)}
    for c in np.unique(cancer_types):
        groups[c] = cancer_types == c

    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        pp = per_patient_per_column_pearson(
            y_true[mask], y_pred[mask], patient_ids[mask], col_names, min_regions
        )
        sp = spearman_per_column(
            y_true[mask], y_pred[mask], patient_ids[mask], col_names, min_regions
        )
        summ = target_summary(pp, thresholds)
        summ["mean_spearman"] = float(sp["spearman"].mean()) if len(sp) else np.nan
        summ["mean_mse"] = float(pp["mse"].mean()) if len(pp) else np.nan
        summ["mean_mae"] = float(pp["mae"].mean()) if len(pp) else np.nan
        out[name] = summ
    return out
