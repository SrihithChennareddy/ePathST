"""Step 22 (Section 26) additional scientific analyses: multiscale
contribution, spatial-autocorrelation-vs-graph-improvement correlation,
per-pathway analysis, per-cancer analysis. Nothing here assumes an outcome
ahead of time (Section 26.C/D: 'Do not assume ahead of time which pathways
win' / 'Determine whether spatial benefit is consistent or cancer-dependent')."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sts.metrics import morans_i, per_patient_per_column_pearson


def multiscale_contribution(r_local_by_gene: pd.Series, r_multiscale_by_gene: pd.Series) -> pd.DataFrame:
    """Section 26.A: per-gene delta_g = r_multiscale,g - r_local,g."""
    df = pd.DataFrame({"r_local": r_local_by_gene, "r_multiscale": r_multiscale_by_gene}).dropna()
    df["delta"] = df["r_multiscale"] - df["r_local"]
    return df


def ground_truth_morans_i_per_gene(expr_true: np.ndarray, coords_um: np.ndarray, patient_ids: np.ndarray,
                                    col_names: list[str], k: int = 6) -> pd.DataFrame:
    """Section 26.B: Moran's I of ground-truth expression, computed within
    each patient's own spatial coordinates then averaged across patients
    per gene (Moran's I is not comparable across differently-scaled/sized
    tissue sections, so per-patient computation first is required)."""
    rows = []
    for pid in sorted(set(patient_ids)):
        mask = patient_ids == pid
        if mask.sum() < k + 1:
            continue
        coords = coords_um[mask]
        for j, col in enumerate(col_names):
            mi = morans_i(expr_true[mask, j], coords, k=k)
            rows.append({"patient_id": pid, "column": col, "morans_i": mi})
    df = pd.DataFrame(rows)
    return df.groupby("column")["morans_i"].mean().reset_index()


def spatial_benefit_vs_autocorrelation(delta_by_gene: pd.DataFrame, morans_i_by_gene: pd.DataFrame) -> dict:
    """Section 26.B: Corr(delta_g, MoranI_g). Hypothesis: genes with
    stronger spatial organization benefit more from explicit graph
    modeling. Reported, not assumed."""
    merged = delta_by_gene.merge(morans_i_by_gene, left_index=True, right_on="column") \
        if delta_by_gene.index.name == "column" else delta_by_gene.reset_index().merge(morans_i_by_gene, on="column" if "column" in delta_by_gene.columns else None)
    merged = merged.dropna(subset=["delta", "morans_i"])
    if len(merged) < 5:
        return {"n_genes": len(merged), "pearson_r": np.nan, "p_value": np.nan}
    r, p = pearsonr(merged["delta"], merged["morans_i"])
    return {"n_genes": len(merged), "pearson_r": float(r), "p_value": float(p)}


def per_pathway_delta(r_patch_by_pathway: pd.Series, r_graph_by_pathway: pd.Series) -> pd.DataFrame:
    """Section 26.C: which pathways benefit most from graph context (no
    prior assumption of winners)."""
    df = pd.DataFrame({"r_patch": r_patch_by_pathway, "r_graph": r_graph_by_pathway}).dropna()
    df["delta"] = df["r_graph"] - df["r_patch"]
    return df.sort_values("delta", ascending=False)


def per_cancer_consistency(results_by_cancer: dict[str, dict]) -> pd.DataFrame:
    """Section 26.D: tabulate the primary metric per cancer to check
    whether spatial benefit is consistent or cancer-dependent."""
    rows = []
    for cancer, res in results_by_cancer.items():
        rows.append({"cancer": cancer, **res})
    return pd.DataFrame(rows)
