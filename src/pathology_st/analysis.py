"""Additional scientific analyses: multiscale contribution, spatial-autocorrelation-vs-graph-improvement,
per-pathway, per-cancer -- nothing here assumes an outcome ahead of time; results are reported as found."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from pathology_st.metrics import morans_i


def multiscale_contribution(
    r_local_by_gene: pd.Series, r_multiscale_by_gene: pd.Series
) -> pd.DataFrame:
    """Per-gene delta_g = r_multiscale,g - r_local,g."""
    df = pd.DataFrame({"r_local": r_local_by_gene, "r_multiscale": r_multiscale_by_gene}).dropna()
    df["delta"] = df["r_multiscale"] - df["r_local"]
    return df


def ground_truth_morans_i_per_gene(
    expr_true: np.ndarray,
    coords_um: np.ndarray,
    patient_ids: np.ndarray,
    col_names: list[str],
    k: int = 6,
) -> pd.DataFrame:
    """Moran's I of ground-truth expression per patient's own coordinates, then averaged across patients per gene -- required since Moran's I isn't comparable across differently-scaled/sized tissue sections."""
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


def spatial_benefit_vs_autocorrelation(
    delta_by_gene: pd.DataFrame, morans_i_by_gene: pd.DataFrame
) -> dict:
    """Corr(delta_g, MoranI_g); hypothesis that genes with stronger spatial organization benefit more from explicit graph modeling is reported, not assumed."""
    if delta_by_gene.index.name == "column":
        merged = delta_by_gene.merge(morans_i_by_gene, left_index=True, right_on="column")
    else:
        on_col = "column" if "column" in delta_by_gene.columns else None
        merged = delta_by_gene.reset_index().merge(morans_i_by_gene, on=on_col)
    merged = merged.dropna(subset=["delta", "morans_i"])
    if len(merged) < 5:
        return {"n_genes": len(merged), "pearson_r": np.nan, "p_value": np.nan}
    r, p = pearsonr(merged["delta"], merged["morans_i"])
    return {"n_genes": len(merged), "pearson_r": float(r), "p_value": float(p)}


def per_pathway_delta(r_patch_by_pathway: pd.Series, r_graph_by_pathway: pd.Series) -> pd.DataFrame:
    """Which pathways benefit most from graph context -- no prior assumption of winners."""
    df = pd.DataFrame({"r_patch": r_patch_by_pathway, "r_graph": r_graph_by_pathway}).dropna()
    df["delta"] = df["r_graph"] - df["r_patch"]
    return df.sort_values("delta", ascending=False)


def per_cancer_consistency(results_by_cancer: dict[str, dict]) -> pd.DataFrame:
    """Tabulates the primary metric per cancer to check whether spatial benefit is consistent or cancer-dependent."""
    rows = []
    for cancer, res in results_by_cancer.items():
        rows.append({"cancer": cancer, **res})
    return pd.DataFrame(rows)
