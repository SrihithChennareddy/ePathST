"""Evaluation-time-only analysis: scores predicted gene expression with the same PROGENy pipeline used
for ground truth, then compares those post-hoc scores to ground truth and to the model's direct pathway head."""
from __future__ import annotations

import numpy as np
import pandas as pd

from epathst.data.pathways import compute_progeny_scores


def posthoc_pathway_scores(
    predicted_genes: np.ndarray,
    gene_names: list[str],
    progeny_net: pd.DataFrame,
    spot_ids: list[str],
    min_n: int = 5,
) -> pd.DataFrame:
    """`gene_names` must be the exact locked HVG1000 order, never re-derived/reordered; returns (n_spots,n_pathways) like compute_progeny_scores's ground-truth output -- gotcha: PROGENy expects log-normalized expression, so these scores live on a different scale than ground truth, compare ranks/correlations not raw magnitudes."""
    expr_df = pd.DataFrame(predicted_genes, index=spot_ids, columns=gene_names)
    return compute_progeny_scores(expr_df, progeny_net, min_n=min_n)


def compare_posthoc_vs_direct(
    posthoc_scores: pd.DataFrame,
    direct_scores: np.ndarray,
    pathway_names: list[str],
    ground_truth_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Per-pathway Pearson correlation of post-hoc-from-predicted-genes vs. ground truth and direct-pathway-head vs. ground truth, to see whether deriving pathway activity from gene predictions beats a dedicated pathway head."""
    rows = []
    direct_df = pd.DataFrame(direct_scores, index=posthoc_scores.index, columns=pathway_names)
    for p in pathway_names:
        if p not in ground_truth_scores.columns:
            continue
        gt = ground_truth_scores[p].values
        ph = posthoc_scores[p].values if p in posthoc_scores.columns else np.full_like(gt, np.nan)
        di = direct_df[p].values

        def _corr(a, b):
            valid = ~(np.isnan(a) | np.isnan(b))
            if valid.sum() < 3 or np.nanstd(a[valid]) < 1e-12 or np.nanstd(b[valid]) < 1e-12:
                return np.nan
            return float(np.corrcoef(a[valid], b[valid])[0, 1])

        rows.append({
            "pathway": p,
            "r_posthoc_vs_gt": _corr(ph, gt),
            "r_direct_head_vs_gt": _corr(di, gt),
        })
    return pd.DataFrame(rows)
