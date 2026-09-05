"""Pathway activity computed POST HOC from PREDICTED gene expression, as an
alternative/complement to the direct pathway head. This is scientifically
cleaner than a direct head: it tests biological consistency (do predicted
genes imply the right pathway activity?) rather than giving the model the
answer through an auxiliary objective.

Purely an evaluation-time analysis (no gradients flow through this module):
take a model's HVG1000 gene predictions, run the same PROGENy scoring used
for ground truth, and compare the resulting pathway score to (a) the
ground-truth PROGENy score and (b) the model's own direct pathway head
output (if it has one)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathology_st.pathways import compute_progeny_scores


def posthoc_pathway_scores(predicted_genes: np.ndarray, gene_names: list[str], progeny_net: pd.DataFrame,
                            spot_ids: list[str], min_n: int = 5) -> pd.DataFrame:
    """`predicted_genes`: (n_spots, n_genes) standardized-scale model
    output. `gene_names` must be the exact HVG1000 order used to produce
    it -- never re-derive or reorder, it must match the fold's locked gene
    order. Returns (n_spots, n_pathways) post-hoc PROGENy scores, same
    shape/convention as `pathology_st.pathways.compute_progeny_scores`'s
    ground-truth output.

    Gotcha: PROGENy's MLM scoring expects log-normalized expression-scale
    values, not standardized (z-scored) targets, so post-hoc scores from
    standardized predictions live on a different scale than the
    ground-truth PROGENy scores. Compare ranks/correlations, not raw
    magnitudes, unless both sides are rescaled first."""
    expr_df = pd.DataFrame(predicted_genes, index=spot_ids, columns=gene_names)
    return compute_progeny_scores(expr_df, progeny_net, min_n=min_n)


def compare_posthoc_vs_direct(posthoc_scores: pd.DataFrame, direct_scores: np.ndarray, pathway_names: list[str],
                               ground_truth_scores: pd.DataFrame) -> pd.DataFrame:
    """Per-pathway Pearson correlation of (a) post-hoc-from-predicted-genes
    vs. ground truth, and (b) direct-pathway-head vs. ground truth --
    lets you see whether deriving pathway activity from the gene
    predictions is as good as, better than, or worse than a dedicated
    pathway head, per pathway."""
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

        rows.append({"pathway": p, "r_posthoc_vs_gt": _corr(ph, gt), "r_direct_head_vs_gt": _corr(di, gt)})
    return pd.DataFrame(rows)
