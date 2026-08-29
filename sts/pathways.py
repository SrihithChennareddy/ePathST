"""PROGENy14 pathway scoring (Final_Project.txt Section 14 target set) via
decoupler's MLM method. Adapted from STS_PoC/PoC_5/crcspatial/pathways.py.

decoupler's API was restructured between v1 and v2 -- confirmed live
against the installed v2.2.0 (2026-08-29, during the V1->V2 real-data
validation run): `dc.get_progeny(...)` -> `dc.op.progeny(...)` (same
kwargs/return schema: source/target/weight/padj columns) and
`dc.run_mlm(mat=..., net=..., source=..., target=..., weight=...,
min_n=...)` -> `dc.mt.mlm(data=..., net=..., tmin=...)` (net's
source/target/weight column NAMES are still the standard decoupler
schema and no longer passed as separate kwargs; `min_n` renamed `tmin`).
Return shape/type (a (acts_df, pvals_df) tuple, spots x pathways) is
unchanged."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def fetch_and_cache_progeny_net(cache_path: str | Path, top: int = 100, organism: str = "human") -> pd.DataFrame:
    """Fetches the PROGENy model (14 pathways x weighted target genes) once
    and caches it so re-runs reuse the exact same gene-weight table."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_csv(cache_path)
    import decoupler as dc

    net = dc.op.progeny(organism=organism, top=top)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    net.to_csv(cache_path, index=False)
    return net


def check_pathway_gene_coverage(net: pd.DataFrame, genes_by_slide: dict[str, set[str]]) -> pd.DataFrame:
    """Section 14: check pathway gene coverage for every ST platform/slide.
    Only keep pathways with acceptable target-gene coverage across all
    retained slides."""
    rows = []
    for pathway, sub in net.groupby("source"):
        targets = set(sub["target"])
        n_present_per_slide = [len(targets & genes) for genes in genes_by_slide.values()]
        rows.append({
            "pathway": pathway,
            "n_target_genes": len(targets),
            "min_present_across_slides": min(n_present_per_slide) if n_present_per_slide else 0,
            "mean_present_across_slides": sum(n_present_per_slide) / len(n_present_per_slide) if n_present_per_slide else 0,
        })
    return pd.DataFrame(rows).sort_values("pathway").reset_index(drop=True)


def compute_progeny_scores(expr_df: pd.DataFrame, net: pd.DataFrame, min_n: int = 5) -> pd.DataFrame:
    """Runs decoupler's MLM method on log-normalized full expression
    (not just HVG1000). Returns (spots x pathways), reindexed to exactly
    expr_df's row index/order; spots decoupler drops as degenerate are
    filled NaN (handled by the same missingness-aware QC as any other
    target)."""
    import decoupler as dc

    acts, pvals = dc.mt.mlm(data=expr_df, net=net, tmin=min_n, verbose=False)
    acts.index = acts.index.astype(expr_df.index.dtype)
    n_dropped = expr_df.shape[0] - acts.shape[0]
    if n_dropped:
        print(f"    [compute_progeny_scores] decoupler dropped {n_dropped}/{expr_df.shape[0]} spots as degenerate")
    return acts.reindex(expr_df.index)
