#!/usr/bin/env python
"""Draws the gene-level spatial-context figure: dot plots A-C per cohort and Moran's I scatter D.
Reads figures/data/gene_context/; writes gene_context_dotplot.png to the output dir."""
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, Normalize
import matplotlib.cm as cm

from _paths import DATA_DIR, FIG_OUT

IN_DIR = DATA_DIR / "gene_context"
OUT_PATH = FIG_OUT / "gene_context_dotplot.png"
CANCERS = ["IDC", "COAD", "CCRCC"]
TOP_N = 18  # genes shown per cohort (ranked by |r_ePathST - r_Patch|)

DELTA_COLS = [
    ("delta_L", "Local", "sig_L"),
    ("delta_M", "Meso", "sig_M"),
    ("delta_G", "Macro", "sig_G"),
    ("delta_multi", "Multi", "sig_multi"),
    ("delta_aware", "Aware", "sig_aware"),
]

tables_full = {c: pd.read_csv(IN_DIR / f"gene_context_{c}.csv") for c in CANCERS}
spearman = json.loads((IN_DIR / "gene_context_spearman.json").read_text())

# Keep the TOP_N genes by |gain| so strongly helped and strongly hurt genes both appear, then
# sort by signed gain for display. Fewer genes per cohort keeps the labels legible.
tables = {}
for c in CANCERS:
    t = tables_full[c].copy()
    t["abs_gain"] = t["sort_key_ept_minus_patch"].abs()
    t = t.sort_values("abs_gain", ascending=False).head(TOP_N)
    t = t.sort_values("sort_key_ept_minus_patch", ascending=False)
    tables[c] = t.reset_index(drop=True)

# Dot colour = deltaPCC, dot size = |deltaPCC|, black ring = 95% CI excludes 0. The colour and
# size scales use the 98th percentile of |deltaPCC| over all genes and cohorts, so a few extreme
# genes do not compress the rest.
all_deltas = np.concatenate(
    [tables_full[c][[d for d, _, _ in DELTA_COLS]].values.flatten() for c in CANCERS]
)
all_deltas = all_deltas[np.isfinite(all_deltas)]
vmax_delta = np.nanpercentile(np.abs(all_deltas), 98) if len(all_deltas) else 0.1
div_norm = TwoSlopeNorm(vmin=-vmax_delta, vcenter=0.0, vmax=vmax_delta)
div_cmap = plt.get_cmap("RdBu_r")

all_patch = np.concatenate([tables_full[c]["r_patch"].values for c in CANCERS])
all_patch = all_patch[np.isfinite(all_patch)]
seq_norm = Normalize(vmin=np.nanmin(all_patch), vmax=np.nanmax(all_patch))
seq_cmap = plt.get_cmap("Blues")

MAX_DOT = 260  # points^2, at vmax_delta magnitude


def size_for(vals):
    return MAX_DOT * np.clip(np.abs(vals) / vmax_delta, 0.08, 1.0)


fig = plt.figure(figsize=(16, 18), dpi=200)
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], hspace=0.22, wspace=0.30)
panel_axes = {
    "IDC": fig.add_subplot(gs[0, 0]),
    "COAD": fig.add_subplot(gs[0, 1]),
    "CCRCC": fig.add_subplot(gs[1, 0]),
}
panel_letter = {"IDC": "A", "COAD": "B", "CCRCC": "C"}

col_labels = ["Patch\nPCC"] + [lbl for _, lbl, _ in DELTA_COLS]
n_cols = len(col_labels)

for cancer, ax in panel_axes.items():
    t = tables[cancer]
    genes = t["gene"].tolist()
    n_genes = len(genes)
    y = np.arange(n_genes)[::-1]  # top gene at top of plot

    # Patch PCC column (sequential, no significance ring)
    x0 = np.zeros(n_genes)
    vals0 = t["r_patch"].values
    ax.scatter(
        x0,
        y,
        s=MAX_DOT * np.clip((vals0 - seq_norm.vmin) / (seq_norm.vmax - seq_norm.vmin), 0.1, 1.0),
        c=seq_cmap(seq_norm(vals0)),
        edgecolors="none",
        zorder=3,
    )

    for j, (col, lbl, sig_col) in enumerate(DELTA_COLS, start=1):
        vals = t[col].values
        sig = t[sig_col].values.astype(bool)
        colors = div_cmap(div_norm(vals))
        sizes = size_for(vals)
        edge_colors = np.where(sig, "black", "none")
        edge_widths = np.where(sig, 1.1, 0.0)
        ax.scatter(
            np.full(n_genes, j),
            y,
            s=sizes,
            c=colors,
            edgecolors=list(edge_colors),
            linewidths=list(edge_widths),
            zorder=3,
        )

    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_ylim(-0.8, n_genes - 0.2)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(genes, fontsize=8.5)
    ax.set_title(
        f"{panel_letter[cancer]}. {cancer}  (top {TOP_N} genes by |ePathST − Patch| gain)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax.grid(axis="y", color="0.92", linewidth=0.6, zorder=0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

# colorbars + size/significance legend
cbar_ax1 = fig.add_axes([0.06, 0.965, 0.13, 0.01])
fig.colorbar(
    cm.ScalarMappable(norm=seq_norm, cmap=seq_cmap),
    cax=cbar_ax1,
    orientation="horizontal",
    label="Patch PCC",
)
cbar_ax2 = fig.add_axes([0.24, 0.965, 0.16, 0.01])
fig.colorbar(
    cm.ScalarMappable(norm=div_norm, cmap=div_cmap),
    cax=cbar_ax2,
    orientation="horizontal",
    label="ΔPCC",
)
fig.text(
    0.44,
    0.972,
    "dot size = |ΔPCC|   ● black ring = 95% CI excludes 0",
    fontsize=9,
    va="center",
    ha="left",
)

# Panel D: ground-truth Moran's I vs deltaPCC at the matched scale, all genes of all cohorts.
panel_d_bbox = gs[1, 1].get_position(fig)
fig.text(
    panel_d_bbox.x0,
    panel_d_bbox.y1 + 0.012,
    "D. Ground-truth spatial organization vs. predictive context gain",
    fontsize=13,
    fontweight="bold",
    ha="left",
    va="bottom",
)
gs_d = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 1], wspace=0.4)
scale_specs = [
    ("local", "moran_I_local", "delta_L", "Local (0-150μm)"),
    ("meso", "moran_I_meso", "delta_M", "Meso (150-300μm)"),
    ("macro", "moran_I_macro", "delta_G", "Macro (300-600μm)"),
]
cohort_colors = {"IDC": "#1f6fb2", "COAD": "#c0392b", "CCRCC": "#2e8b57"}

for k, (scale, moran_col, delta_col, title) in enumerate(scale_specs):
    ax = fig.add_subplot(gs_d[0, k])
    for cancer in CANCERS:
        tf = tables_full[cancer]
        x, yv = tf[moran_col].values, tf[delta_col].values
        valid = np.isfinite(x) & np.isfinite(yv)
        ax.scatter(x[valid], yv[valid], s=14, alpha=0.6, color=cohort_colors[cancer], label=cancer)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel(f"Moran's I ({scale})", fontsize=8)
    if k == 0:
        ax.set_ylabel("ΔPCC (matched scale)", fontsize=8)
    rho_txt = "\n".join(f"{c}: ρ={spearman[c][scale]['spearman_rho']:.2f}" for c in CANCERS)
    ax.text(
        0.03,
        0.97,
        rho_txt,
        transform=ax.transAxes,
        fontsize=6.5,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.3),
    )
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    if k == 0:
        ax.legend(fontsize=6, loc="lower right", frameon=False)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=200, facecolor="white", bbox_inches="tight")
print(f"saved {OUT_PATH}")
