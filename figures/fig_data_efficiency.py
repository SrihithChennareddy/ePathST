"""Draws Fig. 3: HEST average PCC vs pretraining slides (a) and tiles (b).
Reads figures/data/data_efficiency_points.csv; writes fig3_data_efficiency.{png,pdf}."""

from __future__ import annotations

import ast
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

from _paths import DATA_DIR, FIG_OUT

OUT = FIG_OUT / "fig3_pretraining_data"
OUT.mkdir(parents=True, exist_ok=True)
for f in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
INK, MUTED, GRID = "#1a1a1a", "#555555", "#e3e3e0"
# Data access is encoded by both colour and marker shape.
STYLE = {
    "Private": ("#eb6834", "s"),
    "Public": ("#1baf7a", "o"),
    "Ours": ("#2a78d6", "D"),
}


def _value(text):
    return ast.literal_eval(text) if text else None


with open(DATA_DIR / "data_efficiency_points.csv", newline="") as f:
    POINTS = list(csv.DictReader(f))
# name, avg PCC, slides, tiles, access, label offset slides (pt), label offset tiles (pt), ha
M = [
    (
        r["name"],
        _value(r["pcc"]),
        _value(r["slides"]),
        _value(r["tiles"]),
        r["access"],
        (_value(r["slide_dx"]), _value(r["slide_dy"]), r["slide_ha"]),
        (_value(r["tile_dx"]), _value(r["tile_dy"]), r["tile_ha"]) if r["tile_ha"] else None,
    )
    for r in POINTS
]
# Rows with in_trend False (ours and one public model) are left out of the trend fit.
IN_TREND = [r["in_trend"] == "True" for r in POINTS]
RESNET = 0.3164  # HEST average PCC of ResNet50 (ImageNet, no slides), drawn as a reference line
LOWER_BOUND = {"H-Optimus-0", "UNI2-h", "CONCH v1.5"}  # authors give '>' counts; see tables

fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.35), sharey=True)
for ax in axes:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#8a8a86")
    ax.tick_params(labelsize=6.8, colors=INK, length=3)
    ax.axhline(RESNET, color="#8a8a86", lw=0.9, ls=(0, (1.5, 2.5)), zorder=1)

# Panel (a): PCC vs pretraining slides, with a log-linear least-squares trend (dashed).
ax = axes[0]
base = [m for m, in_trend in zip(M, IN_TREND) if in_trend]
x = np.log10([m[2] for m in base])
y = np.array([m[1] for m in base])
slope, icpt = np.polyfit(x, y, 1)
r2 = np.corrcoef(x, y)[0, 1] ** 2
xx = np.linspace(np.log10(2e4), np.log10(4e6), 50)
ax.plot(10**xx, slope * xx + icpt, color="#6b6b66", lw=1.1, ls="--", zorder=2)
for name, pcc, sl, _, acc, off, _o in M:
    c, mk = STYLE[acc]
    ax.scatter(sl, pcc, s=34, color=c, marker=mk, edgecolor="#fcfcfb", linewidth=0.9, zorder=4)
    ax.annotate(
        name,
        (sl, pcc),
        xytext=(off[0], off[1]),
        textcoords="offset points",
        fontsize=6.3,
        color=INK,
        ha=off[2],
        va="center",
        zorder=6,
        bbox=dict(boxstyle="square,pad=0.15", fc="#fcfcfb", ec="none", alpha=0.85),
    )
# Dotted connector from ePathST-E to its teacher H-Optimus-0, whose slides produced the
# distillation targets.
ho = next(m for m in M if m[0] == "H-Optimus-0")
pe = next(m for m in M if m[0] == "ePathST-E (ours)")
ax.annotate(
    "",
    xy=(ho[2], ho[1]),
    xytext=(pe[2], pe[1]),
    arrowprops=dict(
        arrowstyle="->",
        color=STYLE["Ours"][0],
        lw=0.9,
        ls=(0, (2, 2)),
        shrinkA=5,
        shrinkB=5,
        alpha=0.8,
    ),
    zorder=3,
)
ax.text(
    8e3,
    0.3745,
    "distilled from H-Optimus-0\n(teacher slides)",
    fontsize=5.8,
    color=STYLE["Ours"][0],
    ha="center",
    va="bottom",
    rotation=0,
    zorder=5,
)
ax.text(
    2.1e2,
    RESNET + 0.0012,
    "ResNet50 (ImageNet, no slides)",
    fontsize=5.8,
    color=MUTED,
    ha="left",
    va="bottom",
)
ax.text(
    4.0e6,
    slope * np.log10(4e6) + icpt + 0.003,
    f"trend of earlier models\n(n = {len(base)}, R$^2$ = {r2:.2f})",
    fontsize=5.8,
    color="#6b6b66",
    ha="right",
    va="bottom",
)
ax.set_xscale("log")
ax.set_xlim(180, 1.2e7)
ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7])
ax.set_xticklabels(["1k", "10k", "100k", "1M", "10M"])
ax.set_xlabel("Pretraining slides (WSIs)", fontsize=7.4, color=INK)
ax.set_ylabel("HEST average PCC", fontsize=7.4, color=INK)
ax.set_ylim(0.305, 0.402)
ax.text(
    0.0,
    1.04,
    "(a)",
    transform=ax.transAxes,
    fontsize=9,
    fontweight="bold",
    color=INK,
    ha="left",
    va="bottom",
)

# Panel (b): PCC vs pretraining tiles; models without a reported tile count are skipped.
ax = axes[1]
for name, pcc, _, tl, acc, _o, off in M:
    if tl is None:
        continue
    c, mk = STYLE[acc]
    ax.scatter(tl, pcc, s=34, color=c, marker=mk, edgecolor="#fcfcfb", linewidth=0.9, zorder=4)
    ax.annotate(
        name,
        (tl, pcc),
        xytext=(off[0], off[1]),
        textcoords="offset points",
        fontsize=6.3,
        color=INK,
        ha=off[2],
        va="center",
        zorder=6,
        bbox=dict(boxstyle="square,pad=0.15", fc="#fcfcfb", ec="none", alpha=0.85),
    )
ax.text(
    1.05e6,
    RESNET + 0.0012,
    "ResNet50 (1.28 M ImageNet images)",
    fontsize=5.8,
    color=MUTED,
    ha="left",
    va="bottom",
)
ax.set_xscale("log")
ax.set_xlim(2.5e5, 6e9)
ax.set_xticks([1e6, 1e7, 1e8, 1e9])
ax.set_xticklabels(["1M", "10M", "100M", "1B"])
ax.set_xlabel("Pretraining tiles / patches", fontsize=7.4, color=INK)
ax.text(
    0.0,
    1.04,
    "(b)",
    transform=ax.transAxes,
    fontsize=9,
    fontweight="bold",
    color=INK,
    ha="left",
    va="bottom",
)
ax.text(
    1.0,
    0.012,
    "not shown: H-Optimus-0, Virchow2 (tile count not reported),\nKaiko (online sampling, no fixed set)",
    transform=ax.transAxes,
    fontsize=5.5,
    color=MUTED,
    ha="right",
    va="bottom",
)

handles = [
    Line2D(
        [0],
        [0],
        marker=STYLE[k][1],
        color="none",
        markerfacecolor=STYLE[k][0],
        markeredgecolor="#fcfcfb",
        markersize=6.5,
        label=lbl,
    )
    for k, lbl in (
        ("Private", "Private data"),
        ("Public", "Public data"),
        ("Ours", "Ours (public CPTAC)"),
    )
]
axes[0].legend(
    handles=handles,
    loc="lower right",
    fontsize=6.2,
    frameon=True,
    framealpha=0.95,
    edgecolor=GRID,
    borderpad=0.5,
    handletextpad=0.3,
    labelspacing=0.35,
)
fig.subplots_adjust(left=0.075, right=0.995, top=0.93, bottom=0.135, wspace=0.05)
fig.savefig(OUT / "fig3_data_efficiency.png", dpi=300)
fig.savefig(OUT / "fig3_data_efficiency.pdf")

# Ratios quoted in the paper text.
v = {m[0]: m for m in M}
pe_full, pe_e, hop, giga = (
    v["ePathST (ours)"],
    v["ePathST-E (ours)"],
    v["H-Optimus-0"],
    v["GigaPath"],
)
# the one public model left out of the trend fit, compared against it below
gb = next(m for m, in_trend in zip(M, IN_TREND) if not in_trend and m[4] != "Ours")
print(
    f"trend (all FMs except {gb[0]} and ours, n={len(base)}): PCC = {slope:.4f} * log10(slides) + {icpt:.4f}; "
    f"+{slope:.4f} PCC per 10x slides; R^2 = {r2:.2f}"
)
print(
    f"{gb[0]}: {gb[1]:.4f} vs trend {slope * np.log10(gb[2]) + icpt:.4f} at {gb[2]:,} slides "
    f"(+{gb[1] - (slope * np.log10(gb[2]) + icpt):.4f})"
)
print(
    f"ePathST vs H-Optimus-0: {pe_full[1] / hop[1] * 100:.1f}% of PCC ({pe_full[1]:.4f}/{hop[1]:.4f}); trained on {pe_full[2] / hop[2] * 100:.3f}% as many slides "
    f"({pe_full[2]} vs >{hop[2]:,})"
)
print(
    f"ePathST-E vs H-Optimus-0: {pe_e[1] / hop[1] * 100:.1f}% of PCC; ePathST vs GigaPath: {pe_full[1]:.4f} vs {giga[1]:.4f} with "
    f"{pe_full[2] / giga[2] * 100:.2f}% as many slides; ePathST vs {gb[0]}: {pe_full[1] / gb[1] * 100:.1f}% of PCC, "
    f"{pe_full[2] / gb[2] * 100:.2f}% of slides, {pe_full[3] / gb[3] * 100:.2f}% of tiles"
)
print(
    f"ePathST-E ({pe_e[1]:.4f}) vs CTransPath ({v['CTransPath'][1]:.4f}) with {v['CTransPath'][2] / pe_e[2]:.0f}x fewer slides; vs Kaiko "
    f"({v['Kaiko ViT-B/16'][1]:.4f}) with {v['Kaiko ViT-B/16'][2] / pe_e[2]:.0f}x fewer slides"
)
