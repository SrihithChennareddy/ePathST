#!/usr/bin/env python
"""Draws HEST-1k average PCC vs TFLOPs per slide with an OLS trend and a Pareto frontier.
Reads figures/data/compute_vs_pcc_points.csv; writes a 3.5 in, 600 DPI PNG to the output dir."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from _paths import DATA_DIR, FIG_OUT

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["STIXGeneral", "cmr10", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

N_TILES = 11845  # TENX95 (IDC), largest gene-complete slide in the cohort; scales GFLOPs to slide

POINTS = pd.read_csv(DATA_DIR / "compute_vs_pcc_points.csv")
DATA = [
    (r.label, r.gflops_per_patch * N_TILES / 1000, r.pcc, bool(r.ours)) for r in POINTS.itertuples()
]

GRAY = "#4B4B4B"
GRAY_LIGHT = "#828282"
GRID_MAJOR = "#BEBEBE"
GRID_MINOR = "#DEDEDE"
RED = "#D52027"
INK = "#1a1a1a"
BLACK = "#000000"
TREND_COLOR = "#00549A"  # Stony Brook Royal Blue
EQN_DARK_COLOR = "#0A1F44"  # dark navy blue
PARETO_COLOR = "#6B8F4E"  # darkened Stony Brook Celery

label_offsets = {r.label: (r.dx, r.dy) for r in POINTS.itertuples()}
label_text = {r.label: r.text for r in POINTS.itertuples() if isinstance(r.text, str)}
label_ha = {r.label: r.ha for r in POINTS.itertuples() if isinstance(r.ha, str)}
label_va = {r.label: r.va for r in POINTS.itertuples() if isinstance(r.va, str)}

xticks = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
DARK_X = {20, 100, 500, 2000, 5000, 10000}
FINE_X = [v for n in (1, 3) for k in (3, 4, 6, 7, 8, 9) for v in [k * 10**n]]
FINE_X += [12000, 14000, 16000]

fig, ax = plt.subplots(figsize=(3.5, 3.05), dpi=600)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for label, x, y, ours in DATA:
    if ours:
        ax.scatter(x, y, s=85, marker="*", color=RED, edgecolor="white", zorder=5, linewidths=0.5)
    else:
        ax.scatter(x, y, s=26, marker="D", color=GRAY, edgecolor="white", zorder=4, linewidths=0.4)

for label, x, y, ours in DATA:
    dx, dy = label_offsets[label]
    ha = label_ha.get(label, "left")
    va = label_va.get(label, "center")
    text = label_text.get(label, label)
    color = RED if ours else INK
    weight = "bold" if ours else "normal"
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=6.2,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        zorder=6,
    )

ax.set_xscale("log")
ax.set_xlim(10, 20000)
ax.set_ylim(0.30, 0.405)

ax.set_ylabel("HEST-1k Average PCC", fontsize=7.5, color=INK, labelpad=4)
ax.set_xlabel("TFLOPs/slide", fontsize=7.5, color=INK, labelpad=4)

ax.set_xticks(xticks)
ax.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: (f"{v/1000:g}k" if v >= 1000 else f"{v:g}"))
)
ax.set_xticks([], minor=True)

ax.set_yticks([0.30, 0.32, 0.34, 0.36, 0.38, 0.40])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
ax.set_yticks([0.31, 0.33, 0.35, 0.37, 0.39], minor=True)

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRAY_LIGHT)
    ax.spines[side].set_linewidth(0.6)

ax.grid(True, which="major", axis="y", color=GRID_MAJOR, linewidth=0.5, zorder=0)
ax.grid(True, which="minor", axis="y", color=GRID_MINOR, linewidth=0.35, zorder=0)

for xv in xticks:
    color = GRID_MAJOR if xv in DARK_X else GRID_MINOR
    lw = 0.55 if xv in DARK_X else 0.35
    ax.axvline(xv, color=color, linewidth=lw, zorder=0)

xmin, xmax = ax.get_xlim()
for xv in FINE_X:
    if xmin < xv < xmax:
        ax.axvline(xv, color=GRID_MINOR, linewidth=0.25, alpha=0.8, zorder=0)

ax.tick_params(axis="both", which="major", length=2.5, width=0.6, labelsize=6.3, colors=BLACK)
ax.tick_params(axis="y", which="minor", length=0)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_color(BLACK)

# OLS fit PCC = b0 + b1 * log10(TFLOPs/slide) on the foundation models only (ours == False).
pfm_x = np.array([x for _, x, y, ours in DATA if not ours])
pfm_y = np.array([y for _, x, y, ours in DATA if not ours])
log_x = np.log10(pfm_x)
b1, b0 = np.polyfit(log_x, pfm_y, 1)
y_hat = b0 + b1 * log_x
ss_res = float(np.sum((pfm_y - y_hat) ** 2))
ss_tot = float(np.sum((pfm_y - pfm_y.mean()) ** 2))
r2 = 1 - ss_res / ss_tot
print(
    f"[trend] PCC = {b0:.4f} + {b1:.4f} * log10(TFLOPs/slide)   (R^2 = {r2:.3f}, n={len(pfm_x)} PFMs)"
)

x_pfm_min, x_pfm_max = pfm_x.min(), pfm_x.max()
xs_fit = np.logspace(np.log10(x_pfm_min), np.log10(x_pfm_max), 200)
ys_fit = b0 + b1 * np.log10(xs_fit)
ax.plot(xs_fit, ys_fit, color=TREND_COLOR, linewidth=1.1, zorder=3, linestyle=(0, (1, 1.3)))

xs_ext = np.logspace(np.log10(10), np.log10(x_pfm_min), 60)
ys_ext = b0 + b1 * np.log10(xs_ext)
ax.plot(
    xs_ext, ys_ext, color=TREND_COLOR, linewidth=0.8, linestyle=(0, (1, 1.3)), zorder=3, alpha=0.85
)

EQN_FONTSIZE = 6.0
sign = "+" if b0 >= 0 else "-"
line1 = rf"$\mathrm{{PCC}} = {b1:.4f}\,\log_{{10}}(\mathrm{{TFLOPs}}) {sign} {abs(b0):.4f}$"
line2 = rf"$R^2 = {r2:.2f}$"
x_anchor = 2000.0
y_on_line = b0 + b1 * np.log10(x_anchor)
x_text = 450.0
y_text = 0.337

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
prefix = rf"$\mathrm{{PCC}} = {b1:.4f}\,\log_{{10}}$"
tmp = ax.text(x_text, y_text, prefix, ha="left", va="top", fontsize=EQN_FONTSIZE, alpha=0)
fig.canvas.draw()
bbox_paren = tmp.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
x_paren = bbox_paren.x1
tmp.remove()

ann1 = ax.annotate(
    line1,
    xy=(x_anchor, y_on_line),
    xytext=(x_text, y_text),
    xycoords="data",
    textcoords="data",
    ha="left",
    va="top",
    fontsize=EQN_FONTSIZE,
    color=EQN_DARK_COLOR,
    arrowprops=dict(
        arrowstyle="-|>",
        color=EQN_DARK_COLOR,
        linewidth=0.7,
        connectionstyle="arc3,rad=-0.3",
        shrinkA=1,
        shrinkB=3,
    ),
)
ax.text(
    x_paren,
    y_text - 0.0045,
    line2,
    ha="center",
    va="top",
    fontsize=EQN_FONTSIZE,
    color=EQN_DARK_COLOR,
)

# Pareto frontier over all points, ePathST included: a point stays if it beats the PCC of every
# cheaper one.
all_pts = sorted(((x, y, label) for label, x, y, ours in DATA), key=lambda p: p[0])
frontier = []
running_max = -np.inf
for x, y, label in all_pts:
    if y > running_max:
        frontier.append((x, y, label))
        running_max = y
print("[pareto] frontier:")
for x, y, label in frontier:
    print(f"    {label:22s} {x:10.2f} TFLOPs/slide   PCC={y}")

fx = [p[0] for p in frontier]
fy = [p[1] for p in frontier]
ax.plot(
    fx,
    fy,
    color=PARETO_COLOR,
    linewidth=0.9,
    linestyle=(0, (1, 1.2)),
    marker="o",
    markersize=3.2,
    markerfacecolor="none",
    markeredgecolor=PARETO_COLOR,
    markeredgewidth=0.7,
    zorder=3.5,
)

legend_elems = [
    Line2D(
        [0],
        [0],
        marker="D",
        color="none",
        markerfacecolor=GRAY,
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.5,
        label="Pathology foundation models",
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        color="none",
        markerfacecolor=RED,
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=0.55,
        label="ePathST (ours)",
    ),
    Line2D(
        [0],
        [0],
        color=TREND_COLOR,
        linewidth=1.1,
        linestyle=(0, (1, 1.3)),
        label="OLS trend (PFMs only)",
    ),
    Line2D(
        [0],
        [0],
        color=PARETO_COLOR,
        linewidth=0.9,
        linestyle=(0, (1, 1.2)),
        marker="o",
        markerfacecolor="none",
        markeredgecolor=PARETO_COLOR,
        label="Pareto frontier (all models)",
    ),
]
leg = ax.legend(
    handles=legend_elems,
    loc="lower right",
    frameon=False,
    fontsize=5.0,
    handletextpad=0.4,
    borderaxespad=0.3,
    labelspacing=0.35,
)
for text in leg.get_texts():
    text.set_color(INK)

plt.tight_layout(pad=0.3)
out_path = FIG_OUT / "tflops_vs_hest1k_pcc_ieee_trend_pareto.png"
plt.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
print(f"saved {out_path}")
