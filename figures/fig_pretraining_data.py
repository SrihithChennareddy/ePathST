"""Draws Fig. 3: pretraining data of every model as table panels a-d plus a merged figure.
Reads figures/data/pretraining_table.csv; writes PNG/PDF/TeX/CSV to fig3_pretraining_data/."""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

from _paths import DATA_DIR, FIG_OUT

OUT = FIG_OUT / "fig3_pretraining_data"
OUT.mkdir(parents=True, exist_ok=True)

# Columns: model, params (M), relative size, HEST avg PCC, slides, patients, tiles/patches,
# tile note, access, data source, evidence. Values are as reported by the model authors
# ('n/r' = not reported; sources in pretraining_sources.md). Read as strings with NA parsing
# off so entries such as 'n/r' and '>' lower bounds print verbatim.
df = pd.read_csv(DATA_DIR / "pretraining_table.csv", dtype=str, keep_default_na=False)
df.to_csv(OUT / "fig3_master_table.csv", index=False)

PANELS = [
    (
        "a",
        "Pretraining slides",
        "fig3a_slides",
        ["model", "slides", "patients"],
        ["Model", "Slides (WSIs)", "Patients"],
        [0.40, 0.36, 0.24],
    ),
    (
        "b",
        "Pretraining tiles / patches",
        "fig3b_tiles",
        ["model", "tiles", "tile_note"],
        ["Model", "Tiles / patches", "Notes"],
        [0.30, 0.31, 0.39],
    ),
    (
        "c",
        "Pretraining data access",
        "fig3c_access",
        ["model", "access", "source"],
        ["Model", "Data", "Source"],
        [0.28, 0.13, 0.59],
    ),
    (
        "d",
        "Model size and HEST score",
        "fig3d_size_score",
        ["model", "params_M", "rel_size", "avg_pcc"],
        ["Model", "Params (M)", "Rel. size", "Avg. PCC"],
        [0.37, 0.21, 0.21, 0.21],
    ),
]
FOOT = {
    "a": "n/r: not reported. '>' marks a lower bound as stated by the authors.",
    "b": "† Only press coverage ('several hundred million tiles'); no paper or model card states a tile count.",
    "c": "Data access refers to the pretraining data; GigaPath, H-Optimus-0, UNI2-h, Virchow2 and CONCH v1.5 data are private.",
    "d": "Baselines: HEST linear probe. ePathST-E/ePathST: trained heads (not a like-for-like encoder comparison).",
}

for f in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
INK, MUTED, RULE, BAND, OURS = "#1a1a1a", "#555555", "#222222", "#f1f1f1", "#e4eef7"


def draw_panel(ax, letter, title, keys, heads, widths, fs=6.6, wrap=84):
    ax.set_axis_off()
    n = len(df)
    top, rh = 0.90, 0.0665
    ax.text(
        0.0,
        top + 0.035,
        f"({letter})  {title}",
        fontsize=fs + 1.6,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )
    xs = [sum(widths[:i]) for i in range(len(widths))]
    y = top
    ax.plot([0, 1], [y, y], color=RULE, lw=1.1, transform=ax.transAxes, clip_on=False)
    for x, w, h in zip(xs, widths, heads):
        ax.text(
            x + 0.005,
            y - rh / 2,
            h,
            fontsize=fs,
            fontweight="bold",
            color=INK,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )
    y -= rh
    ax.plot([0, 1], [y, y], color=RULE, lw=0.6, transform=ax.transAxes, clip_on=False)
    for i, r in df.iterrows():
        ours = "ours" in r["model"]
        if ours or i % 2 == 1:
            ax.add_patch(
                plt.Rectangle(
                    (0, y - rh),
                    1,
                    rh,
                    color=OURS if ours else BAND,
                    transform=ax.transAxes,
                    lw=0,
                    zorder=0,
                )
            )
        for x, w, k in zip(xs, widths, keys):
            ax.text(
                x + 0.005,
                y - rh / 2,
                str(r[k]),
                fontsize=fs,
                color=INK if k == "model" or k != "tile_note" else MUTED,
                fontweight="bold" if (ours and k == "model") else "normal",
                ha="left",
                va="center",
                transform=ax.transAxes,
            )
        y -= rh
    ax.plot([0, 1], [y, y], color=RULE, lw=1.1, transform=ax.transAxes, clip_on=False)
    ax.text(
        0.0,
        y - 0.03,
        textwrap.fill(FOOT[letter], wrap),
        fontsize=fs - 0.9,
        color=MUTED,
        ha="left",
        va="top",
        transform=ax.transAxes,
        linespacing=1.25,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(y - 0.20, top + 0.22)


# Merged figure at IEEE double-column width (7.16 in).
fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.9))
for ax, (letter, title, _, keys, heads, widths) in zip(axes.ravel(), PANELS):
    draw_panel(ax, letter, title, keys, heads, widths)
fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005, wspace=0.06, hspace=0.12)
fig.savefig(OUT / "fig3_combined.png", dpi=300)
fig.savefig(OUT / "fig3_combined.pdf")
plt.close(fig)

# Each panel alone at IEEE single-column width (3.5 in).
for letter, title, name, keys, heads, widths in PANELS:
    fig, ax = plt.subplots(figsize=(3.5, 2.95))
    draw_panel(ax, letter, title, keys, heads, widths, wrap=68)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005)
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def tex_escape(s):
    return (
        str(s)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("≈", r"$\approx$")
        .replace("≤", r"$\leq$")
        .replace("×", r"$\times$")
        .replace("µ", r"$\mu$")
        .replace("†", r"$^\dagger$")
        .replace("–", "--")
        .replace("—", "--")
        .replace("+", "+")
        .replace(">", r"$>$")
        .replace("<", r"$<$")
    )


with open(OUT / "fig3_panels.tex", "w") as f:
    f.write(
        "% Fig. 3 panels as booktabs tables (requires \\usepackage{booktabs}). Panels a-d; merge as subtables.\n"
    )
    for letter, title, _, keys, heads, widths in PANELS:
        f.write(
            f"\n% ({letter}) {title}\n\\begin{{tabular}}{{l{'l' * (len(keys) - 1)}}}\n\\toprule\n"
        )
        f.write(" & ".join(f"\\textbf{{{tex_escape(h)}}}" for h in heads) + " \\\\\n\\midrule\n")
        for _, r in df.iterrows():
            f.write(" & ".join(tex_escape(r[k]) for k in keys) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(f"% footnote: {FOOT[letter]}\n")

# The source list is shipped next to the figure files as their README.
(OUT / "README.md").write_text((DATA_DIR / "pretraining_sources.md").read_text())
print("wrote", OUT)
for p in sorted(OUT.iterdir()):
    print(f"  {p.name:28s} {p.stat().st_size / 1e3:8.1f} KB")
