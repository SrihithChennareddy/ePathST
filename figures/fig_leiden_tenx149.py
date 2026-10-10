#!/usr/bin/env python
"""Draws the COAD/TENX149 Leiden clustering comparison: WSI, GT, ePathST-E, ePathST, H-Optimus-0.
Reads figures/data/leiden_tenx149/; writes leiden_tenx149_comparison.png to the output dir."""
import json

import numpy as np
import openslide
import anndata as ad
from scipy.spatial import cKDTree
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from _paths import BENCH_ROOT, DATA_DIR, FIG_OUT, WSI_ROOT

# ePathST-E and ePathST cluster labels come from the earlier training run, kept for exact
# reproducibility; H-Optimus-0 labels are recomputed (see figures/README.md).
LEIDEN_DIR = DATA_DIR / "leiden_tenx149"
WSI_TIF = str(WSI_ROOT / "colon/wsis/TENX149.tif")
OUT_PATH = FIG_OUT / "leiden_tenx149_comparison.png"

CANCER, SAMPLE = "COAD", "TENX149"
COLS = ["ePathST-E", "ePathST", "H-Optimus-0"]

npz = np.load(LEIDEN_DIR / "tenx149_results.npz")
meta = json.loads((LEIDEN_DIR / "tenx149_meta.json").read_text())
coords_um = npz["coords_um"]
gt_labels = npz["gt_labels"]

adata = ad.read_h5ad(BENCH_ROOT / CANCER / "adata" / f"{SAMPLE}.h5ad")
tech = next(iter(adata.uns["spatial"].keys()))
sf = adata.uns["spatial"][tech]["scalefactors"]
mpp = 55.0 / sf["spot_diameter_fullres"]  # Visium spots are 55 um wide; um per full-res pixel

slide = openslide.OpenSlide(WSI_TIF)
coords_px = coords_um / mpp
margin = 0.04 * max(np.ptp(coords_px[:, 0]), np.ptp(coords_px[:, 1]))
x0 = max(0, coords_px[:, 0].min() - margin)
y0 = max(0, coords_px[:, 1].min() - margin)
x1 = min(slide.dimensions[0], coords_px[:, 0].max() + margin)
y1 = min(slide.dimensions[1], coords_px[:, 1].max() + margin)
width_px, height_px = x1 - x0, y1 - y0
# Pick the pyramid level that gives roughly 900 px on the long side of the crop.
desired_downsample = max(width_px, height_px) / 900.0
level = slide.get_best_level_for_downsample(max(1.0, desired_downsample))
downsample = slide.level_downsamples[level]
size = (max(1, int(width_px / downsample)), max(1, int(height_px / downsample)))
region = slide.read_region((int(x0), int(y0)), level, size).convert("RGB")
wsi_img = np.asarray(region)

# Palette is cut to the number of GT clusters; GT and predictions share the colormap.
n_clusters = meta["gt_n_clusters"]
PALETTE = ["#1f6fb2", "#c0392b", "#8fbf5f", "#e08a2b", "#c9d67a", "#a9c9e8", "#8e6bb0", "#5a5a5a"]
cmap = ListedColormap(PALETTE[: max(n_clusters, len(set(gt_labels)))])
vmax_c = len(PALETTE[: max(n_clusters, len(set(gt_labels)))]) - 1

x, y = coords_um[:, 0], -coords_um[:, 1]
span_um = max(x.max() - x.min(), 1.0)
tree = cKDTree(coords_um)
nn_dist, _ = tree.query(coords_um, k=2)
pitch_um = np.median(nn_dist[:, 1])

n_cols = 2 + len(COLS)
fig, axes = plt.subplots(1, n_cols, figsize=(3.0 * n_cols, 3.4), dpi=300)

# Marker size from the axes width (3.0 in per column less padding) so spots tile without gaps.
axes_width_in = 3.0 - 0.3
pts_per_um = axes_width_in * 72 / span_um
s = max(0.5, (pts_per_um * pitch_um * 1.0) ** 2)

ax = axes[0]
ax.imshow(wsi_img, aspect="auto")
ax.axis("off")
ax.set_title("WSI", fontsize=14, fontweight="bold", pad=6)

ax = axes[1]
ax.scatter(x, y, c=gt_labels, cmap=cmap, vmin=0, vmax=vmax_c, s=s, marker="o", edgecolors="none")
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("GT", fontsize=14, fontweight="bold", pad=6)

for j, disp in enumerate(COLS, start=2):
    pred_labels = npz[f"pred_labels__{disp}"]
    ari = meta["ari_scores"][disp]
    ax = axes[j]
    ax.scatter(
        x, y, c=pred_labels, cmap=cmap, vmin=0, vmax=vmax_c, s=s, marker="o", edgecolors="none"
    )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{disp} ({ari:.2f})", fontsize=14, fontweight="bold", pad=6)

plt.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.02, wspace=0.05)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
print(f"saved {OUT_PATH}")
