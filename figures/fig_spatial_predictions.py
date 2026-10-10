#!/usr/bin/env python
"""Draws the spatial gene panel: WSI | ground truth | ePathST | H-Optimus-0 for IDC, COAD, CCRCC.
Reads figures/data/spatial_predictions/ and HEST-bench slides; writes one PNG per colormap."""
import json

import h5py
import numpy as np
import openslide
import anndata as ad
from scipy.spatial import cKDTree
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import BENCH_ROOT, DATA_DIR, FIG_OUT, WSI_ROOT

# ePathST predictions come from the earlier training run, kept for exact reproducibility;
# H-Optimus-0 predictions are recomputed (see figures/README.md).
NPZ_DIR = DATA_DIR / "spatial_predictions"
OUT_DIR = FIG_OUT

# IDC has no readable full-resolution .tif (its HEST wsi_path points into another group's
# storage), so its WSI column falls back to a mosaic of per-spot patch colours.
WSI_TIF = {
    "COAD": str(WSI_ROOT / "colon/wsis/TENX148.tif"),
    "CCRCC": str(WSI_ROOT / "kidney/wsis/INT15.tif"),
}

ROWS = ["IDC", "COAD", "CCRCC"]  # one gene and one sample per cancer

# Green-toned sequential colormaps; one output PNG per entry.
GREEN_CMAPS = {
    "greens": "Greens",  # white -> pure dark green
    "ylgn": "YlGn",  # pale yellow-green -> deep forest green
    "bugn": "BuGn",  # pale blue-green -> deep teal-green
    "gnbu": "GnBu",  # pale green -> deep blue-green
    "summer": "summer",  # yellow-green -> deep sea-green (matplotlib built-in)
}


def get_mpp(adata) -> float:
    tech = next(iter(adata.uns["spatial"].keys()))
    sf = adata.uns["spatial"][tech]["scalefactors"]
    # Visium spots are 55 um wide, so this is um per full-resolution pixel.
    return 55.0 / sf["spot_diameter_fullres"]


def load(cancer):
    npz = np.load(NPZ_DIR / f"{cancer}_infer.npz")
    meta = json.loads((NPZ_DIR / f"{cancer}_infer_meta.json").read_text())
    coords_um = npz["coords_um"]
    sample = meta["sample"]
    adata = ad.read_h5ad(BENCH_ROOT / cancer / "adata" / f"{sample}.h5ad")
    mpp = get_mpp(adata)
    out = {
        "gene": meta["gene"],
        "sample": sample,
        "coords_um": coords_um,
        "gt": npz["ground_truth"],
        "a8": npz["a8_pred"],
        "hoptimus": npz["hoptimus_pred"],
        "r_a8": meta["r_a8_sample"],
        "r_hoptimus": meta["r_hoptimus_sample"],
    }
    if cancer in WSI_TIF:
        slide = openslide.OpenSlide(WSI_TIF[cancer])
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
        out["wsi_img"] = np.asarray(region)
    else:
        with h5py.File(BENCH_ROOT / cancer / "patches" / f"{sample}.h5", "r") as f:
            patch_coords_px = f["coords"][:].astype(np.float64)
            patch_rgb_mean = f["img"][:].mean(axis=(1, 2)) / 255.0
        patch_coords_um = patch_coords_px * mpp
        tree = cKDTree(patch_coords_um)
        _, idx = tree.query(coords_um, k=1)
        out["wsi_rgb"] = patch_rgb_mean[idx]
    return out


print("[fig] loading real per-spot data + real WSI thumbnails...", flush=True)
rows_data = [load(c) for c in ROWS]


def build_figure(cmap_name, out_path, use_percentile=True):
    fig, axes = plt.subplots(3, 4, figsize=(9.4, 6.6), dpi=450)
    # Invisible (s=0) scatters set the data limits, and the layout must be drawn before
    # get_window_extent() below reports true per-axes pixel sizes.
    for i, d in enumerate(rows_data):
        coords = d["coords_um"]
        axes[i, 0].scatter(coords[:, 0], -coords[:, 1], s=0)
        for j in range(1, 4):
            axes[i, j].scatter(coords[:, 0], -coords[:, 1], s=0)
    plt.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.02, wspace=0.05, hspace=0.35)
    fig.canvas.draw()

    for i, d in enumerate(rows_data):
        coords = d["coords_um"]
        x, y = coords[:, 0], -coords[:, 1]
        gt, a8, hoptimus = d["gt"], d["a8"], d["hoptimus"]
        # One colour scale per row, shared by truth and both predictions. The 2nd-98th
        # percentile window stops a few extreme spots from washing out the rest of the tissue.
        pooled = np.concatenate([gt, a8, hoptimus])
        if use_percentile:
            vmin, vmax = np.percentile(pooled, [2, 98])
        else:
            vmin, vmax = pooled.min(), pooled.max()

        tree = cKDTree(coords)
        nn_dist, _ = tree.query(coords, k=2)
        pitch_um = np.median(nn_dist[:, 1])
        x_span_um = max(coords[:, 0].max() - coords[:, 0].min(), 1.0)
        y_span_um = max(coords[:, 1].max() - coords[:, 1].min(), 1.0)
        # Panels fill their cell with no fixed aspect ratio, so a wide sample like IDC is
        # stretched much more along one axis. Size markers from the less-stretched axis;
        # sizing from x alone left gaps between spot rows on wide samples.
        bbox = axes[i, 1].get_window_extent()
        pts_per_um_x = (bbox.width * 72 / fig.dpi) / x_span_um
        pts_per_um_y = (bbox.height * 72 / fig.dpi) / y_span_um
        pts_per_um = min(pts_per_um_x, pts_per_um_y)
        s = max(0.6, (pts_per_um * pitch_um * 1.15) ** 2)

        ax = axes[i, 0]
        if "wsi_img" in d:
            ax.imshow(d["wsi_img"], aspect="auto")
        else:
            ax.scatter(x, y, c=d["wsi_rgb"], s=s, marker="o", edgecolors="none")
        ax.axis("off")
        ax.set_title("WSI", fontsize=12, fontweight="bold", pad=4)

        panels = [
            ("Ground Truth", gt),
            (f"ePathST ({d['r_a8']:.2f})", a8),
            (f"H-Optimus-0 ({d['r_hoptimus']:.2f})", hoptimus),
        ]

        for j, (title, vals) in enumerate(panels, start=1):
            ax = axes[i, j]
            ax.scatter(
                x,
                y,
                c=vals,
                cmap=cmap_name,
                vmin=vmin,
                vmax=vmax,
                s=s,
                marker="o",
                edgecolors="none",
            )
            ax.axis("off")
            ax.set_title(title, fontsize=12, fontweight="bold", pad=4)

        axes[i, 0].text(
            -0.16,
            0.5,
            f"{ROWS[i]} ({d['gene']})",
            transform=axes[i, 0].transAxes,
            fontsize=15,
            fontweight="bold",
            rotation=90,
            va="center",
            ha="center",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=450, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"saved {out_path}")


# Base figure: viridis with raw per-row min/max; the green variants use the percentile window.
build_figure("viridis", OUT_DIR / "ePathST_vs_HOptimus0_two_gene_panel.png", use_percentile=False)

for tag, cmap_name in GREEN_CMAPS.items():
    build_figure(cmap_name, OUT_DIR / f"ePathST_vs_HOptimus0_two_gene_panel_{tag}.png")

for d in rows_data:
    print(
        f"  gene={d['gene']:10s} sample={d['sample']:10s} r_ePathST={d['r_a8']:.4f} r_HOptimus0={d['r_hoptimus']:.4f}"
    )
