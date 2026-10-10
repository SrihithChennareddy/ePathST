#!/usr/bin/env python
"""Per-slide geometry of the downloaded TCGA diagnostic slides: scan size (mm), native mpp, tissue
area (mm^2) and the spot count a hexagonal grid of pitch d would place on the tissue.
Used to size the 100 um pseudo-spot lattice; reads slide headers and one thumbnail per slide only.

Tissue area uses the same saturation/value thumbnail mask as patch extraction;
N ~= tissue_area / ((sqrt(3)/2) d^2).
Earlier variant kept for provenance; the lattice is now built in `src/scripts/24_tcga_validation.py
predict`.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from epathst.data.cptac_select import tissue_mask_from_thumbnail
from epathst.data.wsi_patches import open_slide

TCGA = Path("/lustre/nvwulf/scratch/schennareddy/TCGA")
OUT = Path("/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/tcga/results")
PITCH_UM = 100.0
THUMB_MAX = 2048


def one(rec):
    case, project, path = rec
    try:
        r = open_slide(path)
        w0, h0 = r.dimensions
        mpp = r.mpp
        thumb = r.get_thumbnail(THUMB_MAX)
        if hasattr(r, "close"):
            r.close()
        m = tissue_mask_from_thumbnail(thumb)
        use = mpp or 0.5
        th, tw = m.shape
        px_area_um2 = (w0 * use / tw) * (h0 * use / th)
        tissue_mm2 = float(m.sum() * px_area_um2 / 1e6)
        ys, xs = np.where(m)
        bbox_w = (xs.max() - xs.min() + 1) * (w0 * use / tw) / 1e3 if len(xs) else 0.0
        bbox_h = (ys.max() - ys.min() + 1) * (h0 * use / th) / 1e3 if len(ys) else 0.0
        return {
            "case": case,
            "project": project,
            "mpp": mpp,
            "scan_w_mm": w0 * use / 1e3,
            "scan_h_mm": h0 * use / 1e3,
            "tissue_mm2": tissue_mm2,
            "tissue_bbox_w_mm": float(bbox_w),
            "tissue_bbox_h_mm": float(bbox_h),
            "hex_spots_100um": tissue_mm2 * 1e6 / (np.sqrt(3) / 2 * PITCH_UM**2),
            "status": "ok",
        }
    except Exception as e:
        return {"case": case, "project": project, "status": f"FAILED {type(e).__name__}: {e}"[:160]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    recs = []
    for p in ("TCGA-BRCA", "TCGA-COAD", "TCGA-KIRC"):
        s = pd.read_csv(TCGA / "metadata" / f"{p}_slides.csv")
        s = s[s["status"] == "downloaded"]
        recs += [(fn[:12], p, lp) for fn, lp in zip(s["file_name"], s["local_path"])]
    with mp.Pool(args.workers) as pool:
        rows = pool.map(one, recs, chunksize=4)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "slide_geometry.csv", index=False)
    ok = df[df["status"] == "ok"]
    print("failed:", int((df["status"] != "ok").sum()), "of", len(df))
    print(
        ok.groupby("project")
        .agg(
            slides=("case", "count"),
            tissue_mm2_median=("tissue_mm2", "median"),
            tissue_mm2_p10=("tissue_mm2", lambda x: x.quantile(0.1)),
            tissue_mm2_p90=("tissue_mm2", lambda x: x.quantile(0.9)),
            hex_spots_median=("hex_spots_100um", "median"),
            hex_spots_p10=("hex_spots_100um", lambda x: x.quantile(0.1)),
            hex_spots_p90=("hex_spots_100um", lambda x: x.quantile(0.9)),
        )
        .round(0)
        .to_string()
    )
    print("native mpp counts:", ok["mpp"].round(2).value_counts(dropna=False).to_dict())
    print(
        "scan size mm (median w x h):",
        round(ok.scan_w_mm.median(), 1),
        "x",
        round(ok.scan_h_mm.median(), 1),
        "| tissue bbox mm (median w x h):",
        round(ok.tissue_bbox_w_mm.median(), 1),
        "x",
        round(ok.tissue_bbox_h_mm.median(), 1),
    )


if __name__ == "__main__":
    main()
