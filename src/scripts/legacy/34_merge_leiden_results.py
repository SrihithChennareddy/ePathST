#!/usr/bin/env python
"""Joins per-sample Leiden ARIs of ePathST-E/ePathST (32_leiden_full_cohort.py) with the zero-shot
H-Optimus-0 ARIs (33_leiden_hoptimus_baseline.py).

The two use different ground-truth clusterings (32: fold-selected HVG1000; 33: each sample's own
top-variance genes), so the H-Optimus-0 column is a reference, not a matched comparison.
Earlier variant kept for provenance; official-panel Leiden is now
`src/scripts/25_leiden_official_panel.py official`."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DIR = (
    "/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/"
    "cptac_hoptimus0_distill_hest_pna/leiden_clustering"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--ours", default="per_sample_ari.csv")
    args = ap.parse_args()
    d = Path(args.dir)

    ours = pd.read_csv(d / args.ours)
    hopt = pd.read_csv(d / "per_sample_ari_hoptimus.csv")[
        ["sample_id", "gt_resolution", "gt_n_clusters", "ari_HOptimus0"]
    ]
    hopt = hopt.rename(
        columns={"gt_resolution": "gt_resolution_hopt", "gt_n_clusters": "gt_n_clusters_hopt"}
    )
    merged = ours.merge(hopt, on="sample_id", how="outer", indicator=True)
    print(merged["_merge"].value_counts().to_string())
    merged = merged[merged["_merge"] == "both"].drop(columns="_merge")
    merged.to_csv(d / "per_sample_ari_all_models.csv", index=False)

    cols = [c for c in merged.columns if c.startswith("ari_")]
    rows = []
    for cancer, g in list(merged.groupby("cancer")) + [("ALL", merged)]:
        row = {"cancer": cancer, "n_samples": len(g)}
        for c in cols:
            row[f"{c}_mean"], row[f"{c}_std"] = g[c].mean(), g[c].std()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(d / "ari_summary_all_models.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
