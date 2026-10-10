"""Measures how reproducible the saved Leiden labels are.
Run as `src/scripts/25_leiden_official_panel.py reproducibility <run results dir>`.

For every saved matrix (measured + 3 systems) the identical clustering is re-run twice and
compared (ARI saved vs re-derived, and between the two re-runs), with n_spots recorded. Each
sample's ARI vs ground truth is then recomputed from the re-derived labels to show how far the
headline numbers move. Writes <run>/leiden_reproducibility.csv and prints a summary.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from epathst.evaluation.leiden import RESOLUTIONS, leiden_labels

SYS = ["ePathST-E", "ePathST", "H-Optimus-0"]


def main(run: str):
    rows, ari_rows = [], []
    for f in sorted(glob.glob(f"{run}/labels/*.npz")):
        cancer, sid = Path(f).stem.split("_", 1)
        z = np.load(f, allow_pickle=True)
        n = len(z["y_true"])
        lab = {}
        for name, m in {"gt": z["y_true"], **{s: z[f"y_pred__{s}"] for s in SYS}}.items():
            a, b = leiden_labels(m, seed=0), leiden_labels(m, seed=0)
            lab[name] = (a, b)
            for res in RESOLUTIONS:
                saved = z[f"gt__res{res}"] if name == "gt" else z[f"{name}__res{res}"]
                rows.append(
                    {
                        "cancer": cancer,
                        "sample": sid,
                        "n_spots": n,
                        "matrix": name,
                        "resolution": res,
                        "ari_saved_vs_rederived": adjusted_rand_score(saved, a[res]),
                        "ari_rederived_twice": adjusted_rand_score(a[res], b[res]),
                        "n_clusters_saved": len(set(saved)),
                        "n_clusters_rederived": len(set(a[res])),
                    }
                )
        for s in SYS:
            for res in RESOLUTIONS:
                ari_rows.append(
                    {
                        "cancer": cancer,
                        "sample": sid,
                        "system": s,
                        "resolution": res,
                        "ari_saved": adjusted_rand_score(z[f"gt__res{res}"], z[f"{s}__res{res}"]),
                        "ari_rederived_1": adjusted_rand_score(lab["gt"][0][res], lab[s][0][res]),
                        "ari_rederived_2": adjusted_rand_score(lab["gt"][1][res], lab[s][1][res]),
                    }
                )
    d, a = pd.DataFrame(rows), pd.DataFrame(ari_rows)
    d.to_csv(f"{run}/leiden_reproducibility.csv", index=False)
    a.to_csv(f"{run}/leiden_ari_rederived.csv", index=False)
    d5 = d[d.resolution == 0.5]
    d5 = d5.assign(over4096=d5.n_spots > 4096)
    print(
        "labels identical on re-run (ARI==1):",
        (d5.ari_saved_vs_rederived > 0.999999).mean().round(3),
        "| two fresh re-runs identical:",
        (d5.ari_rederived_twice > 0.999999).mean().round(3),
    )
    print(
        d5.groupby("over4096")
        .agg(
            matrices=("sample", "count"),
            identical_frac=("ari_saved_vs_rederived", lambda x: (x > 0.999999).mean()),
            min_ari=("ari_saved_vs_rederived", "min"),
            median_ari=("ari_saved_vs_rederived", "median"),
        )
        .round(3)
        .to_string()
    )
    a5 = a[a.resolution == 0.5]
    print("mean ARI vs ground truth @0.5 (32 samples): saved / re-derived #1 / re-derived #2")
    print(
        pd.concat(
            [
                a5.groupby("system")[c].mean()
                for c in ["ari_saved", "ari_rederived_1", "ari_rederived_2"]
            ],
            axis=1,
        )
        .round(4)
        .to_string()
    )
    print(
        "max |saved - rederived| per-sample ARI:",
        float((a5.ari_saved - a5.ari_rederived_1).abs().max().round(3)),
        "| mean abs:",
        float((a5.ari_saved - a5.ari_rederived_1).abs().mean().round(4)),
    )
