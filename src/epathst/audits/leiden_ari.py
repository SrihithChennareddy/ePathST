"""Independently re-derives the official-cohort Leiden numbers of one run directory.
Run as `src/scripts/25_leiden_official_panel.py verify <run results dir>`; writes
ari_verification.json into the run dir.

Three checks:
 (a) ARI/AMI recomputed with sklearn from the saved cluster labels vs per_sample_leiden.csv;
 (b) labels re-derived from the saved measured/predicted matrices vs the saved labels
     (ARI must be 1.0);
 (c) the saved measured matrix vs expression re-read from the raw benchmark h5ad (official
     panel, log1p, fold standardizer).
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from epathst.evaluation.leiden import RESOLUTIONS, leiden_labels

BENCH = Path("/lustre/nvwulf/scratch/schennareddy/HEST-1k/hest_bench")
SYS = ["ePathST-E", "ePathST", "H-Optimus-0"]


def main(run: str):
    csv = pd.read_csv(f"{run}/per_sample_leiden.csv")
    out = {
        "run": run,
        "n_label_files": 0,
        "a_max_abs_diff_ari": 0.0,
        "a_max_abs_diff_ami": 0.0,
        "a_rows_checked": 0,
        "b_min_ari_saved_vs_rederived": 1.0,
        "b_matrices_checked": 0,
        "c": [],
    }
    for f in sorted(glob.glob(f"{run}/labels/*.npz")):
        cancer, sid = Path(f).stem.split("_", 1)
        z = np.load(f, allow_pickle=True)
        out["n_label_files"] += 1
        # (a) metrics from saved labels
        for system in SYS:
            for res in RESOLUTIONS:
                row = csv[(csv.sample_id == sid) & (csv.system == system) & (csv.resolution == res)]
                assert len(row) == 1, (sid, system, res, len(row))
                g, p = z[f"gt__res{res}"], z[f"{system}__res{res}"]
                out["a_max_abs_diff_ari"] = max(
                    out["a_max_abs_diff_ari"], abs(adjusted_rand_score(g, p) - row.ari.iloc[0])
                )
                out["a_max_abs_diff_ami"] = max(
                    out["a_max_abs_diff_ami"],
                    abs(adjusted_mutual_info_score(g, p) - row.ami.iloc[0]),
                )
                out["a_rows_checked"] += 1
        # (b) labels re-derived from saved matrices
        mats = {"gt": z["y_true"], **{s: z[f"y_pred__{s}"] for s in SYS}}
        for name, m in mats.items():
            lab = leiden_labels(m, seed=0)
            for res in RESOLUTIONS:
                saved = z[f"gt__res{res}"] if name == "gt" else z[f"{name}__res{res}"]
                out["b_min_ari_saved_vs_rederived"] = min(
                    out["b_min_ari_saved_vs_rederived"], adjusted_rand_score(saved, lab[res])
                )
            out["b_matrices_checked"] += 1
        # (c) measured matrix vs raw expression (one sample per cancer)
        if not any(c["cancer"] == cancer for c in out["c"]):
            fold = next(
                i
                for i in range(12)
                if os.path.exists(f"{run}/models/{cancer}_fold{i}/meta.json")
                and sid
                in json.load(open(f"{run}/models/{cancer}_fold{i}/meta.json"))["test_samples"]
            )
            sd = np.load(f"{run}/models/{cancer}_fold{fold}/standardizer.npz")
            genes = [str(g) for g in z["genes"]]
            tdf = pd.read_csv(BENCH / cancer / "splits" / f"test_{fold}.csv")
            ad_ = ad.read_h5ad(BENCH / cancer / tdf[tdf.sample_id == sid].iloc[0]["expr_path"])
            bc = [u.split("::", 1)[1] for u in z["spot_uid"]]
            a = ad_[bc, genes].copy()
            a.X = a.X.astype(np.float64)
            import scanpy as sc

            sc.pp.log1p(a)
            raw = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
            exp = (raw - sd["mu"]) / (sd["sigma"] + float(sd["eps"]))
            out["c"].append(
                {
                    "cancer": cancer,
                    "sample": sid,
                    "fold": int(fold),
                    "max_abs_diff": float(np.abs(exp - z["y_true"]).max()),
                }
            )
    json.dump(out, open(f"{run}/ari_verification.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items()}, indent=1))
