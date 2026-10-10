"""Checks, per HEST cohort sample, which official 50-gene panel genes exist in its expression file.
Run as `src/scripts/26_hest_audits.py panel-coverage`.

Matches gene symbols, resolving Ensembl-indexed files through their SYMBOL column; only var
metadata is read (h5py). Writes
results/hest_panel_coverage/official_panel_gene_coverage_per_sample.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pandas as pd

from epathst.utils import load_config, resolve_run_paths

BENCH = Path("/lustre/nvwulf/scratch/schennareddy/HEST-1k/hest_bench")
PANEL_DIR = Path(
    "/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/hf_cache/huggingface/hub/"
    "datasets--MahmoodLab--hest-bench/snapshots/b039b5965ed7092f280d4f336da7972ef9a10e1a"
)


def symbols(path: str):
    with h5py.File(path, "r") as f:
        v = f["var"]
        idx = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in v[v.attrs.get("_index", "_index")][:]
        ]
        if idx and idx[0].startswith("ENSG") and "SYMBOL" in v:
            s = v["SYMBOL"]
            if isinstance(s, h5py.Group):
                cats = [c.decode() if isinstance(c, bytes) else str(c) for c in s["categories"][:]]
                s = [cats[c] for c in s["codes"][:]]
            else:
                s = [x.decode() if isinstance(x, bytes) else str(x) for x in s[:]]
            return set(s), len(idx)
        return set(idx), len(idx)


def main():
    cfg = load_config("configs/base.yaml")
    paths = resolve_run_paths(cfg)
    hs = pd.read_csv(paths["manifests"] / "hest_samples.csv")
    panel = {}
    for c in ("IDC", "COAD", "CCRCC"):
        p = BENCH / c / "var_50genes.json"
        panel[c] = json.loads(
            (p if p.exists() else PANEL_DIR / c / "var_50genes.json").read_text()
        )["genes"]
    rows = []
    for _, r in hs.iterrows():
        vn, n = symbols(r["expression_path"])
        miss = [g for g in panel[r["cancer"]] if g not in vn]
        rows.append(
            {
                "cancer": r["cancer"],
                "sample": r["sample_id"],
                "tech": r["st_technology"],
                "n_genes": n,
                "targeted_panel": n < 2000,
                "n_missing": len(miss),
                "missing": ";".join(miss),
            }
        )
    out = paths["results"] / "hest_panel_coverage"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "official_panel_gene_coverage_per_sample.csv", index=False)
    w = df[~df["targeted_panel"]]
    print(
        w.groupby("cancer")
        .agg(
            samples=("sample", "count"),
            fully_covered=("n_missing", lambda x: (x == 0).sum()),
            max_missing=("n_missing", "max"),
        )
        .to_string()
    )
    print("wrote", out)
