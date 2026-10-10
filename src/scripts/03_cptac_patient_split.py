#!/usr/bin/env python
"""Patient-level 80/10/10 train/val/test split of the CPTAC cohort, one per seed.

Stage 03 of the ePathST pipeline: splits are stratified by cancer type and grouped by patient, and
checked against the patch manifest so no patient's patches land in two splits. Writes
manifests/cptac_split_seed{seed}.csv, read by distillation training (05) and evaluation (06).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.data.splits import assert_no_patch_leakage, cptac_patient_split
from epathst.utils import load_config, resolve_run_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)

    selected = pd.read_csv(paths["manifests"] / "cptac_selected_wsi.csv")
    patch_manifest = pd.read_csv(paths["manifests"] / "cptac_patch_manifest.csv")

    for seed in cfg["seeds"]:
        split_df = cptac_patient_split(selected, cfg, seed)
        # Leakage gate: no patient may contribute patches to more than one split.
        assert_no_patch_leakage(patch_manifest, split_df)

        out_path = paths["manifests"] / f"cptac_split_seed{seed}.csv"
        split_df.to_csv(out_path, index=False)
        counts = split_df.groupby(["cancer_type", "split"]).size().unstack(fill_value=0)
        print(f"[03_cptac_patient_split] seed {seed}:\n{counts}\nwrote {out_path}\n")


if __name__ == "__main__":
    main()
