#!/usr/bin/env python
"""Step 3: patient-level 80/10/10 CPTAC split for each seed, stratified by
cancer type, with leakage assertions against the patch manifest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathology_st.patient_split import assert_no_patch_leakage, cptac_patient_split
from pathology_st.utils import load_config, resolve_run_paths


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
        assert_no_patch_leakage(patch_manifest, split_df)
        out_path = paths["manifests"] / f"cptac_split_seed{seed}.csv"
        split_df.to_csv(out_path, index=False)
        counts = split_df.groupby(["cancer_type", "split"]).size().unstack(fill_value=0)
        print(f"[03_cptac_patient_split] seed {seed}:\n{counts}\nwrote {out_path}\n")


if __name__ == "__main__":
    main()
