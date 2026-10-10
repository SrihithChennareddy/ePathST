#!/usr/bin/env python
"""Step 1 follow-up of FINAL_GUIDE.md: rebuilds TCGA ensembles without collapsed fold models.
Frozen inference from the cached embeddings with the remaining saved fold bundles; no training."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.tcga import prediction
from epathst.utils import load_config

SRIHITH = Path("/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith")
MODELS = (
    SRIHITH
    / "cptac_hoptimus0_distill_hest_pna_work"
    / "results"
    / "leiden_official_hvg50"
    / "models"
)
PREDICTIONS = SRIHITH / "tcga" / "predictions" / "hex100_leiden_official_hvg50"
EMBEDDINGS = Path("/lustre/nvwulf/scratch/schennareddy/TCGA/embeddings_hex100")
HEADS = {"TCGA-BRCA": "IDC", "TCGA-COAD": "COAD", "TCGA-KIRC": "CCRCC"}

log = logging.getLogger("50_tcga_rebuild_ensemble")


def rebuild_slide(project, case, bundles, cfg, seed=0):
    z = np.load(EMBEDDINGS / project / f"{case}.npz")
    xy = z["xy_um"]
    preds = prediction.predict_from_embeddings(
        bundles,
        z["student"].astype(np.float32),
        z["teacher"].astype(np.float32),
        xy,
        case,
        cfg,
        "cpu",
        seed,
    )
    return xy, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="configs/analysis_plan.json")
    ap.add_argument("--out", default="results/tcga_final")
    ap.add_argument("--project", default="TCGA-BRCA")
    ap.add_argument("--exclude", nargs="+", default=["IDC_fold2"])
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="3 slides; also checks an all-fold rebuild against the stored files",
    )
    args = ap.parse_args()
    out = (
        Path(args.out)
        / ("predictions_rebuilt_smoke" if args.smoke else "predictions_rebuilt")
        / args.project
    )
    out.mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(Path(args.out) / "logs" / "50_tcga_rebuild_ensemble.log"),
            logging.StreamHandler(),
        ],
    )
    cfg = load_config("configs/base.yaml")
    head = HEADS[args.project]
    all_bundles = prediction.load_bundles(MODELS, head, "cpu")
    kept = [b for b in all_bundles if b["name"] not in args.exclude]
    log.info(
        "%s: %d fold bundles, kept %s", args.project, len(all_bundles), [b["name"] for b in kept]
    )
    qc = pd.read_csv(PREDICTIONS / "qc_all.csv")
    cases = qc[(qc["project"] == args.project) & (qc["status"] == "ok")]["case"].tolist()
    if args.smoke:
        cases = cases[:3]
        for case in cases:
            _, full = rebuild_slide(args.project, case, all_bundles, cfg)
            stored = np.load(PREDICTIONS / args.project / f"{case}.npz")
            gap = max(
                np.abs(full[s] - stored[f"pred__{s}"].astype(np.float32)).max()
                for s in prediction.SYSTEMS
            )
            log.info("%s all-fold rebuild vs stored: max |diff| %.2e", case, gap)
            assert gap < 5e-3, "all-fold rebuild does not reproduce the stored ensemble"
    for i, case in enumerate(cases):
        target = out / f"{case}.npz"
        if target.exists():
            continue
        xy, preds = rebuild_slide(args.project, case, kept, cfg)
        np.savez_compressed(
            target,
            x_um=xy[:, 0],
            y_um=xy[:, 1],
            genes=np.array(kept[0]["meta"]["genes"]),
            n_folds=len(kept),
            folds=np.array([b["name"] for b in kept]),
            pitch_um=100.0,
            **{f"pred__{s}": p.astype(np.float16) for s, p in preds.items()},
        )
        if i % 50 == 0:
            log.info("%d/%d %s", i + 1, len(cases), case)
    (out.parent / f"{args.project}_manifest.json").write_text(
        json.dumps(
            {
                "project": args.project,
                "head": head,
                "kept_folds": [b["name"] for b in kept],
                "excluded": args.exclude,
                "slides": len(cases),
                "seed": 0,
                "source_embeddings": str(EMBEDDINGS),
                "models": str(MODELS),
            },
            indent=1,
        )
    )
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
