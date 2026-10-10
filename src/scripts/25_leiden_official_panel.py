"""Runs one stage of the Leiden spatial-domain analysis on the official 50-gene HEST panels.
Entry point for epathst.evaluation and epathst.audits; ARI is scored against clusterings of the
measured expression.

Usage: 25_leiden_official_panel.py <stage>. Stages: official (HEST test folds), cohort (all
cohort samples), verify, reproducibility."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.cli import dispatch

STAGES = {
    "official": (
        "epathst.evaluation.leiden",
        "cli",
        "cluster all official test samples; save predictions and models",
    ),
    "cohort": (
        "epathst.evaluation.leiden_cohort",
        "cli",
        "cluster every cohort sample under the cohort CV",
    ),
    "verify": (
        "epathst.audits.leiden_ari",
        "run_dir",
        "re-derive ARI, labels and targets of a finished run",
    ),
    "reproducibility": (
        "epathst.audits.leiden_reproducibility",
        "run_dir",
        "re-run the clustering to measure label stability",
    ),
}

if __name__ == "__main__":
    dispatch("25_leiden_official_panel.py", __doc__.splitlines()[0], STAGES)
