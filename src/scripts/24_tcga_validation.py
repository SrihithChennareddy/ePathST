"""Runs one stage of the TCGA external validation of the frozen ePathST models.
Entry point for the epathst.tcga library; it never retrains, only predicts on TCGA slides.

Usage: 24_tcga_validation.py <stage>. Stages run in order: cases, download, patches,
predict-sparse or predict (or worker for the self-scheduling queue), audit, companion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.cli import dispatch

STAGES = {
    "cases": (
        "epathst.tcga.cases",
        "cli",
        "select TCGA cases with diagnostic slides and the matching panel",
    ),
    "download": ("epathst.tcga.download", "cli", "download the selected diagnostic slides"),
    "patches": (
        "epathst.tcga.patches",
        "cli",
        "extract thinned-grid tissue patches from the slides",
    ),
    "predict-sparse": (
        "epathst.tcga.prediction",
        "cli",
        "predict the official panel on the thinned-grid patches",
    ),
    "predict": (
        "epathst.tcga.hexgrid",
        "cli",
        "predict on a 100 um hexagonal pseudo-spot lattice (sharded)",
    ),
    "worker": (
        "epathst.tcga.worker",
        "cli",
        "self-scheduling GPU worker over the whole slide queue",
    ),
    "audit": ("epathst.tcga.audit", "cli", "progress and completeness audit of a prediction run"),
    "companion": (
        "epathst.tcga.companion",
        "cli",
        "download RNA-seq, clinical and purity companion data",
    ),
}

if __name__ == "__main__":
    dispatch("24_tcga_validation.py", __doc__.splitlines()[0], STAGES)
