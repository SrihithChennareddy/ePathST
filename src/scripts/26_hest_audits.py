"""Runs one consistency audit of the HEST panels, the frozen model files or the per-fold samples.
Entry point for epathst.audits; checks that reported results rest on the intended data and weights.

Usage: 26_hest_audits.py <stage>. Stages: panel-coverage, frozen-weights, training-samples.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.cli import dispatch

STAGES = {
    "panel-coverage": (
        "epathst.audits.panel_coverage",
        "cli",
        "official-panel gene coverage of every cohort sample",
    ),
    "frozen-weights": (
        "epathst.audits.frozen_weights",
        "run_dir",
        "checksum manifest of a run's saved model bundles",
    ),
    "training-samples": (
        "epathst.audits.training_samples",
        "path",
        "train, validation and test samples of every fold",
    ),
}

if __name__ == "__main__":
    dispatch("26_hest_audits.py", __doc__.splitlines()[0], STAGES)
