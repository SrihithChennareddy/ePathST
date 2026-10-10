"""Locations shared by the figure scripts. Outputs go to figures/output unless EPATHST_FIG_OUT is
set; HEST-1k data roots can be overridden too.
"""

import os
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent
DATA_DIR = FIG_DIR / "data"
FIG_OUT = Path(os.environ.get("EPATHST_FIG_OUT", FIG_DIR / "output"))
HEST_ROOT = Path(os.environ.get("EPATHST_HEST_ROOT", "/lustre/nvwulf/scratch/schennareddy/HEST-1k"))
BENCH_ROOT = HEST_ROOT / "hest_bench"
WSI_ROOT = HEST_ROOT / "data"
FIG_OUT.mkdir(parents=True, exist_ok=True)
