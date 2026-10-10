#!/usr/bin/env python
"""Times each stage of the TCGA hex-grid pipeline on N evenly spaced lattice points of one slide.
Stages, reported per 1000 patches: slide read (thread pool), tissue filter, student embed, teacher
embed. Used to size the GPU worker of `24_tcga_validation.py worker`; writes nothing.

Usage: profile_hex_slide.py CASE [N] [THREADS]
"""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from epathst.data.feature_cache import extract_student_features
from epathst.data.wsi_patches import patch_tissue_fraction
from epathst.models.student import load_locked_encoder
from epathst.models.teacher import HOptimusTeacher
from epathst.utils import load_config, resolve_run_paths

from epathst.tcga import hexgrid as s40

s38 = s40.prediction

case, N, T = (
    sys.argv[1],
    int(sys.argv[2]) if len(sys.argv) > 2 else 2048,
    int(sys.argv[3]) if len(sys.argv) > 3 else 12,
)
cfg = load_config("configs/base.yaml")
paths = resolve_run_paths(cfg)
dev = "cuda"
slides = pd.concat(
    [pd.read_csv(s38.TCGA_ROOT / "metadata" / f"{p}_slides.csv") for p in s38.PANEL_OF]
)
slides["case"] = slides.file_name.str[:12]
path = slides[slides.case == case].local_path.iloc[0]
student = load_locked_encoder(
    torch.load(paths["student_checkpoints"] / "locked_encoder_seed0.pt", map_location=dev)[
        "encoder_state_dict"
    ],
    dev,
)
teacher = HOptimusTeacher(cfg, dev)
pool = ThreadPoolExecutor(max_workers=T)
cen, mpp, mm2, nl = s40.layout(path)
cen = cen[np.linspace(0, len(cen) - 1, N).astype(int)]
sync = torch.cuda.synchronize


def t(f):
    sync()
    a = time.time()
    r = f()
    sync()
    return r, time.time() - a


patches, t_read = t(
    lambda: np.stack(list(pool.map(s40._read, [(path, x / mpp, y / mpp) for x, y in cen])))
)
_, t_filter = t(lambda: [patch_tissue_fraction(p) for p in patches])
_, t_stu = t(lambda: extract_student_features(student, patches, dev))
bs = cfg["teacher"]["batch_size"]
_, t_tea = t(
    lambda: np.concatenate(
        [teacher.embed_batch(patches[j : j + bs]) for j in range(0, len(patches), bs)]
    )
)
k = 1000.0 / N
print(
    f"{case}: {N} patches, {T} read threads, teacher batch {bs}\n per 1000 patches (s): read {t_read*k:.1f} | tissue filter {t_filter*k:.1f} | student {t_stu*k:.1f} | teacher {t_tea*k:.1f}"
    f"\n => sequential total {(t_read+t_filter+t_stu+t_tea)*k:.1f} s/1000 ({1000/((t_read+t_filter+t_stu+t_tea)*k):.0f} patches/s)"
)
