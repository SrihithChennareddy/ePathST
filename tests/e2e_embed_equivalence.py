"""GPU script (not a pytest test) checking that the pipelined, GPU-filtered slide embedding
(epathst.tcga.worker.embed_slide) reproduces the original sequential pipeline.

Checks: (a) GPU tissue fractions give the same keep decisions as CPU `patch_tissue_fraction`;
(b) kept spots and embeddings match the cache written earlier by the sequential code for the
same slide.

Usage: python tests/e2e_embed_equivalence.py CASE THREADS
"""

import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from epathst.data.wsi_patches import patch_tissue_fraction
from epathst.models.student import load_locked_encoder
from epathst.models.teacher import HOptimusTeacher
from epathst.utils import load_config, resolve_run_paths

from epathst.tcga import hexgrid as s40, prediction as s38, worker as s43

case, T = sys.argv[1], int(sys.argv[2])
cfg = load_config(str(root / "configs/base.yaml"))
paths = resolve_run_paths(cfg)
dev = "cuda"
slides = pd.concat(
    [
        pd.read_csv(s38.TCGA_ROOT / "metadata" / f"{p}_slides.csv").assign(project=p)
        for p in s38.PANEL_OF
    ]
)
slides["case"] = slides.file_name.str[:12]
row = slides[slides.case == case].iloc[0]
student = load_locked_encoder(
    torch.load(paths["student_checkpoints"] / "locked_encoder_seed0.pt", map_location=dev)[
        "encoder_state_dict"
    ],
    dev,
)
teacher = HOptimusTeacher(cfg, dev)
pool = ThreadPoolExecutor(max_workers=T)
cen_all, mpp, mm2, nl = s40.layout(row.local_path)

# (a) 3000 patches spread evenly over the slide
sel = cen_all[np.linspace(0, len(cen_all) - 1, 3000).astype(int)]
patches = np.stack(list(pool.map(s40._read, [(row.local_path, x / mpp, y / mpp) for x, y in sel])))
cpu = np.array([patch_tissue_fraction(p) for p in patches])
gpu = s43.gpu_tissue_fraction(patches, dev)
print(
    f"(a) tissue fraction CPU vs GPU: max abs diff {np.abs(cpu - gpu).max():.2e}; keep-decisions "
    f"at 0.5 identical: {bool(((cpu >= .5) == (gpu >= .5)).all())} ({int((cpu >= .5).sum())}/3000 "
    "kept)"
)
assert ((cpu >= 0.5) == (gpu >= 0.5)).all()

# (b) student cache is stored as float16, so compare after the same rounding
ctx = {
    "device": dev,
    "pool": pool,
    "student": student,
    "teacher": teacher,
    "bs": cfg["teacher"]["batch_size"],
}
t0 = time.time()
cen, x_s, x_t = s43.embed_slide(row.local_path, cen_all, mpp, ctx)
dt = time.time() - t0
z = np.load(s40.EMB_ROOT / row.project / f"{case}.npz")
print(
    f"(b) {len(cen)} spots in {dt:.0f}s ({len(cen)/dt:.0f} patches/s, {T} read threads, GPU "
    f"shared with a running worker); cached spots {len(z['xy_um'])}"
)
assert np.array_equal(cen, z["xy_um"]), "kept spot set/order differs from cache"
ds = np.abs(x_s.astype(np.float16).astype(np.float32) - z["student"].astype(np.float32)).max()
dt_ = np.abs(x_t - z["teacher"].astype(np.float32)).max()
print(
    f"    kept spots identical: True | student max abs diff vs cache {ds:.2e} | teacher max abs "
    f"diff vs cache {dt_:.2e}"
)
assert ds < 5e-3 and dt_ < 5e-3
print("EMBEDDING EQUIVALENCE TEST PASSED")
