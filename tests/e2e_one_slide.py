"""End-to-end check of the TCGA hex-grid pipeline on single slides against the stored production
outputs. Needs one GPU, the TCGA slides, the cached embeddings and the saved model bundles (see
docs/reports/RESULTS_INDEX.md).

    python tests/e2e_one_slide.py TCGA-BRCA:TCGA-A8-A06O TCGA-COAD:TCGA-AA-3846
    TCGA-KIRC:TCGA-A3-3385

Per slide: (a) embeddings recomputed from the WSI match the cached ones, (b) predictions from the
cached embeddings match the stored predictions, (c) predictions from the recomputed embeddings match
the stored predictions.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from epathst.models.student import load_locked_encoder  # noqa: E402
from epathst.models.teacher import HOptimusTeacher  # noqa: E402
from epathst.utils import load_config, resolve_run_paths  # noqa: E402


def load_pipeline():
    """Returns the TCGA pipeline modules (worker, hexgrid, prediction)."""
    from epathst.tcga import hexgrid, prediction, worker

    return worker, hexgrid, prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slides", nargs="+", help="PROJECT:CASE, e.g. TCGA-COAD:TCGA-AA-3846")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--tag", default="hex100_leiden_official_hvg50")
    args = parser.parse_args()

    s43, s40, s38 = load_pipeline()
    cfg = load_config(str(ROOT / "configs/base.yaml"))
    paths = resolve_run_paths(cfg)
    device = "cuda"
    checkpoint = torch.load(
        paths["student_checkpoints"] / "locked_encoder_seed0.pt", map_location=device
    )
    student = load_locked_encoder(checkpoint["encoder_state_dict"], device)
    teacher = HOptimusTeacher(cfg, device)
    pool = ThreadPoolExecutor(max_workers=args.threads)
    ctx = {
        "device": device,
        "pool": pool,
        "student": student,
        "teacher": teacher,
        "bs": cfg["teacher"]["batch_size"],
    }
    bundles = {
        c: s38.load_bundles(Path(s38.DEFAULT_MODELS), c, device)
        for c in sorted(set(s38.PANEL_OF.values()))
    }
    queue_items = {(p, c): path for p, c, path in s43.slide_list()}

    failures = []
    for spec in args.slides:
        project, case = spec.split(":")
        path = queue_items[(project, case)]
        cached = np.load(s40.EMB_ROOT / project / f"{case}.npz")
        reference = np.load(s38.PERSIST / "predictions" / args.tag / project / f"{case}.npz")

        cen_all, mpp, _, _ = s40.layout(path)
        t0 = time.time()
        cen, x_s, x_t = s43.embed_slide(path, cen_all, mpp, ctx)
        dt = time.time() - t0
        same_spots = np.array_equal(cen, cached["xy_um"])
        d_s = np.abs(
            x_s.astype(np.float16).astype(np.float32) - cached["student"].astype(np.float32)
        ).max()
        d_t = np.abs(
            x_t.astype(np.float16).astype(np.float32) - cached["teacher"].astype(np.float32)
        ).max()
        print(
            f"{case}: (a) {len(cen)} spots in {dt:.0f}s, same spots={same_spots}, "
            f"max|student diff|={d_s:.2e}, max|teacher diff|={d_t:.2e}"
        )

        panel = s38.PANEL_OF[project]
        for label, xs, xt in (
            ("b", cached["student"].astype(np.float32), cached["teacher"].astype(np.float32)),
            ("c", x_s, x_t),
        ):
            preds = s38.predict_from_embeddings(
                bundles[panel], xs, xt, cached["xy_um"], case, cfg, device, 0
            )
            worst = max(
                np.abs(preds[s] - reference[f"pred__{s}"].astype(np.float32)).max()
                for s in s38.SYSTEMS
            )
            print(f"{case}: ({label}) predictions vs stored: max abs diff {worst:.2e}")
            if worst > 5e-3:
                failures.append((case, label, worst))
        if not same_spots or d_s > 5e-3 or d_t > 5e-3:
            failures.append((case, "a", max(d_s, d_t)))
    print("PASS" if not failures else f"FAIL {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
