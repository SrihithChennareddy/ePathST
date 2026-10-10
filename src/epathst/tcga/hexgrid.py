"""Hex-lattice pseudo-spot layout and frozen-model prediction of the official 50-gene panel on TCGA
diagnostic slides (ePathST-E / ePathST / H-Optimus-0). Nothing is trained.

Single-process, shardable version of the pipeline that `src/scripts/24_tcga_validation.py worker`
runs with a claim queue. The lattice mimics 10x Visium geometry so TCGA predictions are comparable.

Spot layout (hexagonal, 100 um pitch d; area per spot (sqrt(3)/2)*d^2 = 8,660 um^2, i.e. ~115
spots / mm^2 of tissue):
  * thumbnail (<=2048 px) -> tissue mask by Otsu threshold on the saturation channel;
  * hex lattice in microns (row spacing d*sqrt(3)/2, odd rows shifted d/2), kept where the 112 um
    field around the centre is >= 50% tissue in the mask; centres closer than 56 um to the slide
    edge are dropped;
  * each kept centre is read at 0.5 um/px (224 px = 112 um field, same crop function as the
    training patches) and kept if the patch itself is >= 50% tissue (`patch_tissue_fraction`);
  * slides with < 200 spots or without native mpp metadata are skipped.

Patches are streamed (read, embed, discard). Embeddings (frozen student 320-d, H-Optimus-0 1536-d)
are cached in scratch so the model set can be swapped without re-reading pixels. Per-slide
predictions are saved as float16 (log1p scale, fold-averaged) with micron coordinates. Resumable
and shardable for SLURM arrays.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter

from epathst.data.feature_cache import extract_student_features
from epathst.data.wsi_patches import open_slide, patch_tissue_fraction
from epathst.models.student import load_locked_encoder
from epathst.models.teacher import HOptimusTeacher
from epathst.tcga import prediction
from epathst.utils import load_config, resolve_run_paths

PITCH_UM, FIELD_UM, PATCH_PX = 100.0, 112.0, 224
MIN_TISSUE, THUMB_MAX, CHUNK = 0.5, 2048, 1024
EMB_ROOT = prediction.TCGA_ROOT / "embeddings_hex100"


def otsu_threshold(u8: np.ndarray) -> int:
    hist = np.bincount(u8.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    return int(np.nanargmax(between))


def tissue_mask_otsu(thumb_rgb: np.ndarray) -> np.ndarray:
    arr = thumb_rgb.astype(np.float32) / 255.0
    maxc, minc = arr.max(-1), arr.min(-1)
    sat = np.where(maxc > 0, (maxc - minc) / np.clip(maxc, 1e-6, None), 0.0)
    sat_u8 = np.clip(sat * 255.0, 0, 255).astype(np.uint8)
    return sat_u8 > otsu_threshold(sat_u8)


def hex_centres_um(w_um: float, h_um: float, pitch: float, margin: float) -> np.ndarray:
    dy = pitch * np.sqrt(3) / 2.0
    rows = []
    for j, y in enumerate(np.arange(margin, h_um - margin, dy)):
        xs = np.arange(margin + (pitch / 2.0 if j % 2 else 0.0), w_um - margin, pitch)
        rows.append(np.stack([xs, np.full_like(xs, y)], axis=1))
    return np.concatenate(rows) if rows else np.zeros((0, 2))


_tl = threading.local()


def _read(args):
    path, cx, cy = args
    if not hasattr(_tl, "reader") or _tl.path != path:
        _tl.reader, _tl.path = open_slide(path), path
    return _tl.reader.read_region_at_mpp(cx, cy, FIELD_UM, PATCH_PX)


def layout(path: str):
    """Returns (centres_um (n,2) after mask prefilter, native_mpp, tissue_mm2, n_lattice)."""
    r = open_slide(path)
    w0, h0 = r.dimensions
    mpp = r.mpp
    if mpp is None:
        return None, None, None, 0
    thumb = r.get_thumbnail(THUMB_MAX)
    if hasattr(r, "close"):
        r.close()
    mask = tissue_mask_otsu(thumb)
    th, tw = mask.shape
    w_um, h_um = w0 * mpp, h0 * mpp
    um_per_tx, um_per_ty = w_um / tw, h_um / th
    tissue_mm2 = float(mask.sum() * um_per_tx * um_per_ty / 1e6)
    frac = uniform_filter(mask.astype(np.float32), size=max(1, int(round(FIELD_UM / um_per_tx))))
    cen = hex_centres_um(w_um, h_um, PITCH_UM, FIELD_UM / 2.0)
    tx = np.clip((cen[:, 0] / um_per_tx).astype(int), 0, tw - 1)
    ty = np.clip((cen[:, 1] / um_per_ty).astype(int), 0, th - 1)
    return cen[frac[ty, tx] >= MIN_TISSUE], float(mpp), tissue_mm2, len(cen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--projects", nargs="+", default=list(prediction.PANEL_OF))
    ap.add_argument("--models-dir", default=prediction.DEFAULT_MODELS)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-slides", type=int, default=None)
    ap.add_argument("--min-spots", type=int, default=200)
    ap.add_argument("--io-threads", type=int, default=12)
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    device = args.device if torch.cuda.is_available() else "cpu"
    models_dir = Path(args.models_dir)
    tag = args.tag or f"hex100_{models_dir.parent.name}"
    out_root = prediction.PERSIST / "predictions" / tag
    print(f"[hexgrid] device={device} models={models_dir} -> {out_root}", flush=True)

    ckpt = torch.load(
        paths["student_checkpoints"] / f"locked_encoder_seed{args.feature_seed}.pt",
        map_location=device,
    )
    student = load_locked_encoder(ckpt["encoder_state_dict"], device)
    teacher = HOptimusTeacher(cfg, device)
    bs = cfg["teacher"]["batch_size"]
    pool = ThreadPoolExecutor(max_workers=args.io_threads)

    for project in args.projects:
        cancer = prediction.PANEL_OF[project]
        bundles = prediction.load_bundles(models_dir, cancer, device)
        genes = bundles[0]["meta"]["genes"]
        slides = pd.read_csv(prediction.TCGA_ROOT / "metadata" / f"{project}_slides.csv")
        slides = slides[slides["status"] == "downloaded"].copy()
        slides["case"] = slides["file_name"].str[:12]
        cases = sorted(zip(slides["case"], slides["local_path"]))
        if args.max_slides:
            cases = cases[: args.max_slides]
        cases = cases[args.shard :: args.n_shards]
        (out_root / project).mkdir(parents=True, exist_ok=True)
        (EMB_ROOT / project).mkdir(parents=True, exist_ok=True)
        qc_path = out_root / f"qc_{project}_shard{args.shard}.csv"
        qc_rows = pd.read_csv(qc_path).to_dict("records") if qc_path.exists() else []
        done = {r["case"] for r in qc_rows}
        print(
            f"[hexgrid] {project}: {len(cases)} slides in shard, {len(bundles)} fold bundles ({cancer} "
            "panel)",
            flush=True,
        )

        for i, (case, path) in enumerate(cases):
            out_path = out_root / project / f"{case}.npz"
            if out_path.exists() and case in done:
                continue
            t0 = time.time()
            try:
                emb_path = EMB_ROOT / project / f"{case}.npz"
                if emb_path.exists():
                    z = np.load(emb_path)
                    cen, x_s, x_t = (
                        z["xy_um"],
                        z["student"].astype(np.float32),
                        z["teacher"].astype(np.float32),
                    )
                    mpp, tissue_mm2, n_lattice = (
                        float(z["native_mpp"]),
                        float(z["tissue_mm2"]),
                        int(z["n_lattice"]),
                    )
                else:
                    cen_all, mpp, tissue_mm2, n_lattice = layout(path)
                    if mpp is None:
                        qc_rows.append(
                            {"case": case, "project": project, "status": "skipped_no_mpp"}
                        )
                        continue
                    if len(cen_all) < args.min_spots:
                        qc_rows.append(
                            {
                                "case": case,
                                "project": project,
                                "status": f"skipped_<{args.min_spots}_spots_prefilter",
                                "n_spots": len(cen_all),
                                "tissue_mm2": tissue_mm2,
                            }
                        )
                        continue
                    kept, xs_l, xt_l = [], [], []
                    for s0 in range(0, len(cen_all), CHUNK):
                        c = cen_all[s0 : s0 + CHUNK]
                        patches = np.stack(
                            list(pool.map(_read, [(path, x / mpp, y / mpp) for x, y in c]))
                        )
                        ok = np.array([patch_tissue_fraction(p) >= MIN_TISSUE for p in patches])
                        if not ok.any():
                            continue
                        patches = patches[ok]
                        kept.append(c[ok])
                        xs_l.append(extract_student_features(student, patches, device))
                        xt_l.append(
                            np.concatenate(
                                [
                                    teacher.embed_batch(patches[j : j + bs])
                                    for j in range(0, len(patches), bs)
                                ]
                            )
                        )
                    if not kept:
                        qc_rows.append(
                            {
                                "case": case,
                                "project": project,
                                "status": "skipped_no_tissue_patches",
                            }
                        )
                        continue
                    cen, x_s, x_t = np.concatenate(kept), np.concatenate(xs_l), np.concatenate(xt_l)
                    if len(cen) >= args.min_spots:
                        np.savez(
                            emb_path,
                            xy_um=cen,
                            student=x_s.astype(np.float16),
                            teacher=x_t.astype(np.float16),
                            native_mpp=mpp,
                            tissue_mm2=tissue_mm2,
                            n_lattice=n_lattice,
                        )
                if len(cen) < args.min_spots:
                    qc_rows.append(
                        {
                            "case": case,
                            "project": project,
                            "status": f"skipped_<{args.min_spots}_spots",
                            "n_spots": len(cen),
                            "tissue_mm2": tissue_mm2,
                        }
                    )
                    continue
                t1 = time.time()
                preds = prediction.predict_from_embeddings(
                    bundles, x_s, x_t, cen, case, cfg, device, args.seed
                )
                np.savez_compressed(
                    out_path,
                    x_um=cen[:, 0],
                    y_um=cen[:, 1],
                    genes=np.array(genes),
                    n_folds=len(bundles),
                    native_mpp=mpp,
                    pitch_um=PITCH_UM,
                    **{f"pred__{s}": p.astype(np.float16) for s, p in preds.items()},
                )
                qc_rows.append(
                    {
                        "case": case,
                        "project": project,
                        "status": "ok",
                        "n_spots": len(cen),
                        "n_lattice": n_lattice,
                        "native_mpp": mpp,
                        "tissue_mm2": tissue_mm2,
                        "seconds_total": round(time.time() - t0, 1),
                        "seconds_predict": round(time.time() - t1, 1),
                        **prediction.slide_qc(cen, cfg),
                    }
                )
                print(
                    f"[hexgrid] {project} {i + 1}/{len(cases)} {case}: {len(cen)} spots "
                    f"({tissue_mm2:.0f} mm2 tissue), "
                    f"median NN {qc_rows[-1]['median_nn_um']:.0f}um, "
                    f"{qc_rows[-1]['seconds_total']}s",
                    flush=True,
                )
            except Exception as e:
                qc_rows.append(
                    {
                        "case": case,
                        "project": project,
                        "status": f"FAILED {type(e).__name__}: {e}"[:200],
                    }
                )
                print(
                    f"[hexgrid] {project} {case}: FAILED {type(e).__name__}: {e}",
                    file=sys.stderr,
                    flush=True,
                )
            pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
        pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
    print("[hexgrid] done", flush=True)
