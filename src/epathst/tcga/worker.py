"""Self-scheduling TCGA worker: one GPU, many slides, any number of workers at once.
Runs as `src/scripts/24_tcga_validation.py worker`; same per-slide pipeline as `hexgrid.py`.

Every worker loops over the same deterministic slide list (shuffled so all three cancers progress
together), claims ONE unfinished slide with an atomic `mkdir` lock, runs the hex-grid pipeline (100
um hex pseudo-spots, frozen student + H-Optimus-0 embeddings, fold-averaged ePathST-E / ePathST /
H-Optimus-0 predictions of the cancer's official 50 genes), writes the result, releases the lock
and takes the next slide. Nothing is statically sharded, so workers can join or die at any time:
  * a slide is DONE when `qc/<project>/<case>.json` exists (status ok / skipped_*; FAILED slides are
    retried up to 2 times);
  * a lock whose heartbeat is older than 20 minutes belongs to a dead worker and is taken over;
  * outputs are written to a temp file and renamed, so a killed worker never leaves a half-written
    result;
  * when the SLURM time limit is near the worker stops claiming and exits with code 10 if
    unfinished slides remain (the SLURM wrapper then resubmits itself), or 0 when all are done.
Output: <PERSIST>/predictions/<tag>/<project>/<case>.npz and <tag>/qc/<project>/<case>.json (same
npz layout as `hexgrid.py`).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from epathst.data.feature_cache import extract_student_features
from epathst.models.student import load_locked_encoder
from epathst.models.teacher import HOptimusTeacher
from epathst.tcga import hexgrid, prediction
from epathst.utils import load_config, resolve_run_paths

STALE_S, MAX_ATTEMPTS = 20 * 60, 2
WORKER = f"{os.environ.get('SLURM_JOB_ID', 'local')}-{uuid.uuid4().hex[:6]}"


def log(msg):
    print(f"[worker {WORKER} {datetime.now():%H:%M:%S}] {msg}", flush=True)


def job_deadline(fallback_hours: float) -> float:
    jid = os.environ.get("SLURM_JOB_ID")
    if jid:
        out = subprocess.run(
            ["scontrol", "show", "job", "-o", jid], capture_output=True, text=True
        ).stdout
        m = re.search(r"EndTime=(\S+)", out)
        if m and m[1] not in ("Unknown", "None"):
            return datetime.fromisoformat(m[1]).timestamp()
    return time.time() + fallback_hours * 3600


def slide_list() -> list[tuple[str, str, str]]:
    items = []
    for project in prediction.PANEL_OF:
        s = pd.read_csv(prediction.TCGA_ROOT / "metadata" / f"{project}_slides.csv")
        s = s[s["status"] == "downloaded"]
        items += [(project, fn[:12], lp) for fn, lp in zip(s["file_name"], s["local_path"])]
    items.sort()
    random.Random(0).shuffle(items)  # same order for every worker; cancers interleaved
    return items


class Queue:
    def __init__(self, root: Path):
        self.root = root

    def qc_file(self, project, case) -> Path:
        return self.root / "qc" / project / f"{case}.json"

    def is_done(self, project, case) -> bool:
        f = self.qc_file(project, case)
        if not f.exists():
            return False
        try:
            q = json.loads(f.read_text())
        except Exception:
            return False
        return not (
            str(q.get("status", "")).startswith("FAILED") and q.get("attempts", 1) < MAX_ATTEMPTS
        )

    def attempts(self, project, case) -> int:
        f = self.qc_file(project, case)
        try:
            return int(json.loads(f.read_text()).get("attempts", 0)) if f.exists() else 0
        except Exception:
            return 0

    def try_claim(self, project, case):
        d = self.root / "claims" / project / case
        d.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(d)
        except FileExistsError:
            hb = d / "heartbeat"
            try:
                age = time.time() - (hb.stat().st_mtime if hb.exists() else d.stat().st_mtime)
            except FileNotFoundError:
                return None
            if age < STALE_S:
                return None
            try:  # dead worker: exactly one contender wins the rename
                os.rename(d, d.with_name(f"{case}.stale.{uuid.uuid4().hex[:6]}"))
                os.mkdir(d)
            except OSError:
                return None
        (d / "owner").write_text(WORKER)
        (d / "heartbeat").touch()
        return d

    def release(self, d: Path):
        for f in ("heartbeat", "owner"):
            try:
                (d / f).unlink()
            except FileNotFoundError:
                pass
        try:
            d.rmdir()
        except OSError:
            pass


class Heartbeat(threading.Thread):
    def __init__(self, d: Path):
        super().__init__(daemon=True)
        self.d, self.stop_evt = d, threading.Event()

    def run(self):
        while not self.stop_evt.wait(60):
            try:
                (self.d / "heartbeat").touch()
            except OSError:
                pass


def atomic_savez(path: Path, compressed: bool, **arrays):
    tmp = path.with_name(f".tmp.{uuid.uuid4().hex[:8]}.{path.name}")
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


def gpu_tissue_fraction(patches_u8: np.ndarray, device: str) -> np.ndarray:
    """GPU port of `patch_tissue_fraction` (saturation > 15 and value < 235, same float32
    arithmetic) for a batch of uint8 (B, H, W, 3) patches; decisions match the CPU version (checked
    in tests/test_tcga_embed_equivalence.py).
    """
    a = torch.from_numpy(patches_u8).to(device).float() / 255.0
    maxc, minc = a.amax(-1), a.amin(-1)
    sat = (
        torch.where(maxc > 0, (maxc - minc) / maxc.clamp(min=1e-6), torch.zeros_like(maxc)) * 255.0
    )
    val = maxc * 255.0
    return ((sat > 15) & (val < 235)).float().mean(dim=(1, 2)).cpu().numpy()


def embed_slide(path, cen_all, mpp, ctx):
    """Reads the hex-lattice patches in chunks on a background thread (slide I/O + resize on the
    thread pool) while the main thread filters (>=50% tissue, on GPU) and embeds the previous chunk
    with the frozen student and H-Optimus-0, so the GPU no longer idles during reads. Returns (kept
    centres, student embeddings, teacher embeddings).
    """
    device, pool, student, teacher, bs = (
        ctx["device"],
        ctx["pool"],
        ctx["student"],
        ctx["teacher"],
        ctx["bs"],
    )
    q, stop = queue.Queue(maxsize=2), threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=1)
                return
            except queue.Full:
                pass

    def producer():
        try:
            for s0 in range(0, len(cen_all), hexgrid.CHUNK):
                if stop.is_set():
                    return
                c = cen_all[s0 : s0 + hexgrid.CHUNK]
                put(
                    (
                        c,
                        np.stack(
                            list(pool.map(hexgrid._read, [(path, x / mpp, y / mpp) for x, y in c]))
                        ),
                    )
                )
            put(None)
        except Exception as e:  # surfaced in the consumer
            put(e)

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    kept, xs_l, xt_l = [], [], []
    try:
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            c, patches = item
            ok = gpu_tissue_fraction(patches, device) >= hexgrid.MIN_TISSUE
            if not ok.any():
                continue
            patches = patches[ok]
            kept.append(c[ok])
            xs_l.append(extract_student_features(student, patches, device))
            xt_l.append(
                np.concatenate(
                    [teacher.embed_batch(patches[j : j + bs]) for j in range(0, len(patches), bs)]
                )
            )
    finally:
        stop.set()
        th.join(timeout=30)
    if not kept:
        return None, None, None
    return np.concatenate(kept), np.concatenate(xs_l), np.concatenate(xt_l)


def process_slide(project, case, path, ctx) -> dict:
    cfg, device, pool, student, teacher, bs, bundles, out_root = (
        ctx["cfg"],
        ctx["device"],
        ctx["pool"],
        ctx["student"],
        ctx["teacher"],
        ctx["bs"],
        ctx["bundles"],
        ctx["out_root"],
    )
    cancer = prediction.PANEL_OF[project]
    t0 = time.time()
    emb_path = hexgrid.EMB_ROOT / project / f"{case}.npz"
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    if emb_path.exists():
        z = np.load(emb_path)
        cen, x_s, x_t = z["xy_um"], z["student"].astype(np.float32), z["teacher"].astype(np.float32)
        mpp, tissue_mm2, n_lattice = (
            float(z["native_mpp"]),
            float(z["tissue_mm2"]),
            int(z["n_lattice"]),
        )
    else:
        cen_all, mpp, tissue_mm2, n_lattice = hexgrid.layout(path)
        if mpp is None:
            return {"status": "skipped_no_mpp"}
        if len(cen_all) < ctx["min_spots"]:
            return {
                "status": f"skipped_<{ctx['min_spots']}_spots_prefilter",
                "n_spots": int(len(cen_all)),
                "tissue_mm2": tissue_mm2,
            }
        cen, x_s, x_t = embed_slide(path, cen_all, mpp, ctx)
        if cen is None:
            return {"status": "skipped_no_tissue_patches"}
        if len(cen) >= ctx["min_spots"]:
            atomic_savez(
                emb_path,
                False,
                xy_um=cen,
                student=x_s.astype(np.float16),
                teacher=x_t.astype(np.float16),
                native_mpp=mpp,
                tissue_mm2=tissue_mm2,
                n_lattice=n_lattice,
            )
    if len(cen) < ctx["min_spots"]:
        return {
            "status": f"skipped_<{ctx['min_spots']}_spots",
            "n_spots": int(len(cen)),
            "tissue_mm2": tissue_mm2,
        }
    t1 = time.time()
    b = bundles[cancer]
    preds = prediction.predict_from_embeddings(b, x_s, x_t, cen, case, cfg, device, ctx["seed"])
    (out_root / project).mkdir(parents=True, exist_ok=True)
    atomic_savez(
        out_root / project / f"{case}.npz",
        True,
        x_um=cen[:, 0],
        y_um=cen[:, 1],
        genes=np.array(b[0]["meta"]["genes"]),
        n_folds=len(b),
        native_mpp=mpp,
        pitch_um=hexgrid.PITCH_UM,
        **{f"pred__{s}": p.astype(np.float16) for s, p in preds.items()},
    )
    return {
        "status": "ok",
        "n_spots": int(len(cen)),
        "n_lattice": int(n_lattice),
        "native_mpp": mpp,
        "tissue_mm2": tissue_mm2,
        "seconds_total": round(time.time() - t0, 1),
        "seconds_predict": round(time.time() - t1, 1),
        **prediction.slide_qc(cen, cfg),
    }


def run_final_audit(root: Path, tag: str):
    """The first worker to see an empty queue runs the full audit once (lock = audit.lock); result
    in AUDIT.json / AUDIT.log.
    """
    try:
        os.mkdir(root / "audit.lock")
    except FileExistsError:
        return
    log("queue empty: running the final audit (24_tcga_validation.py audit --full-check)")
    with open(root / "AUDIT.log", "w") as f:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[2] / "scripts" / "24_tcga_validation.py"),
                "audit",
                "--tag",
                tag,
                "--full-check",
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--models-dir", default=prediction.DEFAULT_MODELS)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--min-spots", type=int, default=200)
    ap.add_argument("--io-threads", type=int, default=12)
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--limit", type=int, default=None, help="smoke test: stop after this many slides"
    )
    ap.add_argument("--fallback-hours", type=float, default=7.5)
    ap.add_argument(
        "--margin-min",
        type=float,
        default=15.0,
        help="stop claiming this many minutes before the time limit",
    )
    ap.add_argument(
        "--check-queue", action="store_true", help="print the number of unfinished slides and exit"
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    device = args.device if torch.cuda.is_available() else "cpu"
    models_dir = Path(args.models_dir)
    tag = args.tag or f"hex100_{models_dir.parent.name}"
    out_root = prediction.PERSIST / "predictions" / tag
    q = Queue(out_root)
    items = slide_list()
    n_unfinished = sum(not q.is_done(p, c) for p, c, _ in items)
    if args.check_queue:
        print(n_unfinished)
        sys.exit(0)
    if n_unfinished == 0:
        log("queue already empty")
        run_final_audit(out_root, tag)
        sys.exit(0)
    deadline = job_deadline(args.fallback_hours)
    log(f"device={device} tag={tag} deadline in {(deadline - time.time()) / 3600:.1f} h")

    ckpt = torch.load(
        paths["student_checkpoints"] / f"locked_encoder_seed{args.feature_seed}.pt",
        map_location=device,
    )
    student = load_locked_encoder(ckpt["encoder_state_dict"], device)
    teacher = HOptimusTeacher(cfg, device)
    bundles = {
        c: prediction.load_bundles(models_dir, c, device)
        for c in sorted(set(prediction.PANEL_OF.values()))
    }
    ctx = {
        "cfg": cfg,
        "device": device,
        "pool": ThreadPoolExecutor(max_workers=args.io_threads),
        "student": student,
        "teacher": teacher,
        "bs": cfg["teacher"]["batch_size"],
        "bundles": bundles,
        "out_root": out_root,
        "min_spots": args.min_spots,
        "seed": args.seed,
    }
    log(f"{len(items)} slides in the queue, {n_unfinished} unfinished")
    max_seen, n_done = 0.0, 0

    while True:
        unfinished = [it for it in items if not q.is_done(it[0], it[1])]
        if not unfinished:
            log("queue empty: every slide has a result")
            run_final_audit(out_root, tag)
            sys.exit(0)
        if args.limit is not None and n_done >= args.limit:
            log(f"--limit {args.limit} reached")
            sys.exit(0)
        margin = max(args.margin_min * 60.0, 2.5 * max_seen)
        if time.time() + margin > deadline:
            log(f"time limit near; {len(unfinished)} slides unfinished, exiting for resubmission")
            sys.exit(10)
        claim = None
        for project, case, path in unfinished:
            if q.attempts(project, case) >= MAX_ATTEMPTS and not q.is_done(project, case):
                continue
            claim = q.try_claim(project, case)
            if claim is not None:
                break
        if claim is None:
            log(f"nothing claimable ({len(unfinished)} unfinished, held by other workers); waiting")
            time.sleep(60)
            continue
        if q.is_done(project, case):  # finished by someone else between listing and claiming
            q.release(claim)
            continue
        hb = Heartbeat(claim)
        hb.start()
        t0 = time.time()
        qc = {
            "case": case,
            "project": project,
            "worker": WORKER,
            "attempts": q.attempts(project, case) + 1,
        }
        try:
            qc.update(process_slide(project, case, path, ctx))
        except Exception as e:
            qc["status"] = f"FAILED {type(e).__name__}: {e}"[:300]
            log(f"{project} {case}: {qc['status']}")
            torch.cuda.empty_cache()
        finally:
            hb.stop_evt.set()
        qc["finished"] = datetime.now().isoformat(timespec="seconds")
        qf = q.qc_file(project, case)
        qf.parent.mkdir(parents=True, exist_ok=True)
        tmp = qf.with_name(f".tmp.{uuid.uuid4().hex[:8]}.json")
        tmp.write_text(json.dumps(qc))
        os.replace(tmp, qf)
        q.release(claim)
        max_seen = max(max_seen, time.time() - t0)
        n_done += 1
        log(
            f"{project} {case}: {qc['status']} "
            + (f"{qc['n_spots']} spots, {qc['seconds_total']}s" if qc["status"] == "ok" else "")
            + f" | {len(unfinished) - 1} left"
        )
