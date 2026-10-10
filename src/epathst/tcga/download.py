"""Downloads TCGA-BRCA/KIRC/COAD FFPE diagnostic slides into scratch.
Run as `src/scripts/24_tcga_validation.py download`; slides are pixel data, so scratch only,
never YurovskyGroup (see the `tcga` section of configs/base.yaml).

One subdir per project, with a per-cancer download metadata CSV (file id/name/size/status) in
scratch, separate from the cohort manifests. A timing probe (default 10 files) measures real
throughput and prints an ETA before the rest downloads in parallel. Resumable: files already
present at their expected size are skipped."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from epathst.tcga.gdc import download_gdc_file
from epathst.utils import load_config


def human_gb(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.1f} GB"


def human_eta(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return "unknown"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def load_worklist(tcfg: dict, projects: list[str]) -> pd.DataFrame:
    persist_dir = Path(tcfg["paths"]["persist_dir"])
    scratch_dir = Path(tcfg["paths"]["scratch_dir"])
    manifests_dir = persist_dir / tcfg["paths"]["manifests"]

    rows = []
    for project in projects:
        df = pd.read_csv(manifests_dir / f"tcga_cohort_{project}.csv")
        slide_dir = scratch_dir / tcfg["paths"]["slides"] / project
        for _, r in df.iterrows():
            rows.append(
                {
                    "project": project,
                    "file_id": r["slide_file_id"],
                    "file_name": r["slide_file_name"],
                    "expected_size": int(r["slide_file_size"]),
                    "dest": slide_dir / r["slide_file_name"],
                }
            )
    return pd.DataFrame(rows)


def download_one(gdc_data_url: str, row: dict) -> dict:
    result = download_gdc_file(gdc_data_url, row["file_id"], row["dest"], row["expected_size"])
    result.update(
        {
            "project": row["project"],
            "file_id": row["file_id"],
            "file_name": row["file_name"],
            "expected_size": row["expected_size"],
            "local_path": str(row["dest"]),
        }
    )
    return result


def run_batch(gdc_data_url: str, rows: list[dict], workers: int, label: str) -> list[dict]:
    results = []
    t_start = time.time()
    bytes_done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, gdc_data_url, r): r for r in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            bytes_done += res["bytes"] if res["status"] == "downloaded" else 0
            if res["status"] == "failed":
                print(
                    f"[tcga.download] {label} FAILED {res['file_name']}: {res['error']}",
                    file=sys.stderr,
                )
            if i % 25 == 0 or i == len(rows):
                elapsed = time.time() - t_start
                thr = bytes_done / elapsed if elapsed > 0 and bytes_done > 0 else 0
                print(
                    f"[tcga.download] {label}: {i}/{len(rows)} files, "
                    f"{human_gb(bytes_done)} downloaded, {thr/1e6:.1f} MB/s"
                )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--projects", nargs="+", default=None, help="default: all of tcga.projects")
    ap.add_argument("--workers", type=int, default=6, help="parallel download streams")
    ap.add_argument(
        "--probe-n", type=int, default=10, help="files per project for the timing probe"
    )
    ap.add_argument(
        "--probe-only", action="store_true", help="run only the ETA probe, don't download the rest"
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["tcga"]
    projects = args.projects or tcfg["projects"]

    worklist = load_worklist(tcfg, projects)

    # drop finished files first so a resumed run doesn't skew the probe/ETA
    def already_done(row):
        p = Path(row["dest"])
        return p.exists() and p.stat().st_size == row["expected_size"]

    worklist["done"] = worklist.apply(already_done, axis=1)
    todo = worklist[~worklist["done"]].to_dict("records")
    n_done = int(worklist["done"].sum())
    total_bytes = int(worklist["expected_size"].sum())
    remaining_bytes = int(worklist.loc[~worklist["done"], "expected_size"].sum())

    print(
        f"[tcga.download] cohort: {len(worklist)} slides, {human_gb(total_bytes)} total "
        f"({', '.join(f'{p}={int((worklist.project==p).sum())}' for p in projects)})"
    )
    print(
        f"[tcga.download] already present: {n_done} files -- resuming the rest "
        f"({len(todo)} files, {human_gb(remaining_bytes)})"
    )
    if not todo:
        print("[tcga.download] nothing to do")
        return

    probe_rows = todo[: args.probe_n]
    print(
        f"[tcga.download] probing with {len(probe_rows)} files "
        f"({args.workers} parallel streams)..."
    )
    probe_results = run_batch(tcfg["gdc_data"], probe_rows, args.workers, "PROBE")
    probe_bytes = sum(r["bytes"] for r in probe_results if r["status"] == "downloaded")
    probe_time = max(
        (r["elapsed_s"] for r in probe_results if r["status"] == "downloaded"), default=0
    )
    # approximate wall-clock: per-file times overlap across parallel workers
    probe_wall = max(1e-6, sum(r["elapsed_s"] for r in probe_results) / max(1, args.workers))
    throughput = probe_bytes / probe_wall if probe_wall > 0 else 0
    remaining_after_probe = remaining_bytes - probe_bytes
    eta_s = remaining_after_probe / throughput if throughput > 0 else float("inf")
    print(
        f"[tcga.download] probe: {human_gb(probe_bytes)} in ~{probe_wall:.0f}s "
        f"-> ~{throughput/1e6:.1f} MB/s aggregate across {args.workers} streams"
    )
    print(
        f"[tcga.download] ESTIMATED ETA for remaining {human_gb(remaining_after_probe)}: "
        f"{human_eta(eta_s)}"
    )

    all_results = list(probe_results)
    if not args.probe_only:
        rest = todo[args.probe_n :]
        if rest:
            print(f"[tcga.download] downloading remaining {len(rest)} files...")
            all_results += run_batch(tcfg["gdc_data"], rest, args.workers, "DOWNLOAD")

    # download metadata stays in scratch with the slides, not YurovskyGroup
    scratch_dir = Path(tcfg["paths"]["scratch_dir"])
    meta_dir = scratch_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "status",
        "bytes",
        "elapsed_s",
        "error",
        "project",
        "file_id",
        "file_name",
        "expected_size",
        "local_path",
    ]
    res_df = pd.DataFrame(all_results, columns=cols)
    for project in projects:
        # always write one CSV per cancer, merged with any earlier run, so reruns on a
        # project subset keep the split complete
        out = meta_dir / f"{project}_slides.csv"
        new_df = res_df[res_df["project"] == project]
        if out.exists():
            old_df = pd.read_csv(out)
            merged = pd.concat([old_df, new_df]).drop_duplicates("file_id", keep="last")
        else:
            merged = new_df
        merged.to_csv(out, index=False)
        n_ok = int((merged["status"] != "failed").sum()) if len(merged) else 0
        n_fail = int((merged["status"] == "failed").sum()) if len(merged) else 0
        print(f"[tcga.download] {project}: {n_ok} ok, {n_fail} failed -> {out}")

    n_fail_total = int((res_df["status"] == "failed").sum()) if len(res_df) else 0
    if n_fail_total:
        print(
            f"[tcga.download] {n_fail_total} file(s) failed -- rerun this script "
            f"to retry (already-downloaded files are skipped)",
            file=sys.stderr,
        )
    print("[tcga.download] DONE" if not args.probe_only else "[tcga.download] PROBE DONE")
