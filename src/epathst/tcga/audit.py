"""Audits completion and integrity of the TCGA hex-grid prediction run.
Run as `src/scripts/24_tcga_validation.py audit [--tag TAG] [--full-check]`.

Reports per slide whether it is done and its status (ok / skipped_* / FAILED), and writes
qc_all.csv and AUDIT.json into the tag folder. With --full-check every `ok` slide's npz is
opened: all systems present, shape (n_spots, 50), finite, 50 unique genes, and spot count
equal to the QC record.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from epathst.tcga import prediction, worker

SYSTEMS = prediction.SYSTEMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="hex100_leiden_official_hvg50")
    ap.add_argument("--full-check", action="store_true")
    args = ap.parse_args()
    root = prediction.PERSIST / "predictions" / args.tag
    q = worker.Queue(root)
    rows, problems = [], []
    for project, case, _ in worker.slide_list():
        f = q.qc_file(project, case)
        rec = {"project": project, "case": case, "done": q.is_done(project, case)}
        if f.exists():
            rec.update(json.loads(f.read_text()))
        else:
            rec["status"] = "pending"
        if args.full_check and rec.get("status") == "ok":
            npz = root / project / f"{case}.npz"
            try:
                z = np.load(npz)
                n = len(z["x_um"])
                assert n == rec["n_spots"], f"n_spots {n} != qc {rec['n_spots']}"
                assert len(set(z["genes"])) == 50 and len(z["genes"]) == 50, "genes"
                for s in SYSTEMS:
                    p = z[f"pred__{s}"]
                    assert (
                        p.shape == (n, 50) and np.isfinite(p.astype(np.float32)).all()
                    ), f"{s} shape/finite"
            except Exception as e:
                problems.append((project, case, f"{type(e).__name__}: {e}"))
                rec["status"] = f"CORRUPT {e}"[:200]
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["status_short"] = df["status"].astype(str).str.split(" ").str[0]
    df.to_csv(root / "qc_all.csv", index=False)
    summ = df.groupby(["project", "status_short"]).size().unstack(fill_value=0)
    summ["total"] = summ.sum(1)
    print(summ.to_string())
    ok = df[df["status"] == "ok"]
    out = {
        "tag": args.tag,
        "slides": len(df),
        "done": int(df["done"].sum()),
        "ok": len(ok),
        "problems": problems,
        "unfinished": int((~df["done"]).sum()),
        "total_spots": int(ok["n_spots"].sum()) if len(ok) else 0,
        "median_seconds_per_slide": float(ok["seconds_total"].median()) if len(ok) else None,
    }
    json.dump(out, open(root / "AUDIT.json", "w"), indent=1)
    print(
        json.dumps({k: v for k, v in out.items() if k != "problems"}), "| problems:", len(problems)
    )
    print("COMPLETE" if out["unfinished"] == 0 and not problems else "NOT COMPLETE")
