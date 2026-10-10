"""Stress test of the TCGA worker claim queue (epathst.tcga.worker.Queue) on the shared file system.
Guards the one-GPU-worker-per-slide invariant of `24_tcga_validation.py worker`.

N competing processes must process every slide exactly once, and a stale lock must be taken over
by exactly one contender.
"""

import json, multiprocessing as mp, os, sys, time, uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from epathst.tcga import worker as s43

N_ITEMS, N_PROCS = 300, 8
ROOT = Path("/lustre/nvwulf/scratch/schennareddy/claimtest") / uuid.uuid4().hex[:8]


def worker(wid, items):
    q = s43.Queue(ROOT)
    mine = []
    while True:
        unfinished = [it for it in items if not q.is_done(it[0], it[1])]
        if not unfinished:
            return mine
        claim = None
        for project, case in unfinished:
            claim = q.try_claim(project, case)
            if claim is not None:
                break
        if claim is None:
            time.sleep(0.01)
            continue
        if q.is_done(project, case):  # re-check after claiming, as the real worker does
            q.release(claim)
            continue
        mine.append((project, case))
        time.sleep(0.005)  # stands in for slide processing
        qf = q.qc_file(project, case)
        qf.parent.mkdir(parents=True, exist_ok=True)
        qf.write_text(json.dumps({"status": "ok", "worker": wid}))
        q.release(claim)


def stale_contender(args):
    root, case = args
    q = s43.Queue(root)
    return q.try_claim("P", case) is not None


if __name__ == "__main__":
    items = [("P%d" % (i % 3), f"CASE{i:04d}") for i in range(N_ITEMS)]
    with mp.Pool(N_PROCS) as pool:
        res = pool.starmap(worker, [(i, items) for i in range(N_PROCS)])
    flat = [x for r in res for x in r]
    dup = len(flat) - len(set(flat))
    print(
        f"{N_PROCS} workers, {N_ITEMS} slides: processed {len(flat)} claims, {len(set(flat))} "
        f"distinct, duplicates={dup}, per-worker={[len(r) for r in res]}"
    )
    assert dup == 0 and len(set(flat)) == N_ITEMS
    left = [p for p in (ROOT / "claims").rglob("*") if p.is_dir() and not p.name.startswith("P")]
    print("lock dirs left behind:", len(left))
    assert not left
    # A dead worker's lock (old heartbeat) must be won by exactly one of the contenders.
    d = ROOT / "claims" / "P" / "STALE"
    d.mkdir(parents=True)
    (d / "heartbeat").touch()
    old = time.time() - 3600
    os.utime(d / "heartbeat", (old, old))
    with mp.Pool(N_PROCS) as pool:
        wins = pool.map(stale_contender, [(ROOT, "STALE")] * N_PROCS)
    print("stale takeover winners among 8 contenders:", sum(wins))
    assert sum(wins) == 1
    # A fresh lock must not be taken over.
    fresh = s43.Queue(ROOT)
    assert fresh.try_claim("P", "FRESH") is not None and fresh.try_claim("P", "FRESH") is None
    print("fresh lock respected: OK")
    print("ALL QUEUE TESTS PASSED")
