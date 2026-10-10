"""Writes FROZEN_WEIGHTS_MANIFEST.json (paths, sizes, sha256) for one official-protocol run.
Run as `src/scripts/26_hest_audits.py frozen-weights <run results dir>`.

Covers the frozen student and teacher encoders and every saved fold bundle, so the exact weights
behind reported numbers can be verified later.
"""

from __future__ import annotations

import datetime
import glob
import hashlib
import json
import os

SRIHITH = "/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def main(run_dir: str):
    meta0 = json.load(open(sorted(glob.glob(f"{run_dir}/models/*_fold*/meta.json"))[0]))
    enc = meta0["student_encoder_checkpoint"]
    ho = glob.glob(
        SRIHITH
        + "hf_cache/huggingface/hub/models--bioptimus--H-optimus-0/snapshots/*/pytorch_model.bin"
    )[0]
    real = os.path.realpath(ho)
    m = {
        "created": datetime.date.today().isoformat(),
        "run_dir": run_dir,
        "seed": meta0["seed"],
        "note": "Encoders are frozen; heads / GNN / PCA / standardizer in fold_bundles are the "
        "trained, saved parts.",
        "frozen_student_encoder": {
            "path": enc,
            "bytes": os.path.getsize(enc),
            "sha256": sha(enc),
            "role": "distilled ConvNeXt V2-Atto, 3.39M params, frozen",
        },
        "frozen_teacher_H-Optimus-0": {
            "hf_repo": meta0["teacher_hf_repo"],
            "path": ho,
            "blob": real,
            "bytes": os.path.getsize(real),
            "sha256": sha(real),
            "role": "teacher, frozen",
        },
        "fold_bundles": {},
    }
    for d in sorted(glob.glob(f"{run_dir}/models/*_fold*")):
        m["fold_bundles"][os.path.basename(d)] = {
            os.path.basename(f): {"bytes": os.path.getsize(f), "sha256": sha(f)}
            for f in sorted(glob.glob(d + "/*"))
        }
    json.dump(m, open(f"{run_dir}/FROZEN_WEIGHTS_MANIFEST.json", "w"), indent=1)
    print("wrote manifest:", len(m["fold_bundles"]), "fold bundles")
