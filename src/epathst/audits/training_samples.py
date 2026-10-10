"""Audits which HEST samples trained, validated and tested each official-protocol fold.
Run as `src/scripts/26_hest_audits.py training-samples <results dir with leiden_official_hvg50*/>`.

Reads the saved bundles' meta.json (the lists actually used), checks them against the official
split files (`<cancer>/splits/{train,test}_<fold>.csv`), and checks sample-level and
patient-level overlap between train/val and test via the `patient` column of the HEST metadata.
The ePathST-E head, H-Optimus-0 head and ePathST graph model are trained in the same fold on
the same train/val samples.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BENCH = Path("/lustre/nvwulf/scratch/schennareddy/HEST-1k/hest_bench")
META = Path(
    "/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/misc/hest_metadata/HEST_v1_3_0.csv"
)


def main(res: Path):
    md = pd.read_csv(META)
    pat = dict(zip(md["id"], md["patient"]))
    rows, problems = [], []
    for run in ("leiden_official_hvg50", "leiden_official_hvg50_seed1"):
        for meta_f in sorted((res / run / "models").glob("*_fold*/meta.json")):
            m = json.loads(meta_f.read_text())
            c, f = m["cancer"], m["fold"]
            tr, va, te = set(m["train_samples"]), set(m["val_samples"]), set(m["test_samples"])
            ot = set(pd.read_csv(BENCH / c / "splits" / f"train_{f}.csv")["sample_id"])
            oe = set(pd.read_csv(BENCH / c / "splits" / f"test_{f}.csv")["sample_id"])
            ok_split = ((tr | va) == ot) and (te == oe)
            sample_overlap = (tr | va) & te
            ppl = {s: pat.get(s) for s in tr | va | te}
            unknown = sorted(s for s, p in ppl.items() if pd.isna(p))
            p_train = {ppl[s] for s in tr | va if not pd.isna(ppl[s])}
            p_test = {ppl[s] for s in te if not pd.isna(ppl[s])}
            shared = sorted(p_train & p_test)
            if not ok_split or sample_overlap or shared:
                problems.append((run, c, f, ok_split, sorted(sample_overlap), shared))
            rows.append(
                {
                    "run": run,
                    "seed": m["seed"],
                    "cancer": c,
                    "fold": f,
                    "n_train": len(tr),
                    "n_val": len(va),
                    "n_test": len(te),
                    "train_samples": ";".join(sorted(tr)),
                    "val_samples": ";".join(sorted(va)),
                    "test_samples": ";".join(sorted(te)),
                    "matches_official_split": ok_split,
                    "sample_overlap_train_test": len(sample_overlap),
                    "patients_shared_train_test": len(shared),
                    "samples_without_patient_id": ";".join(unknown),
                }
            )
    df = pd.DataFrame(rows)
    out = res / "leiden_official_hvg50"
    df.to_csv(out / "training_samples_per_fold.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    s0 = df[df.run == "leiden_official_hvg50"]
    print(
        "SEED 0 (train = train + val used for early stopping; the same samples train ePathST-E "
        "head, H-Optimus-0 head, ePathST graph):"
    )
    print(
        s0[
            [
                "cancer",
                "fold",
                "n_train",
                "n_val",
                "n_test",
                "train_samples",
                "val_samples",
                "test_samples",
            ]
        ].to_string(index=False)
    )
    print("\nseed 1 vs seed 0 train/val/test lists identical?", end=" ")
    s1 = df[df.run == "leiden_official_hvg50_seed1"].set_index(["cancer", "fold"])
    a = s0.set_index(["cancer", "fold"])
    print(
        {k: bool((a[k] == s1[k]).all()) for k in ("train_samples", "val_samples", "test_samples")}
    )
    print(
        "\nall folds match the official split files:",
        bool(df.matches_official_split.all()),
        "| sample overlap train/test:",
        int(df.sample_overlap_train_test.sum()),
        "| patients shared train/test:",
        int(df.patients_shared_train_test.sum()),
    )
    print(
        "samples with no patient id in HEST metadata:",
        sorted(set(";".join(df.samples_without_patient_id).split(";")) - {""}),
    )
    print("problems:", problems if problems else "none")
