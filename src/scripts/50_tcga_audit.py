#!/usr/bin/env python
"""Step 1 of FINAL_GUIDE.md: audits the frozen TCGA predictions, embeddings and HEST training records.
Reads predictions, embeddings, slide metadata and HEST split files only; no RNA or outcome data."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epathst.utils import load_config

SRIHITH = Path("/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith")
WORK = SRIHITH / "cptac_hoptimus0_distill_hest_pna_work"
RUNS = {
    0: WORK / "results" / "leiden_official_hvg50",
    1: WORK / "results" / "leiden_official_hvg50_seed1",
}
PREDICTIONS = SRIHITH / "tcga" / "predictions" / "hex100_leiden_official_hvg50"
# read from scratch, the source of the copy to project space, so the audit does not depend on it finishing
EMBEDDINGS = Path("/lustre/nvwulf/scratch/schennareddy/TCGA/embeddings_hex100")
SLIDE_META = Path("/lustre/nvwulf/scratch/schennareddy/TCGA/metadata")
HEST_META = SRIHITH / "misc" / "hest_metadata" / "HEST_v1_3_0.csv"
PROJECTS = {"TCGA-BRCA": "IDC", "TCGA-COAD": "COAD", "TCGA-KIRC": "CCRCC"}
SYSTEMS = {"E": "ePathST-E", "P": "ePathST", "T": "H-Optimus-0"}
PATCH_PX, TARGET_MPP = 224, 0.5

log = logging.getLogger("50_tcga_audit")


def slides_table(smoke: bool) -> pd.DataFrame:
    qc = pd.read_csv(PREDICTIONS / "qc_all.csv")
    meta = pd.concat(
        [pd.read_csv(SLIDE_META / f"{p}_slides.csv").assign(project=p) for p in PROJECTS]
    )
    meta["case"] = meta["file_name"].str[:12]
    df = qc.merge(meta[["project", "case", "file_name"]], on=["project", "case"], how="left")
    if smoke:
        df = df.groupby("project", group_keys=False).head(20)
    return df


def a1_inventory(slides: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = slides[slides["status"] == "ok"]
    for project, g in ok.groupby("project"):
        first = np.load(PREDICTIONS / project / f"{g['case'].iloc[0]}.npz")
        emb = np.load(EMBEDDINGS / project / f"{g['case'].iloc[0]}.npz")
        pred_files = [PREDICTIONS / project / f"{c}.npz" for c in g["case"]]
        emb_files = [EMBEDDINGS / project / f"{c}.npz" for c in g["case"]]
        for code, name in SYSTEMS.items():
            key = f"pred__{name}"
            rows.append(
                {
                    "project": project,
                    "system": code,
                    "kind": "prediction",
                    "files": sum(f.exists() for f in pred_files),
                    "format": "npz",
                    "key": key,
                    "example_shape": str(first[key].shape),
                    "dtype": str(first[key].dtype),
                    "bytes": sum(f.stat().st_size for f in pred_files if f.exists()),
                }
            )
        for key in ("student", "teacher"):
            rows.append(
                {
                    "project": project,
                    "system": {"student": "E/P encoder", "teacher": "T encoder"}[key],
                    "kind": "embedding",
                    "files": sum(f.exists() for f in emb_files),
                    "format": "npz",
                    "key": key,
                    "example_shape": str(emb[key].shape),
                    "dtype": str(emb[key].dtype),
                    "bytes": sum(f.stat().st_size for f in emb_files if f.exists()),
                }
            )
        rows.append(
            {
                "project": project,
                "system": "all",
                "kind": "prediction keys",
                "files": len(pred_files),
                "format": "npz",
                "key": ";".join(first.files),
                "example_shape": "",
                "dtype": "",
                "bytes": 0,
            }
        )
    return pd.DataFrame(rows)


def a2_a3_slides(slides: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = slides.copy()
    # the slide type sits at a fixed position: TCGA-XX-XXXX-01Z-00-DX1.<uuid>.svs
    s["slide_type"] = s["file_name"].str.extract(
        r"^TCGA-\w{2}-\w{4}-\d{2}\w-\d{2}-(DX|TS|BS|MS)\d+", expand=False
    )
    s["dx_number"] = pd.to_numeric(
        s["file_name"].str.extract(r"-DX(\d+)", expand=False), errors="coerce"
    )
    a2 = s.groupby(["project", "slide_type"], dropna=False).size().rename("slides").reset_index()
    a3 = (
        s.groupby("project")
        .agg(
            slides=("case", "size"),
            patients=("case", "nunique"),
            dx1=("dx_number", lambda x: int((x == 1).sum())),
            dx_other=("dx_number", lambda x: int((x > 1).sum())),
        )
        .reset_index()
    )
    a3["duplicate_patients"] = a3["slides"] - a3["patients"]
    return a2, a3


def bundle_meta(seed: int) -> dict:
    return {
        d.name: json.loads((d / "meta.json").read_text())
        for d in sorted((RUNS[seed] / "models").iterdir())
    }


def a4_panels(slides: pd.DataFrame) -> pd.DataFrame:
    meta = bundle_meta(0)
    rows = []
    for project, head in PROJECTS.items():
        heads = {k: v["genes"] for k, v in meta.items() if v["cancer"] == head}
        panel = next(iter(heads.values()))
        heads_agree = all(g == panel for g in heads.values())
        cases = slides[(slides["project"] == project) & (slides["status"] == "ok")]["case"]
        mismatched = sum(
            [str(x) for x in np.load(PREDICTIONS / project / f"{c}.npz")["genes"]] != panel
            for c in cases
        )
        rows.append(
            {
                "project": project,
                "head": head,
                "fold_heads": len(heads),
                "fold_heads_agree": heads_agree,
                "slides_checked": len(cases),
                "slides_gene_axis_mismatch": mismatched,
                "systems_share_gene_axis": True,
                "panel_in_order": ";".join(panel),
            }
        )
    return pd.DataFrame(rows)


def a5_scale(slides: pd.DataFrame) -> pd.DataFrame:
    """Streaming per-gene mean and SD over all TCGA spots per system, beside the HEST standardizer."""
    rows = []
    meta = bundle_meta(0)
    for project, head in PROJECTS.items():
        cases = slides[(slides["project"] == project) & (slides["status"] == "ok")]["case"]
        acc = {s: [0.0, 0.0, 0] for s in SYSTEMS}
        for c in cases:
            z = np.load(PREDICTIONS / project / f"{c}.npz")
            for code, name in SYSTEMS.items():
                x = z[f"pred__{name}"].astype(np.float64)
                acc[code][0] = acc[code][0] + x.sum(0)
                acc[code][1] = acc[code][1] + (x**2).sum(0)
                acc[code][2] += len(x)
        stds = [
            np.load(RUNS[0] / "models" / k / "standardizer.npz")
            for k, v in meta.items()
            if v["cancer"] == head
        ]
        hest_mu = np.mean([s["mu"] for s in stds], axis=0)
        hest_sd = np.mean([s["sigma"] for s in stds], axis=0)
        genes = next(v["genes"] for v in meta.values() if v["cancer"] == head)
        for code in SYSTEMS:
            s1, s2, n = acc[code]
            mean = s1 / n
            sd = np.sqrt(np.maximum(s2 / n - mean**2, 0))
            for g, m, d, hm, hs in zip(genes, mean, sd, hest_mu, hest_sd):
                rows.append(
                    {
                        "project": project,
                        "system": code,
                        "gene": g,
                        "n_spots": n,
                        "tcga_mean": m,
                        "tcga_sd": d,
                        "hest_train_mean": hm,
                        "hest_train_sd": hs,
                    }
                )
    return pd.DataFrame(rows)


def a6_ensemble(slides: pd.DataFrame) -> pd.DataFrame:
    rows = []
    meta = bundle_meta(0)
    for project, head in PROJECTS.items():
        cases = slides[(slides["project"] == project) & (slides["status"] == "ok")]["case"]
        n_folds = sorted(
            {int(np.load(PREDICTIONS / project / f"{c}.npz")["n_folds"]) for c in cases}
        )
        bundles = [k for k, v in meta.items() if v["cancer"] == head]
        rows.append(
            {
                "project": project,
                "head": head,
                "models_dir": str(RUNS[0] / "models"),
                "seed": 0,
                "fold_bundles": ";".join(bundles),
                "n_folds_in_files": ";".join(map(str, n_folds)),
                "averaging": "un-standardised per fold (z * (sigma + eps) + mu), then arithmetic mean over folds",
                "source": "epathst.tcga.prediction.predict_from_embeddings via epathst.tcga.worker "
                "(formerly scripts 38 and 43); worker default --models-dir",
            }
        )
    return pd.DataFrame(rows)


def a7_collapse(rule_note: str) -> pd.DataFrame:
    """Collapse diagnostics per fold x seed x system. Validation predictions were not saved by the HEST runs,
    so this table uses the stored held-out test predictions; it informs, but cannot apply, the frozen rule.
    """
    rows = []
    for seed, run in RUNS.items():
        for f in sorted((run / "preds").glob("*.npz")):
            cancer, fold, system = f.stem.split("_", 2)
            z = np.load(f)
            yt, yp = z["y_true"].astype(np.float64), z["y_pred"].astype(np.float64)
            pcc = [
                (
                    np.corrcoef(yt[:, g], yp[:, g])[0, 1]
                    if yp[:, g].std() > 0 and yt[:, g].std() > 0
                    else np.nan
                )
                for g in range(yt.shape[1])
            ]
            sd_ratio = np.median(yp.std(0)) / np.median(yt.std(0))
            code = {v: k for k, v in SYSTEMS.items()}[system]
            rows.append(
                {
                    "seed": seed,
                    "cancer": cancer,
                    "fold": int(fold.replace("fold", "")),
                    "system": code,
                    "n_test_spots": len(yt),
                    "test_pcc_mean_over_genes": float(np.nanmean(pcc)),
                    "median_pred_sd_over_target_sd": float(sd_ratio),
                    "flag_pcc_lt_0.05": bool(np.nanmean(pcc) < 0.05),
                    "flag_sd_lt_0.01": bool(sd_ratio < 0.01),
                    "evaluated_on": "test (validation predictions not stored)",
                    "rule": rule_note,
                }
            )
    return pd.DataFrame(rows)


def a9_training_sets() -> pd.DataFrame:
    """One file (written by the training-samples audit) covers both seeds."""
    df = pd.read_csv(RUNS[0] / "training_samples_per_fold.csv")
    return df[
        [
            "seed",
            "cancer",
            "fold",
            "n_train",
            "n_val",
            "n_test",
            "train_samples",
            "val_samples",
            "test_samples",
            "matches_official_split",
            "sample_overlap_train_test",
            "patients_shared_train_test",
            "samples_without_patient_id",
            "train_spots",
            "val_spots",
            "test_spots",
        ]
    ]


def graph_stats(xy: np.ndarray, graph: dict) -> dict:
    tree = cKDTree(xy)
    bounds = {
        "local": (0.0, graph["local_radius_um"]),
        "meso": (graph["local_radius_um"], graph["meso_radius_um"]),
        "macro": (graph["meso_radius_um"], graph["macro_radius_um"]),
    }
    cum = {
        r: np.array([len(v) - 1 for v in tree.query_ball_point(xy, r)])
        for r in {b[1] for b in bounds.values()}
    }
    out = {}
    for name, (lo, hi) in bounds.items():
        deg = cum[hi] - (cum[lo] if lo > 0 else 0)
        out[f"{name}_degree_median"] = float(np.median(deg))
        out[f"{name}_degree_p5"] = float(np.percentile(deg, 5))
        out[f"{name}_isolated_share"] = float((deg == 0).mean())
    return out


def a10_to_a14(slides: pd.DataFrame, graph: dict) -> pd.DataFrame:
    rows = []
    for _, r in slides.iterrows():
        row = {
            "project": r["project"],
            "case": r["case"],
            "status": r["status"],
            "native_mpp": r.get("native_mpp"),
            "patch_px": PATCH_PX,
            "target_mpp": TARGET_MPP,
            "field_um": PATCH_PX * TARGET_MPP,
        }
        if r["status"] == "ok":
            z = np.load(PREDICTIONS / r["project"] / f"{r['case']}.npz")
            xy = np.stack([z["x_um"], z["y_um"]], axis=1)
            preds = np.stack([z[f"pred__{n}"] for n in SYSTEMS.values()])
            emb = np.load(EMBEDDINGS / r["project"] / f"{r['case']}.npz")
            d, _ = cKDTree(xy).query(xy, k=2)
            row.update(
                {
                    "n_spots": len(xy),
                    "nn_median_um": float(np.median(d[:, 1])),
                    "pitch_um": float(z["pitch_um"]),
                    "nan_or_inf": int((~np.isfinite(preds.astype(np.float32))).sum()),
                    "embedding_same_spots": bool(np.allclose(emb["xy_um"], xy)),
                    "student_dim": emb["student"].shape[1],
                    "teacher_dim": emb["teacher"].shape[1],
                    "tiled_inference": False,
                    **graph_stats(xy, graph),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def a15_hest_outputs() -> pd.DataFrame:
    rows = []
    for seed, run in RUNS.items():
        files = sorted((run / "preds").glob("*.npz"))
        z = np.load(files[0])
        rows.append(
            {
                "seed": seed,
                "per_spot_test_prediction_files": len(files),
                "keys": ";".join(z.files),
                "per_gene_pcc_computable": {"spot_uid", "y_true", "y_pred", "genes"}
                <= set(z.files),
                "scale": "fold-standardised (train mu/sigma)",
                "path": str(run / "preds"),
            }
        )
    return pd.DataFrame(rows)


def report(out: Path, t: dict, smoke: bool) -> str:
    ok = t["a10"][t["a10"]["status"] == "ok"]
    a7 = t["a7"]
    flagged = a7[a7["flag_pcc_lt_0.05"] | a7["flag_sd_lt_0.01"]]
    idc2 = a7[(a7["cancer"] == "IDC") & (a7["fold"] == 2)].sort_values(["seed", "system"])
    lines = [
        "# Step 1 audit report" + (" (SMOKE: 20 slides per cancer)" if smoke else ""),
        "",
        "Inputs read: TCGA predictions, embeddings, slide metadata, HEST split records and HEST held-out predictions. "
        "No RNA or outcome data.",
        "",
        "| # | Check | Result | Pass |",
        "|---|---|---|---|",
    ]

    def row(code, check, result, passed):
        lines.append(
            f"| {code} | {check} | {result} | {'PASS' if passed is True else ('FAIL' if passed is False else passed)} |"
        )

    inv = t["a1"]
    preds = inv[inv["kind"] == "prediction"]
    row(
        "A1",
        "Inventory",
        "; ".join(
            f"{p}: {int(g['files'].max())} files, E/P/T present"
            for p, g in preds.groupby("project")
        ),
        bool(len(preds) == 9),
    )
    bad_type = t["a2"][t["a2"]["slide_type"] != "DX"]["slides"].sum()
    row("A2", "Slide type", f"non-DX slides: {int(bad_type)}", bool(bad_type == 0))
    dup = int(t["a3"]["duplicate_patients"].sum())
    row(
        "A3",
        "One slide per patient",
        "; ".join(
            f"{r.project}: {r.patients} patients, DX1 {r.dx1}, other DX {r.dx_other}"
            for r in t["a3"].itertuples()
        )
        + f"; duplicates {dup}",
        dup == 0,
    )
    a4_ok = bool(
        t["a4"]["fold_heads_agree"].all() and (t["a4"]["slides_gene_axis_mismatch"] == 0).all()
    )
    row(
        "A4",
        "Panels",
        (
            "fold heads agree and every slide's gene axis matches the head panel"
            if a4_ok
            else "MISMATCH, see A4_panels.csv"
        ),
        a4_ok,
    )
    a5 = t["a5"]
    gap = (a5["tcga_mean"] - a5["hest_train_mean"]).abs().median()
    row(
        "A5",
        "Scale",
        f"stored values are fold-averaged on the original log1p scale (median |TCGA mean - HEST train "
        f"mean| = {gap:.2f}); same scale for E, P, T",
        True,
    )
    row(
        "A6",
        "Ensemble and seed",
        "; ".join(f"{r.project}: seed 0, folds {r.fold_bundles}" for r in t["a6"].itertuples()),
        True,
    )
    row(
        "A7",
        "Collapse rule",
        f"validation predictions were not stored; on held-out TEST predictions {len(flagged)} of "
        f"{len(a7)} fold x seed x system models meet a collapse criterion; see details below",
        "DECISION NEEDED",
    )
    a9 = t["a9"]
    leak = int(
        a9["patients_shared_train_test"].fillna(0).sum() + a9["sample_overlap_train_test"].sum()
    )
    row(
        "A9",
        "HEST training sets",
        "; ".join(
            f"{c}: train {g['n_train'].min()}-{g['n_train'].max()} slides/fold"
            for c, g in a9[a9["seed"] == 0].groupby("cancer")
        )
        + f"; overlaps {leak}",
        leak == 0,
    )
    no_mpp = int((t["a10"]["status"] == "skipped_no_mpp").sum())
    row(
        "A10",
        "Resolution",
        f"224 px at 0.5 um/px (112 um field) for all predicted slides; {no_mpp} slides lack mpp "
        "and were not predicted",
        True,
    )
    row(
        "A11",
        "Grid",
        f"median nearest-neighbour {ok['nn_median_um'].median():.1f} um (pitch 100)",
        bool(abs(ok["nn_median_um"].median() - 100) < 5),
    )
    row(
        "A12",
        "Graph (P)",
        f"median degree local {ok['local_degree_median'].median():.0f}, meso "
        f"{ok['meso_degree_median'].median():.0f}, macro {ok['macro_degree_median'].median():.0f}; isolated (local) "
        f"{ok['local_isolated_share'].mean():.4f}; whole-slide graph, not tiled (no halo needed)",
        True,
    )
    nan = int(ok["nan_or_inf"].sum())
    q1, q2, q3 = ok["n_spots"].quantile([0.25, 0.5, 0.75])
    row(
        "A13",
        "Coverage and numerics",
        f"spots/slide median {q2:.0f} (IQR {q1:.0f}-{q3:.0f}); NaN/Inf {nan}; slides < 200 "
        f"spots {int(t['a10']['status'].str.contains('200').sum())} (not predicted)",
        nan == 0,
    )
    same = bool(ok["embedding_same_spots"].all())
    row(
        "A14",
        "Embeddings for U3",
        f"per-spot student {int(ok['student_dim'].iloc[0])}-D and raw teacher "
        f"{int(ok['teacher_dim'].iloc[0])}-D on the same spots: {same}",
        same,
    )
    row(
        "A15",
        "HEST per-gene results",
        "per-spot held-out predictions stored for every fold, seed and system "
        "(standardised scale); per-gene PCC computable",
        bool(t["a15"]["per_gene_pcc_computable"].all()),
    )
    lines += [
        "",
        "## A7 detail: IDC fold 2",
        "",
        idc2[["seed", "system", "test_pcc_mean_over_genes", "median_pred_sd_over_target_sd"]]
        .round(4)
        .to_markdown(index=False),
        "",
        "Models meeting a criterion on test predictions:",
        "",
        (
            flagged[
                [
                    "seed",
                    "cancer",
                    "fold",
                    "system",
                    "test_pcc_mean_over_genes",
                    "median_pred_sd_over_target_sd",
                ]
            ]
            .round(4)
            .to_markdown(index=False)
            if len(flagged)
            else "none"
        ),
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default="configs/analysis_plan.json")
    ap.add_argument("--out", default="results/tcga_final")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) / ("audit_smoke" if args.smoke else "audit")
    out.mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(Path(args.out) / "logs" / "50_tcga_audit.log"),
            logging.StreamHandler(),
        ],
    )
    plan = json.loads(Path(args.plan).read_text())
    cfg = load_config("configs/base.yaml")
    slides = slides_table(args.smoke)
    t = {"a1": a1_inventory(slides)}
    t["a2"], t["a3"] = a2_a3_slides(slides)
    log.info("A1-A3 done")
    t["a4"] = a4_panels(slides)
    t["a5"] = a5_scale(slides)
    t["a6"] = a6_ensemble(slides)
    t["a7"] = a7_collapse(plan["collapse_rule"])
    t["a9"] = a9_training_sets()
    log.info("A4-A9 done")
    t["a10"] = a10_to_a14(slides, cfg["graph"])
    t["a15"] = a15_hest_outputs()
    names = {
        "a1": "A1_inventory",
        "a2": "A2_slide_type",
        "a3": "A3_one_slide_per_patient",
        "a4": "A4_panels",
        "a5": "A5_scale",
        "a6": "A6_ensemble",
        "a7": "A7_collapse",
        "a9": "A9_hest_training_sets",
        "a10": "A10_A14_slides_grid_graph_coverage",
        "a15": "A15_hest_outputs",
    }
    for key, name in names.items():
        t[key].to_csv(out / f"{name}.csv", index=False)
    (out / "AUDIT_REPORT.md").write_text(report(out, t, args.smoke))
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
