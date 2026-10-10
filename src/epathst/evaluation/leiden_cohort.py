"""Leiden clustering on the official 50-gene HEST panels for every sample of the project's HEST
cohort (the ~191 whole-transcriptome samples), one seed.
Held-out predictions come from the project 5-fold patient CV; run via
`src/scripts/25_leiden_official_panel.py cohort`.

Folds are the project's patient CV (`hest_cv_folds.json`), so every sample is held out exactly
once. The target is the official `<cancer>/var_50genes.json` panel, log1p only (script 19's
`official_target`), standardized with train-fold statistics. Only these 50 genes are loaded per
sample. A missing or duplicated panel gene stops the run; nothing is dropped or substituted.

Systems: ePathST-E (A2), ePathST (A8), H-Optimus-0 (teacher CLS -> PCA fit on train spots -> same
patch head). Clustering, metrics and tests follow the pre-registered recipe in `leiden.py`
(z-score, PCA(20), k=15, Leiden res 0.5; 0.3 and 1.0 as sensitivity; ARI + AMI). Per-fold test
predictions are saved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.decomposition import PCA

from epathst.data.expression import TrainOnlyStandardizer, ensure_gene_symbol_index
from epathst.data.feature_cache import load_cached_features
from epathst.data.splits import carve_validation_from_train, exclude_targeted_panel_samples
from epathst.evaluation.ablation_runner import run_ablation, train_patch_head
from epathst.evaluation.leiden import SPECS, cluster_sample, summarize
from epathst.evaluation.official_hest import official_pooled_pearson
from epathst.models.teacher import HOptimusTeacher
from epathst.utils import RunEnvironment, load_config, resolve_run_paths, save_json, seed_everything

BENCH_ROOT = Path("/lustre/nvwulf/scratch/schennareddy/HEST-1k/hest_bench")
SYSTEMS = {
    "ePathST-E": ("A2_distilled_patch", "distilled"),
    "ePathST": ("A8_scale_aware_pna", "distilled"),
    "H-Optimus-0": ("A2_distilled_patch", "teacher"),
}


def load_sample(
    row,
    panel: list[str],
    teacher_cache: Path,
    teacher,
    cfg,
    feat_dir: Path,
    patch_dir: Path,
    spot_by_uid: pd.DataFrame,
) -> dict:
    sid = row["sample_id"]
    adata = ensure_gene_symbol_index(ad.read_h5ad(row["expression_path"]))
    missing = [g for g in panel if g not in adata.var_names]
    if missing:  # never drop or substitute panel genes
        raise ValueError(f"{sid}: official panel genes missing from expression data: {missing}")
    assert adata.var_names[
        adata.var_names.isin(panel)
    ].is_unique, f"{sid}: duplicate panel gene names"
    genes = list(panel)
    feat = load_cached_features(feat_dir / f"{sid}.h5", expected_encoder_name="distilled")
    feat_pos = {s: i for i, s in enumerate(feat["spot_id"])}
    in_manifest = set(spot_by_uid.index[spot_by_uid["sample_id"] == sid].str.split("::").str[1])
    spots = [s for s in adata.obs_names.astype(str) if s in feat_pos and s in in_manifest]

    # teacher embeddings cached per sample so reruns resume
    tpath = teacher_cache / f"{sid}.npz"
    if tpath.exists():
        z = np.load(tpath, allow_pickle=True)
        t_pos = {s: i for i, s in enumerate(z["spot_id"])}
        emb_t = z["emb"]
    else:
        with h5py.File(patch_dir / f"{sid}.h5", "r") as f:
            pids = [p.decode() for p in f["patch_id"][:]]
            patches = f["patches"][:]
        t_pos = {s: i for i, s in enumerate(pids)}
        spots = [s for s in spots if s in t_pos]
        rows = [t_pos[s] for s in spots]
        bs = cfg["teacher"]["batch_size"]
        emb_t = np.concatenate(
            [teacher.embed_batch(patches[rows][i : i + bs]) for i in range(0, len(rows), bs)]
        )
        np.savez(tpath, spot_id=np.array(spots), emb=emb_t.astype(np.float16))
        t_pos = {s: i for i, s in enumerate(spots)}
    spots = [s for s in spots if s in t_pos]

    sub = adata[spots, genes].copy()
    sub.X = sub.X.astype(np.float64)
    sc.pp.log1p(sub)  # official target: log1p only, no normalize_total
    y = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    return {
        "uids": np.array([f"{sid}::{s}" for s in spots]),
        "genes": genes,
        "y": y.astype(np.float32),
        "x_distilled": np.stack([feat["embedding"][feat_pos[s]] for s in spots]).astype(np.float32),
        "x_teacher": emb_t[[t_pos[s] for s in spots]].astype(np.float32),
    }


def gather(samples: list[str], data: dict):
    ids = np.concatenate([data[s]["uids"] for s in samples])
    x = {k: np.concatenate([data[s][f"x_{k}"] for s in samples]) for k in ("distilled", "teacher")}
    y = np.concatenate([data[s]["y"] for s in samples])
    return ids, x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cancers", nargs="+", default=None)
    ap.add_argument("--max-folds-per-cancer", type=int, default=None)
    ap.add_argument("--min-spots", type=int, default=50)
    ap.add_argument("--out-name", default="leiden_cohort_official50")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    device = args.device if torch.cuda.is_available() else "cpu"
    seed = args.seed
    seed_everything(seed)

    out_dir = Path(cfg["paths"]["persist_dir"]) / args.out_name
    teacher_cache = (
        Path(cfg["paths"]["persist_dir"]) / "leiden_cohort_official50" / "teacher_embeddings"
    )  # shared by smoke + full runs
    teacher_cache.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preds").mkdir(exist_ok=True)

    hest_samples = exclude_targeted_panel_samples(
        pd.read_csv(paths["manifests"] / "hest_samples.csv")
    )
    spot_manifest = pd.read_csv(paths["manifests"] / "hest_spot_manifest.csv")
    spot_by_uid = spot_manifest.set_index("spot_uid")
    cv_grid = json.loads((paths["manifests"] / "hest_cv_folds.json").read_text())
    if args.cancers:
        cv_grid["per_cancer"] = {
            k: v for k, v in cv_grid["per_cancer"].items() if k in args.cancers
        }
    patch_dir = Path(cfg["paths"]["work_dir"]) / "hest_patches"
    feat_dir = paths["student_embeddings"] / "distilled"
    teacher = HOptimusTeacher(cfg, device=device)

    all_rows = []
    with RunEnvironment("37_leiden_cohort_official50", out_dir / "run_env.json"):
        for cancer, cv in cv_grid["per_cancer"].items():
            official = json.loads((BENCH_ROOT / cancer / "var_50genes.json").read_text())["genes"]
            csamples = hest_samples[hest_samples["cancer"] == cancer]
            print(
                f"[leiden_cohort] {cancer}: loading {len(csamples)} samples (official panel of "
                f"{len(official)} genes)...",
                flush=True,
            )
            data = {}
            for _, row in csamples.iterrows():
                if (
                    not (patch_dir / f"{row['sample_id']}.h5").exists()
                    and not (teacher_cache / f"{row['sample_id']}.npz").exists()
                ):
                    print(
                        f"[leiden_cohort] {row['sample_id']}: no cached patches, skipping",
                        flush=True,
                    )
                    continue
                data[row["sample_id"]] = load_sample(
                    row, official, teacher_cache, teacher, cfg, feat_dir, patch_dir, spot_by_uid
                )
            # load_sample raises unless every sample has all panel genes, in panel order
            panel = official
            print(
                f"[leiden_cohort] {cancer}: {len(data)} samples loaded; all {len(panel)} official genes "
                "present",
                flush=True,
            )

            folds = (
                cv["folds"][: args.max_folds_per_cancer]
                if args.max_folds_per_cancer
                else cv["folds"]
            )
            for fold in folds:
                fi = fold["fold"]
                train_p, val_p = carve_validation_from_train(fold["train_patients"], 0.15, seed)

                def samples_of(patients):
                    return [
                        s
                        for s in csamples[csamples["patient_id"].isin(patients)]["sample_id"]
                        if s in data
                    ]

                train_s, val_s, test_s = (
                    samples_of(train_p),
                    samples_of(val_p),
                    samples_of(fold["test_patients"]),
                )
                print(
                    f"[leiden_cohort] {cancer} fold{fi}: {len(train_s)} train / {len(val_s)} val / "
                    f"{len(test_s)} test samples",
                    flush=True,
                )
                ids_tr, x_tr, y_tr_raw = gather(train_s, data)
                ids_va, x_va, y_va_raw = gather(val_s, data)
                ids_te, x_te, y_te_raw = gather(test_s, data)

                std = TrainOnlyStandardizer().fit(y_tr_raw, ids=ids_tr)
                std.assert_no_leakage(ids_va)
                std.assert_no_leakage(ids_te)
                y_tr, y_va, y_te = (
                    std.transform(y_tr_raw),
                    std.transform(y_va_raw),
                    std.transform(y_te_raw),
                )
                nan = lambda n: np.full((n, 1), np.nan, dtype=np.float32)  # noqa: E731
                yp_tr, yp_va, yp_te = nan(len(ids_tr)), nan(len(ids_va)), nan(len(ids_te))

                pca = PCA(n_components=x_tr["distilled"].shape[1], random_state=0).fit(
                    x_tr["teacher"]
                )
                feats = {
                    "distilled": (x_tr["distilled"], x_va["distilled"], x_te["distilled"]),
                    "teacher": tuple(
                        pca.transform(x["teacher"]).astype(np.float32) for x in (x_tr, x_va, x_te)
                    ),
                }
                heads = {
                    src: train_patch_head(f[0], f[1], y_tr, yp_tr, y_va, yp_va, cfg, device)
                    for src, f in feats.items()
                }

                row_of = {u: i for i, u in enumerate(ids_te)}
                preds_by_sys = {}
                for sysname, (spec_name, src) in SYSTEMS.items():
                    f = feats[src]
                    preds, _, eval_ids = run_ablation(
                        SPECS[spec_name],
                        f[0],
                        f[1],
                        f[2],
                        ids_tr,
                        ids_va,
                        ids_te,
                        y_tr,
                        yp_tr,
                        y_va,
                        yp_va,
                        y_te,
                        yp_te,
                        heads[src],
                        spot_manifest,
                        cfg,
                        seed,
                        device,
                        n_hvg50=len(panel),
                        pathway_names=["_unused_dummy_pathway"],
                    )
                    order = [row_of[u] for u in eval_ids]
                    preds_by_sys[sysname] = (eval_ids, y_te[order], preds["gene"])
                    pcc = official_pooled_pearson(y_te[order], preds["gene"])
                    print(
                        f"[leiden_cohort] {cancer} fold{fi} {sysname}: pooled_pcc={pcc:.4f}",
                        flush=True,
                    )
                    coords = spot_by_uid.loc[eval_ids, ["x_um", "y_um"]].values
                    np.savez_compressed(
                        out_dir / "preds" / f"{cancer}_fold{fi}_{sysname}.npz",
                        spot_uid=np.array(eval_ids),
                        x_um=coords[:, 0],
                        y_um=coords[:, 1],
                        y_true=y_te[order],
                        y_pred=preds["gene"],
                        genes=np.array(panel),
                    )

                for sid in test_s:
                    if cluster_sample(cancer, fi, sid, preds_by_sys, args.min_spots, all_rows):
                        print(f"[leiden_cohort] {cancer} fold{fi} {sid}: done", flush=True)
                    else:
                        print(
                            f"[leiden_cohort] {sid}: < {args.min_spots} spots, skipped", flush=True
                        )
                pd.DataFrame(all_rows).to_csv(out_dir / "per_sample_leiden.csv", index=False)
            del data

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "per_sample_leiden.csv", index=False)
    summarize(df, out_dir)
    save_json({"config": args.__dict__}, out_dir / "run_args.json")
    print(f"[leiden_cohort] wrote {out_dir}")
