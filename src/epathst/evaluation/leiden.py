"""Leiden clustering of measured versus predicted expression on the official 50-gene HEST panels.
Trains the three compared systems per official fold and scores agreement of their domains with ARI
and AMI; run via `src/scripts/25_leiden_official_panel.py official`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from epathst.ablations import ABLATION_GRID
from epathst.data.expression import TrainOnlyStandardizer
from epathst.data.feature_cache import extract_student_features
from epathst.data.splits import carve_validation_from_train
from epathst.evaluation import ablation_runner
from epathst.evaluation.ablation_runner import run_ablation, train_patch_head
from epathst.evaluation.official_hest import (
    SampleCache,
    build_manifest_rows,
    gather,
    official_pooled_pearson,
)
from epathst.models.student import load_locked_encoder
from epathst.models.teacher import HOptimusTeacher
from epathst.utils import load_config, resolve_run_paths, save_json, seed_everything

SPECS = {s.name: s for s in ABLATION_GRID}


# `run_ablation` never returns its fitted ResidualGraphPredictor. Wrap its predict step (in the
# ablation runner's namespace) to keep the model, save it, and check that it reloads.
_CAPTURE: dict = {}


_orig_predict_graph_model = ablation_runner.predict_graph_model


def _capturing_predict(model, graphs, device, backend=None):
    out = _orig_predict_graph_model(model, graphs, device, backend=backend)
    save_path = _CAPTURE.get("save_path")
    if save_path is not None:
        torch.save(
            model, save_path
        )  # whole module: patch head + graph correction incl. PNA degree buffers
        reloaded = torch.load(save_path, map_location=device, weights_only=False)
        again = _orig_predict_graph_model(reloaded, graphs, device, backend=backend)
        # PNA scatter-add on GPU is not bit-exact between calls, so use a tolerance far below any
        # real save/load error (predictions are O(1) z-scores) and report the actual gap.
        gap = float(np.abs(again["gene_pred"] - out["gene_pred"]).max())
        print(f"[leiden] ePathST save/reload max |diff| = {gap:.2e}", flush=True)
        assert gap < 1e-3, f"reloaded ePathST predicts differently (max |diff| {gap:.2e})"
        _CAPTURE["verified"] = True
    return out


ablation_runner.predict_graph_model = _capturing_predict


COMMON_META: dict = {}


RESOLUTIONS = (0.3, 0.5, 1.0)


PRIMARY_RES = 0.5


MIN_SPOTS = 50


DEFAULT_BENCH = "/lustre/nvwulf/scratch/schennareddy/HEST-1k/hest_bench"


def leiden_labels(expr: np.ndarray, seed: int = 0) -> dict[float, np.ndarray]:
    adata = ad.AnnData(X=np.array(expr, dtype=np.float32))  # copy: sc.pp.scale works in place
    sc.pp.scale(adata)
    adata.X = np.nan_to_num(adata.X)
    sc.pp.pca(adata, n_comps=max(2, min(20, adata.n_vars - 1, adata.n_obs - 1)), random_state=seed)
    sc.pp.neighbors(adata, n_neighbors=15, random_state=seed)
    out = {}
    for res in RESOLUTIONS:
        sc.tl.leiden(
            adata,
            resolution=res,
            random_state=seed,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        out[res] = adata.obs["leiden"].astype(int).to_numpy()
    return out


def cluster_sample(cancer, fold, sid, preds_by_sys, min_spots, rows, label_store=None) -> bool:
    """Clusters one test sample for every system. Measured expression is clustered ONCE, in the
    first system's spot order; every system's predictions are reindexed to that order by spot ID.
    Stops (AssertionError) if any system lacks exactly the same spots or its measured matrix
    differs, so nothing is silently misaligned.
    """
    per_sys = {}
    for name, (eval_ids, y_true, y_pred) in preds_by_sys.items():
        idx = [i for i, u in enumerate(eval_ids) if u.startswith(f"{sid}::")]
        per_sys[name] = ([eval_ids[i] for i in idx], y_true[idx], y_pred[idx])
    ref_name = next(iter(per_sys))
    ref_ids, ref_true, _ = per_sys[ref_name]
    if len(ref_ids) < min_spots:
        return False
    gt = leiden_labels(ref_true, seed=0)
    if label_store is not None:
        label_store["spot_uid"] = np.array(ref_ids)
        label_store["y_true"] = ref_true
        for res in RESOLUTIONS:
            label_store[f"gt__res{res}"] = gt[res]
    for name, (ids, y_true, y_pred) in per_sys.items():
        assert len(ids) == len(ref_ids) and set(ids) == set(
            ref_ids
        ), f"{sid}: {name} has a different spot set than {ref_name}"
        pos = {u: i for i, u in enumerate(ids)}
        order = [pos[u] for u in ref_ids]
        assert np.allclose(
            y_true[order], ref_true
        ), f"{sid}: measured matrix for {name} differs from {ref_name}"
        pr = leiden_labels(y_pred[order], seed=0)
        if label_store is not None:
            label_store[f"y_pred__{name}"] = y_pred[order]
            for res in RESOLUTIONS:
                label_store[f"{name}__res{res}"] = pr[res]
        for res in RESOLUTIONS:
            rows.append(
                {
                    "cancer": cancer,
                    "fold": fold,
                    "sample_id": sid,
                    "system": name,
                    "resolution": res,
                    "n_spots": len(ref_ids),
                    "n_clusters_gt": len(set(gt[res])),
                    "n_clusters_pred": len(set(pr[res])),
                    "ari": float(adjusted_rand_score(gt[res], pr[res])),
                    "ami": float(adjusted_mutual_info_score(gt[res], pr[res])),
                }
            )
    return True


def run_cancer(
    bench_root: Path,
    cancer: str,
    embed_fns: dict,
    cfg: dict,
    seed: int,
    device: str,
    max_folds,
    out_dir: Path,
) -> list[dict]:
    cancer_root = bench_root / cancer
    genes = json.loads((cancer_root / "var_50genes.json").read_text())["genes"]
    assert (
        len(genes) == 50 and len(set(genes)) == 50
    ), f"{cancer}: official panel is not 50 unique genes"
    split_dir = cancer_root / "splits"
    n_splits = len(list(split_dir.glob("train_*.csv")))
    assert n_splits > 0, f"no official splits under {split_dir}"
    if max_folds is not None:
        n_splits = min(n_splits, max_folds)
    cache = SampleCache(cancer_root, genes, embed_fns, device)
    rows = []

    for fold in range(n_splits):
        train_df = pd.read_csv(split_dir / f"train_{fold}.csv")
        test_df = pd.read_csv(split_dir / f"test_{fold}.csv")
        all_df = pd.concat([train_df, test_df], ignore_index=True)

        train_all = train_df["sample_id"].tolist()
        if len(train_all) >= 2:
            train_s, val_s = carve_validation_from_train(train_all, 0.15, seed)
            if not train_s:
                train_s, val_s = train_all, train_all
        else:
            train_s, val_s = train_all, train_all
        test_s = test_df["sample_id"].tolist()
        fallback = val_s == train_s
        print(
            f"[leiden] {cancer} fold{fold}: {len(train_s)} train / "
            f"{len(val_s)} val / {len(test_s)} test samples"
            f"{' (val==train fallback)' if fallback else ''}",
            flush=True,
        )

        manifest = pd.DataFrame(
            build_manifest_rows(list(dict.fromkeys(train_s + val_s + test_s)), cache, all_df)
        )
        ids_tr, x_tr, y_tr_raw = gather(train_s, cache, all_df)
        ids_va, x_va, y_va_raw = gather(val_s, cache, all_df)
        ids_te, x_te, y_te_raw = gather(test_s, cache, all_df)

        std = TrainOnlyStandardizer().fit(y_tr_raw, ids=ids_tr)
        if not fallback:
            std.assert_no_leakage(ids_va)
        std.assert_no_leakage(ids_te)
        y_tr, y_va, y_te = std.transform(y_tr_raw), std.transform(y_va_raw), std.transform(y_te_raw)
        nan = lambda n: np.full((n, 1), np.nan, dtype=np.float32)  # noqa: E731
        yp_tr, yp_va, yp_te = nan(len(ids_tr)), nan(len(ids_va)), nan(len(ids_te))

        # teacher PCA is fit on this fold's TRAIN spots only (no test leakage)
        pca = PCA(n_components=x_tr["distilled"].shape[1], random_state=0).fit(x_tr["teacher"])
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

        systems = {
            "ePathST-E": ("A2_distilled_patch", "distilled"),
            "ePathST": ("A8_scale_aware_pna", "distilled"),
            "H-Optimus-0": ("A2_distilled_patch", "teacher"),
        }
        row_of = {u: i for i, u in enumerate(ids_te)}
        mdir = out_dir / "models" / f"{cancer}_fold{fold}"
        mdir.mkdir(parents=True, exist_ok=True)
        np.savez(mdir / "standardizer.npz", mu=std.mu_, sigma=std.sigma_, eps=std.eps)
        np.savez(mdir / "hoptimus_pca.npz", mean=pca.mean_, components=pca.components_)
        torch.save(heads["distilled"], mdir / "epathst_e_patch_head.pt")
        torch.save(heads["teacher"], mdir / "hoptimus_patch_head.pt")
        (mdir / "meta.json").write_text(
            json.dumps(
                {
                    "cancer": cancer,
                    "fold": fold,
                    "seed": seed,
                    "genes": genes,
                    "train_samples": train_s,
                    "val_samples": val_s,
                    "test_samples": test_s,
                    "student_in_dim": int(feats["distilled"][0].shape[1]),
                    "teacher_raw_dim": int(x_tr["teacher"].shape[1]),
                    "teacher_pca_dim": int(pca.n_components_),
                    **COMMON_META,
                },
                indent=1,
            )
        )
        preds_by_sys = {}
        for sysname, (spec_name, src) in systems.items():
            f = feats[src]
            _CAPTURE.clear()
            if sysname == "ePathST":
                _CAPTURE["save_path"] = mdir / "epathst_full.pt"
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
                manifest,
                cfg,
                seed,
                device,
                n_hvg50=len(genes),
                pathway_names=["_unused_dummy_pathway"],
            )
            if sysname == "ePathST":
                assert _CAPTURE.get("verified"), "ePathST model was not saved/verified"
            order = [row_of[u] for u in eval_ids]
            preds_by_sys[sysname] = (eval_ids, y_te[order], preds["gene"])
            pcc = official_pooled_pearson(y_te[order], preds["gene"])
            print(
                f"[leiden] {cancer} fold{fold} {sysname}: " f"official_pooled_pcc={pcc:.4f}",
                flush=True,
            )
            coords = manifest.set_index("spot_uid").loc[eval_ids, ["x_um", "y_um"]].values
            (out_dir / "preds").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_dir / "preds" / f"{cancer}_fold{fold}_{sysname}.npz",
                spot_uid=np.array(eval_ids),
                x_um=coords[:, 0],
                y_um=coords[:, 1],
                y_true=y_te[order],
                y_pred=preds["gene"],
                genes=np.array(genes),
            )

        for sid in test_s:
            store = {}
            if not cluster_sample(cancer, fold, sid, preds_by_sys, MIN_SPOTS, rows, store):
                print(f"[leiden] {sid}: <{MIN_SPOTS} spots, skipped", flush=True)
            else:
                xy = manifest.set_index("spot_uid").loc[store["spot_uid"], ["x_um", "y_um"]].values
                (out_dir / "labels").mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_dir / "labels" / f"{cancer}_{sid}.npz",
                    x_um=xy[:, 0],
                    y_um=xy[:, 1],
                    genes=np.array(genes),
                    **store,
                )
                print(f"[leiden] {cancer} fold{fold} {sid}: done", flush=True)
        pd.DataFrame(rows).to_csv(out_dir / f"per_sample_leiden_{cancer}.csv", index=False)
    return rows


def summarize(df: pd.DataFrame, out_dir: Path) -> None:
    g = df.groupby(["cancer", "system", "resolution"])
    s = g.agg(
        n_samples=("sample_id", "nunique"),
        ari_mean=("ari", "mean"),
        ari_sd=("ari", "std"),
        ami_mean=("ami", "mean"),
        ami_sd=("ami", "std"),
    ).reset_index()
    pooled = (
        df.groupby(["system", "resolution"])
        .agg(
            n_samples=("sample_id", "nunique"),
            ari_mean=("ari", "mean"),
            ari_sd=("ari", "std"),
            ami_mean=("ami", "mean"),
            ami_sd=("ami", "std"),
        )
        .reset_index()
        .assign(cancer="ALL")
    )
    s = pd.concat([s, pooled], ignore_index=True)
    s.to_csv(out_dir / "leiden_summary.csv", index=False)
    print(s[s.resolution == PRIMARY_RES].round(4).to_string(index=False))

    tests = []
    for a, b in (
        ("ePathST", "ePathST-E"),
        ("ePathST", "H-Optimus-0"),
        ("H-Optimus-0", "ePathST-E"),
    ):
        for cancer in ["ALL", *sorted(df.cancer.unique())]:
            d = df if cancer == "ALL" else df[df.cancer == cancer]
            for res in RESOLUTIONS:
                p = d[d.resolution == res].pivot_table(
                    index="sample_id", columns="system", values="ari"
                )
                p = p[[a, b]].dropna()
                if len(p) >= 6 and (p[a] - p[b]).abs().sum() > 0:
                    stat = wilcoxon(p[a], p[b])
                    tests.append(
                        {
                            "cancer": cancer,
                            "resolution": res,
                            "a": a,
                            "b": b,
                            "n": len(p),
                            "mean_diff_ari": float((p[a] - p[b]).mean()),
                            "wilcoxon_p": float(stat.pvalue),
                        }
                    )
    pd.DataFrame(tests).to_csv(out_dir / "leiden_paired_tests.csv", index=False)


def main():
    """Leiden clustering on the official HEST-1k Benchmark 50-gene panels (IDC, COAD, CCRCC;
    `<cancer>/var_50genes.json`) for every test sample of the official k-fold splits.

    Systems (same splits, genes, log1p-only target, patch-head recipe):
      ePathST-E   = A2_distilled_patch (frozen distilled student, patch head only)
      ePathST     = A8_scale_aware_pna (adds multiscale scale-aware PNA graph correction)
      H-Optimus-0 = teacher CLS -> PCA(320, fit on train-fold spots) -> same patch head
    Training reuses the `official_hest` loaders and `ablation_runner.run_ablation` unmodified.

    Clustering recipe (pre-registered): per test sample, z-score the 50 genes, PCA(20), kNN k=15,
    Leiden (igraph, 2 iterations), random_state 0, resolution 0.5 (sensitivity 0.3 and 1.0).
    Measured and predicted expression are clustered independently at the same resolution (no
    per-sample resolution search) and compared on the same spots with ARI and AMI. Samples with
    <50 spots are skipped. Per-fold test predictions are saved (.npz) so downstream analyses need
    no retraining.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--bench-root", default=DEFAULT_BENCH)
    ap.add_argument("--cancers", nargs="+", default=["IDC", "COAD", "CCRCC"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--max-folds-per-cancer", type=int, default=None)
    ap.add_argument("--out-name", default="leiden_official_hvg50")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    seed_everything(args.seed)

    ckpt = torch.load(
        paths["student_checkpoints"] / f"locked_encoder_seed{args.feature_seed}.pt",
        map_location=args.device,
    )
    student = load_locked_encoder(ckpt["encoder_state_dict"], args.device)
    COMMON_META.update(
        {
            "student_encoder_checkpoint": str(
                paths["student_checkpoints"] / f"locked_encoder_seed{args.feature_seed}.pt"
            ),
            "teacher_hf_repo": cfg["teacher"]["hf_repo"],
            "bench_root": args.bench_root,
            "graph_cfg": cfg["graph"],
            "patch_head_cfg": cfg["patch_head"],
            "systems": {
                "ePathST-E": "A2 patch head on frozen student features",
                "ePathST": "A8 scale-aware PNA ResidualGraphPredictor on frozen student features",
                "H-Optimus-0": "patch head on teacher CLS -> PCA(student dim)",
            },
        }
    )
    teacher = HOptimusTeacher(cfg, args.device)
    bs = cfg["teacher"]["batch_size"]

    def embed_teacher(imgs):
        return np.concatenate(
            [teacher.embed_batch(imgs[i : i + bs]) for i in range(0, len(imgs), bs)]
        )

    embed_fns = {
        "distilled": lambda imgs: extract_student_features(student, imgs, args.device),
        "teacher": embed_teacher,
    }

    out_dir = paths["results"] / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cancer in args.cancers:
        rows.extend(
            run_cancer(
                Path(args.bench_root),
                cancer,
                embed_fns,
                cfg,
                args.seed,
                args.device,
                args.max_folds_per_cancer,
                out_dir,
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_sample_leiden.csv", index=False)
    summarize(df, out_dir)
    save_json({"config": args.__dict__}, out_dir / "run_args.json")
    print(f"[leiden] wrote {out_dir}")
