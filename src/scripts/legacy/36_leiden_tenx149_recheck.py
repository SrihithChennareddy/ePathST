#!/usr/bin/env python
"""Re-runs the TENX149 spatial-domain figure data on the COAD official fold0 (test samples TENX149,
TENX148, TENX147): retrains ePathST-E (A2), ePathST (A8) and the H-Optimus-0 PCA(256)+Ridge
baseline. Saves per-spot Leiden labels for TENX149 for the GT | ePathST-E | ePathST | H-Optimus-0
figure.

Usage: 36_leiden_tenx149_recheck.py SEED. Earlier variant kept for provenance; official-panel Leiden
is now `src/scripts/25_leiden_official_panel.py official`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/src")
from epathst.ablations import ABLATION_GRID
from epathst.data.expression import TrainOnlyStandardizer
from epathst.data.feature_cache import extract_student_features
from epathst.data.splits import carve_validation_from_train
from epathst.evaluation.ablation_runner import run_ablation, train_patch_head
from epathst.evaluation.official_hest import (
    SampleCache,
    build_manifest_rows,
    decode_barcodes,
    gather,
    official_target,
    row_for_sample,
)
from epathst.models.student import load_locked_encoder
from epathst.models.teacher import HOptimusTeacher
from epathst.utils import load_config, resolve_run_paths, seed_everything

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(sys.argv[1])
BENCH_ROOT = Path("/lustre/nvwulf/scratch/schennareddy/HEST-1k/hest_bench")
OUT_DIR = (
    Path("/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/results_tenx149_recheck")
    / f"seed{SEED}"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANCER, FOLD, SAMPLE = "COAD", 0, "TENX149"

print(f"[tenx149] device={DEVICE}", flush=True)
seed_everything(SEED)
cfg = load_config("/lustre/nvwulf/home/schennareddy/cellvit_project/STS_src/configs/base.yaml")
paths = resolve_run_paths(cfg)

distilled = load_locked_encoder(
    torch.load(paths["student_checkpoints"] / "locked_encoder_seed0.pt", map_location=DEVICE)[
        "encoder_state_dict"
    ],
    DEVICE,
)
teacher = HOptimusTeacher(cfg, device=DEVICE)


def embed_distilled(imgs):
    return extract_student_features(distilled, imgs, DEVICE)


def embed_hoptimus(imgs):
    bs = cfg["teacher"]["batch_size"]
    return np.concatenate(
        [teacher.embed_batch(imgs[s : s + bs]) for s in range(0, len(imgs), bs)], axis=0
    )


cancer_root = BENCH_ROOT / CANCER
genes = json.loads((cancer_root / "var_50genes.json").read_text())["genes"]
n_genes = len(genes)
split_dir = cancer_root / "splits"
train_df = pd.read_csv(split_dir / f"train_{FOLD}.csv")
test_df = pd.read_csv(split_dir / f"test_{FOLD}.csv")
all_df = pd.concat([train_df, test_df], ignore_index=True)
train_samples_all = train_df["sample_id"].tolist()
test_ids_s = test_df["sample_id"].tolist()
assert SAMPLE in test_ids_s, f"expected {SAMPLE} in fold{FOLD} test set, got {test_ids_s}"

if len(train_samples_all) >= 2:
    train_ids_s, val_ids_s = carve_validation_from_train(train_samples_all, 0.15, SEED)
    if not train_ids_s:
        train_ids_s, val_ids_s = train_samples_all, train_samples_all
else:
    train_ids_s, val_ids_s = train_samples_all, train_samples_all

cache = SampleCache(cancer_root, genes, {"distilled": embed_distilled}, DEVICE)
manifest_sample_ids = list(dict.fromkeys(train_ids_s + val_ids_s + test_ids_s))
manifest = pd.DataFrame(build_manifest_rows(manifest_sample_ids, cache, all_df))
ids_train, x_by_src_train, y_train_raw = gather(train_ids_s, cache, all_df)
ids_val, x_by_src_val, y_val_raw = gather(val_ids_s, cache, all_df)
ids_test, x_by_src_test, y_test_raw = gather(test_ids_s, cache, all_df)
x_train, x_val, x_test = (
    x_by_src_train["distilled"],
    x_by_src_val["distilled"],
    x_by_src_test["distilled"],
)

std = TrainOnlyStandardizer().fit(y_train_raw, ids=ids_train)
y_gene_train, y_gene_val, y_gene_test = (
    std.transform(y_train_raw),
    std.transform(y_val_raw),
    std.transform(y_test_raw),
)
y_path_train = np.full((len(ids_train), 1), np.nan, dtype=np.float32)
y_path_val = np.full((len(ids_val), 1), np.nan, dtype=np.float32)
y_path_test = np.full((len(ids_test), 1), np.nan, dtype=np.float32)

patch_head = train_patch_head(
    x_train, x_val, y_gene_train, y_path_train, y_gene_val, y_path_val, cfg, DEVICE
)
row_of = {uid: i for i, uid in enumerate(ids_test)}

model_preds = {}
for name in ["A2_distilled_patch", "A8_scale_aware_pna"]:
    spec = next(a for a in ABLATION_GRID if a.name == name)
    preds, r_metrics, eval_ids_test = run_ablation(
        spec,
        x_train,
        x_val,
        x_test,
        ids_train,
        ids_val,
        ids_test,
        y_gene_train,
        y_path_train,
        y_gene_val,
        y_path_val,
        y_gene_test,
        y_path_test,
        patch_head,
        manifest,
        cfg,
        SEED,
        DEVICE,
        n_hvg50=n_genes,
        pathway_names=["_unused_dummy_pathway"],
    )
    order = [row_of[u] for u in eval_ids_test]
    pred_real = preds["gene"] * (std.sigma_ + std.eps) + std.mu_
    model_preds[name] = {"pred_real": pred_real, "eval_ids_test": eval_ids_test}
    print(f"[tenx149] {name}: done", flush=True)

embed_cache = {}


def load_split_hoptimus(df):
    feats, targets, uids = [], [], []
    for _, row in df.iterrows():
        sid = row["sample_id"]
        if sid not in embed_cache:
            with h5py.File(cancer_root / row["patches_path"], "r") as f:
                imgs = f["img"][:]
                barcodes = decode_barcodes(f["barcode"][:])
            emb = embed_hoptimus(imgs)
            embed_cache[sid] = (barcodes, emb)
        barcodes, emb = embed_cache[sid]
        y = official_target(cancer_root / row["expr_path"], genes, barcodes)
        feats.append(emb)
        targets.append(y)
        uids.extend([f"{sid}::{bc}" for bc in barcodes])
    return (
        np.concatenate(feats, axis=0),
        np.concatenate(targets, axis=0).astype(np.float32),
        np.array(uids),
    )


Xh_train, yh_train, _ = load_split_hoptimus(train_df)
Xh_test, yh_test, uids_test_hoptimus = load_split_hoptimus(test_df)
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(256, len(Xh_train) - 1), random_state=SEED)),
    ]
)
Xh_train_p = pipe.fit_transform(Xh_train)
Xh_test_p = pipe.transform(Xh_test)
alpha = 100.0 / (Xh_train_p.shape[1] * yh_train.shape[1])
reg = Ridge(solver="lsqr", alpha=alpha, random_state=SEED, fit_intercept=False, max_iter=1000)
reg.fit(Xh_train_p, yh_train)
hoptimus_pred_all = reg.predict(Xh_test_p)
print("[tenx149] H-Optimus-0: done", flush=True)

y_gene_test_real_by_uid = {ids_test[i]: y_test_raw[i] for i in range(len(ids_test))}
hoptimus_pred_by_uid = {
    uids_test_hoptimus[i]: hoptimus_pred_all[i] for i in range(len(uids_test_hoptimus))
}

eval_ids_a8 = model_preds["A8_scale_aware_pna"]["eval_ids_test"]
sample_uids = [u for u in eval_ids_a8 if u.split("::")[0] == SAMPLE]
gt_matrix = np.stack([y_gene_test_real_by_uid[u] for u in sample_uids])
hoptimus_matrix = np.stack([hoptimus_pred_by_uid[u] for u in sample_uids])
valid_gene_mask = ~np.isnan(gt_matrix).any(axis=0)
print(
    f"[tenx149] {len(sample_uids)} real spots, {valid_gene_mask.sum()}/{len(valid_gene_mask)} "
    "valid genes",
    flush=True,
)
gt_matrix = gt_matrix[:, valid_gene_mask]
hoptimus_matrix = hoptimus_matrix[:, valid_gene_mask]

sample_row = row_for_sample(SAMPLE, all_df)
entry = cache.get(SAMPLE, sample_row["patches_path"], sample_row["expr_path"])
bc_to_idx = {bc: i for i, bc in enumerate(entry["barcodes"])}
spot_order_in_sample = [bc_to_idx[u.split("::")[1]] for u in sample_uids]
coords_um = entry["coords_um"][spot_order_in_sample]


def leiden_cluster(expr_matrix, resolution, seed=0):
    adata = ad.AnnData(X=expr_matrix.astype(np.float32))
    sc.pp.scale(adata, max_value=10)
    n_comps = min(30, expr_matrix.shape[1] - 1, expr_matrix.shape[0] - 1)
    sc.pp.pca(adata, n_comps=max(2, n_comps), random_state=seed)
    sc.pp.neighbors(adata, n_neighbors=15, random_state=seed)
    sc.tl.leiden(adata, resolution=resolution, random_state=seed, flavor="igraph", n_iterations=2)
    return adata.obs["leiden"].astype(int).to_numpy()


def gt_clustering(gt_matrix, min_clusters=5, max_clusters=9):
    labels, res, k = None, None, None
    for res in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]:
        labels = leiden_cluster(gt_matrix, res)
        k = len(set(labels))
        if min_clusters <= k <= max_clusters:
            return labels, res, k
    return labels, res, k


gt_labels, gt_res, gt_k = gt_clustering(gt_matrix)
print(f"[tenx149] GT: {gt_k} clusters at resolution={gt_res}", flush=True)

model_names = {"A2_distilled_patch": "ePathST-E", "A8_scale_aware_pna": "ePathST"}
ari_scores, pred_labels_all = {}, {}
for name, disp in model_names.items():
    eval_ids_n = model_preds[name]["eval_ids_test"]
    uid_to_row = {u: i for i, u in enumerate(eval_ids_n)}
    pred_rows = [uid_to_row[u] for u in sample_uids]
    pred_matrix = model_preds[name]["pred_real"][pred_rows][:, valid_gene_mask]
    pred_labels = leiden_cluster(pred_matrix, gt_res)
    ari_scores[disp] = float(adjusted_rand_score(gt_labels, pred_labels))
    pred_labels_all[disp] = pred_labels
    print(f"[tenx149] {disp}: ARI={ari_scores[disp]:.4f}", flush=True)

hoptimus_labels = leiden_cluster(hoptimus_matrix, gt_res)
ari_scores["H-Optimus-0"] = float(adjusted_rand_score(gt_labels, hoptimus_labels))
pred_labels_all["H-Optimus-0"] = hoptimus_labels
print(f"[tenx149] H-Optimus-0: ARI={ari_scores['H-Optimus-0']:.4f}", flush=True)

# Hungarian-match predicted label IDs to GT IDs for plotting only; ARI above does not need it.
gt_k_real = len(set(gt_labels))
pred_labels_aligned = {}
for disp, pl in pred_labels_all.items():
    pred_k = len(set(pl))
    k = max(gt_k_real, pred_k)
    overlap = np.zeros((k, k), dtype=int)
    for gi, pi in zip(gt_labels, pl):
        overlap[gi, pi] += 1
    row_ind, col_ind = linear_sum_assignment(-overlap)
    remap = {int(col_ind[i]): int(row_ind[i]) for i in range(len(col_ind))}
    pred_labels_aligned[disp] = np.array([remap.get(int(p), int(p)) for p in pl])

np.savez(
    OUT_DIR / "tenx149_results.npz",
    coords_um=coords_um,
    gt_labels=gt_labels,
    **{f"pred_labels__{k}": v for k, v in pred_labels_aligned.items()},
)
with open(OUT_DIR / "tenx149_meta.json", "w") as f:
    json.dump(
        {
            "cancer": CANCER,
            "sample": SAMPLE,
            "n_spots": len(sample_uids),
            "gt_resolution": gt_res,
            "gt_n_clusters": gt_k,
            "ari_scores": ari_scores,
        },
        f,
        indent=2,
    )
print("[tenx149] done", flush=True)
