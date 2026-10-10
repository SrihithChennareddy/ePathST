"""Per-spot A8 (ePathST) and H-Optimus-0 + Ridge predictions on official HEST-bench fold 0.
Feeds fig_spatial_predictions.py; writes {cancer}_infer.npz and {cancer}_infer_meta.json.

Outputs go to spatial_predictions/ under the figure output dir. Reuses the official-protocol code
unchanged (epathst.evaluation.official_hest, ablation_runner's train_patch_head and run_ablation,
and the PCA256 + Ridge recipe of 19_official_hest_benchmark.py). The only new logic keeps the
per-spot predictions that are normally discarded and runs just fold 0 x A8, since the other
ablations are not needed here. The per-sample Pearson r is reported as computed, even where A8
loses to H-Optimus-0."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "figures"))
from epathst.ablations import ABLATION_GRID
from epathst.data.expression import TrainOnlyStandardizer
from epathst.data.feature_cache import extract_student_features
from epathst.models.student import load_locked_encoder
from epathst.models.teacher import HOptimusTeacher
from epathst.utils import load_config, resolve_run_paths, seed_everything
from epathst.evaluation.ablation_runner import run_ablation, train_patch_head
from epathst.evaluation.official_hest import (
    SampleCache,
    build_manifest_rows,
    decode_barcodes,
    gather,
    official_pooled_pearson,
    official_target,
    row_for_sample,
)
from _paths import BENCH_ROOT, FIG_OUT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1  # official BenchmarkConfig default, matches scripts 19/22
OUT_DIR = FIG_OUT / "spatial_predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENE_PREFERENCE = {"IDC": "GATA3", "COAD": "CEACAM6", "CCRCC": "TMEM91"}
A8_SPEC = next(s for s in ABLATION_GRID if s.name == "A8_scale_aware_pna")

print(f"[infer] device={DEVICE}", flush=True)
seed_everything(SEED)
cfg = load_config(str(ROOT / "configs/base.yaml"))
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


def pick_test_sample(cancer_root: Path, test_ids: list[str]) -> str:
    """Pick the test sample with the most irregular spot layout (and enough spots).

    Returns the only sample when there is one. Not called at present: run_cancer pins the
    (sample, gene) pair instead.
    """
    if len(test_ids) == 1:
        return test_ids[0]
    from scipy import ndimage

    best_sid, best_score = test_ids[0], -1.0
    for sid in test_ids:
        with h5py.File(cancer_root / "patches" / f"{sid}.h5", "r") as f:
            coords = f["coords"][:].astype(np.float64)
        # Occupancy grid at the spot pitch; a low fill fraction means an irregular silhouette.
        gx = np.round(
            (coords[:, 0] - coords[:, 0].min())
            / max(1.0, np.median(np.diff(np.sort(np.unique(coords[:, 0])))))
        )
        gy = np.round(
            (coords[:, 1] - coords[:, 1].min())
            / max(1.0, np.median(np.diff(np.sort(np.unique(coords[:, 1])))))
        )
        occ = np.zeros((int(gy.max()) + 1, int(gx.max()) + 1), dtype=bool)
        occ[gy.astype(int), gx.astype(int)] = True
        fill_frac = occ.sum() / occ.size
        irregularity = 1.0 - fill_frac  # full rectangle -> 0; irregular blob -> higher
        n_spots = len(coords)
        score = irregularity * min(1.0, n_spots / 300.0)  # down-weight samples under 300 spots
        print(
            f"[infer]   candidate {sid}: n={n_spots} fill_frac={fill_frac:.2f} score={score:.3f}",
            flush=True,
        )
        if score > best_score:
            best_sid, best_score = sid, score
    return best_sid


def run_cancer(cancer: str) -> dict:
    print(f"[infer] ===== {cancer} =====", flush=True)
    cancer_root = BENCH_ROOT / cancer
    genes = json.loads((cancer_root / "var_50genes.json").read_text())["genes"]
    # Pinned to the (sample, gene) of the stored ePathST predictions
    # (figures/data/spatial_predictions/{cancer}_infer_meta.json) instead of re-deriving it with
    # pick_test_sample/GENE_PREFERENCE: CCRCC's original choice came from a candidate-gene
    # search that was lost, and a tie in pick_test_sample's score picks a different sample
    # non-deterministically.
    PINNED = {
        "IDC": ("TENX99", "GATA3"),
        "COAD": ("TENX148", "CEACAM6"),
        "CCRCC": ("INT15", "STK39"),
    }
    chosen_sample, gene_used = PINNED[cancer]
    gene_idx = genes.index(gene_used)
    print(
        f"[infer] {cancer}: pinned to the same sample/gene as the stored ePathST predictions "
        f"({chosen_sample}/{gene_used})",
        flush=True,
    )

    split_dir = cancer_root / "splits"
    train_df = pd.read_csv(split_dir / "train_0.csv")
    test_df = pd.read_csv(split_dir / "test_0.csv")
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    train_ids = train_df["sample_id"].tolist()
    test_ids = test_df["sample_id"].tolist()
    print(
        f"[infer] {cancer} fold0: {len(train_ids)} train / {len(test_ids)} test samples: {test_ids}",
        flush=True,
    )
    assert (
        chosen_sample in test_ids
    ), f"pinned sample {chosen_sample} not in {cancer} fold0 test set {test_ids}"

    # A8: distilled features and the official manifest, built with the official_hest helpers
    # shared with 22_official_hest_ablation_grid.py.
    cache = SampleCache(cancer_root, genes, {"distilled": embed_distilled}, DEVICE)
    from epathst.data.splits import carve_validation_from_train

    if len(train_ids) >= 2:
        train_ids_s, val_ids_s = carve_validation_from_train(train_ids, 0.15, SEED)
        if not train_ids_s:
            train_ids_s, val_ids_s = train_ids, train_ids
    else:
        train_ids_s, val_ids_s = train_ids, train_ids
    manifest_sample_ids = list(dict.fromkeys(train_ids_s + val_ids_s + test_ids))
    manifest = pd.DataFrame(build_manifest_rows(manifest_sample_ids, cache, all_df))

    ids_train, x_train_d, y_train_raw = gather(train_ids_s, cache, all_df)
    ids_val, x_val_d, y_val_raw = gather(val_ids_s, cache, all_df)
    ids_test, x_test_d, y_test_raw = gather(test_ids, cache, all_df)
    x_train, x_val, x_test = x_train_d["distilled"], x_val_d["distilled"], x_test_d["distilled"]

    std = TrainOnlyStandardizer().fit(y_train_raw, ids=ids_train)
    y_gene_train, y_gene_val, y_gene_test = (
        std.transform(y_train_raw),
        std.transform(y_val_raw),
        std.transform(y_test_raw),
    )
    y_path_train = np.full((len(ids_train), 1), np.nan, dtype=np.float32)
    y_path_val = np.full((len(ids_val), 1), np.nan, dtype=np.float32)
    y_path_test = np.full((len(ids_test), 1), np.nan, dtype=np.float32)

    print(f"[infer] {cancer}: training patch_head (distilled)...", flush=True)
    patch_head = train_patch_head(
        x_train, x_val, y_gene_train, y_path_train, y_gene_val, y_path_val, cfg, DEVICE
    )

    print(f"[infer] {cancer}: training A8 graph-correction model...", flush=True)
    preds, r_metrics, eval_ids_test = run_ablation(
        A8_SPEC,
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
        n_hvg50=len(genes),
        pathway_names=["_unused_dummy_pathway"],
    )
    a8_gene_pred_z = preds["gene"]  # standardized space, aligned to eval_ids_test order

    row_of = {uid: i for i, uid in enumerate(ids_test)}
    order = [row_of[u] for u in eval_ids_test]
    y_gene_test_ord = y_gene_test[order]
    official_r = official_pooled_pearson(y_gene_test_ord, a8_gene_pred_z)
    print(
        f"[infer] {cancer}: A8 official_pooled_pearson (all test samples, all genes) = {official_r:.4f} "
        f"(sanity check vs known official table ~0.40-0.55 IDC / ~0.29-0.32 COAD / ~0.16-0.29 CCRCC per-fold)",
        flush=True,
    )

    # Undo the train-only standardization to get log1p expression units for plotting.
    a8_gene_pred_real = a8_gene_pred_z * (std.sigma_ + std.eps) + std.mu_
    y_gene_test_real = y_gene_test_ord * (std.sigma_ + std.eps) + std.mu_

    # The test fold holds several samples; keep only the chosen sample's spots.
    sample_mask = np.array([uid.split("::")[0] == chosen_sample for uid in eval_ids_test])
    n_sample = sample_mask.sum()
    print(
        f"[infer] {cancer}: {n_sample} real spots belong to chosen sample {chosen_sample}",
        flush=True,
    )

    # H-Optimus-0: raw embeddings -> PCA(256) + Ridge, same recipe as 19_official_hest_benchmark.py.
    print(f"[infer] {cancer}: embedding H-Optimus-0 (train+test) + fitting Ridge...", flush=True)
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
        [("scaler", StandardScaler()), ("pca", PCA(n_components=256, random_state=SEED))]
    )
    Xh_train_p = pipe.fit_transform(Xh_train)
    Xh_test_p = pipe.transform(Xh_test)
    alpha = 100.0 / (Xh_train_p.shape[1] * yh_train.shape[1])
    reg = Ridge(solver="lsqr", alpha=alpha, random_state=SEED, fit_intercept=False, max_iter=1000)
    reg.fit(Xh_train_p, yh_train)
    hoptimus_pred_real_all = reg.predict(
        Xh_test_p
    )  # already in log1p units; this path does not standardize

    hoptimus_r_official = official_pooled_pearson(yh_test, hoptimus_pred_real_all)
    print(
        f"[infer] {cancer}: H-Optimus-0-Ridge official_pooled_pearson (all test samples, all genes) = "
        f"{hoptimus_r_official:.4f} (first real H-Optimus-0 measurement, no prior reference table)",
        flush=True,
    )

    hoptimus_sample_mask = np.array([u.split("::")[0] == chosen_sample for u in uids_test_hoptimus])
    assert (
        hoptimus_sample_mask.sum() == n_sample
    ), "A8 vs H-Optimus-0 spot count mismatch for chosen sample"

    # Put both models' predictions in the same spot order (matched by barcode).
    a8_uids_sample = [eval_ids_test[i] for i in range(len(eval_ids_test)) if sample_mask[i]]
    hoptimus_uids_sample = list(uids_test_hoptimus[hoptimus_sample_mask])
    assert sorted(a8_uids_sample) == sorted(
        hoptimus_uids_sample
    ), "spot_uid sets differ between A8 and H-Optimus-0 paths"
    hoptimus_row_of = {u: i for i, u in enumerate(uids_test_hoptimus)}
    reindex = [
        hoptimus_row_of[u] for u in a8_uids_sample
    ]  # put H-Optimus-0 rows in A8's own sample order

    # Spot coordinates in um, same convention as get_mpp/coords_um in the 22_ ablation grid.
    sample_row = row_for_sample(chosen_sample, all_df)
    entry = cache.get(chosen_sample, sample_row["patches_path"], sample_row["expr_path"])
    bc_to_idx = {bc: i for i, bc in enumerate(entry["barcodes"])}
    spot_order_in_sample = [bc_to_idx[u.split("::")[1]] for u in a8_uids_sample]
    coords_um_sample = entry["coords_um"][spot_order_in_sample]

    gt_gene = y_gene_test_real[sample_mask, gene_idx]
    a8_pred_gene = a8_gene_pred_real[sample_mask, gene_idx]
    hoptimus_pred_gene = hoptimus_pred_real_all[reindex, gene_idx]

    r_a8_sample, _ = pearsonr(gt_gene, a8_pred_gene)
    r_hoptimus_sample, _ = pearsonr(gt_gene, hoptimus_pred_gene)
    print(
        f"[infer] {cancer}: REAL per-sample ({chosen_sample}), gene={gene_used}: "
        f"r_A8={r_a8_sample:.4f}  r_HOptimus0={r_hoptimus_sample:.4f}",
        flush=True,
    )

    out = {
        "cancer": cancer,
        "sample": chosen_sample,
        "gene": gene_used,
        "coords_um": coords_um_sample,
        "ground_truth": gt_gene,
        "a8_pred": a8_pred_gene,
        "hoptimus_pred": hoptimus_pred_gene,
        "r_a8_sample": float(r_a8_sample),
        "r_hoptimus_sample": float(r_hoptimus_sample),
        "r_a8_official_all_test": float(official_r),
        "r_hoptimus_official_all_test": float(hoptimus_r_official),
    }
    np.savez(
        OUT_DIR / f"{cancer}_infer.npz",
        **{k: v for k, v in out.items() if isinstance(v, np.ndarray)},
    )
    with open(OUT_DIR / f"{cancer}_infer_meta.json", "w") as f:
        json.dump({k: v for k, v in out.items() if not isinstance(v, np.ndarray)}, f, indent=2)
    return out


results = {}
for cancer in ["IDC", "COAD", "CCRCC"]:
    try:
        results[cancer] = run_cancer(cancer)
    except Exception as e:
        import traceback

        print(f"[infer] {cancer} FAILED: {e}", flush=True)
        traceback.print_exc()

print("[infer] ===== SUMMARY =====", flush=True)
for cancer, r in results.items():
    print(
        f"{cancer}: sample={r['sample']} gene={r['gene']} "
        f"r_A8(sample)={r['r_a8_sample']:.4f} r_HOptimus0(sample)={r['r_hoptimus_sample']:.4f} | "
        f"official-all-test: A8={r['r_a8_official_all_test']:.4f} HOptimus0={r['r_hoptimus_official_all_test']:.4f}",
        flush=True,
    )
print("[infer] done", flush=True)
