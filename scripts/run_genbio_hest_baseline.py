#!/usr/bin/env python
"""Standalone GenBio-PathFM patch-only baseline for the HEST IDC/COAD/CCRCC
downstream experiment: frozen GenBio-PathFM (the same teacher used for
CNN/GNN distillation, sts.teacher_genbio.GenBioPathFMTeacher) embeds every
HEST H&E ST spot once (4608-D CLS representation, cached to disk,
resumable), then a small trainable projection+prediction head (4608 -> 320
-> 128 -> {HVG1000, PROGENy14}) is fit on TRAIN-patient spots only, using
the EXACT existing seed/fold patient split the CNN/GNN experiment
(scripts/10_run_patient_cv.py) uses. HVG50 is reported as a strict column
slice of the HVG1000 predictions/targets -- no separate model. GenBio
itself is never fine-tuned and no graph/GNN of any kind is built here; this
is a patch-only pathology-foundation-model baseline for comparison against
the CNN/GNN pipeline, not a replacement for it.

Reuses (imports, never modifies): sts.teacher_genbio.GenBioPathFMTeacher
(frozen GenBio-PathFM loader -- the exact HF repo/dtype/embedding-dim
config.teacher.* used as the CPTAC distillation teacher), sts.expression
(normalize_log1p_slide / TrainOnlyStandardizer / select_hvg_for_fold),
sts.pathways (fetch_and_cache_progeny_net / compute_progeny_scores,
identical PROGENy14 top-100-weighted-target setting), sts.cv_folds
(build_cv_grid / carve_validation_from_train -- deterministic GroupKFold,
no RNG, so recomputing it from the same hest_samples.csv patient list that
scripts/10_run_patient_cv.py itself derives its split from is bit-identical
to loading a persisted split), sts.patch_head.patch_head_loss (identical
gene-only-by-default loss), sts.metrics (patient-level Fisher-z Pearson
aggregation), sts.utils (config/paths/env-capture conventions), and
scripts/10_run_patient_cv.py's own `load_fold_expression` helper (imported
via importlib from the numerically-named module -- never copy-pasted).
Reads (never writes) scripts/08_prepare_hest_metadata.py's outputs
(manifests/hest_samples.csv, manifests/hest_spot_manifest.csv,
<work_dir>/hest_patches/*.h5) -- the exact same 224x224 @ 0.5um/px ST-spot-
centered patches / alignment QC the CNN/GNN experiment uses. Writes only to
its own new output locations: <persist_dir>/genbio_baseline_embeddings/
(GenBio embedding cache, analogous to but separate from
student_embeddings/) and results/genbio_baseline/ (summary + per-gene/
per-pathway metrics + saved HVG lists + saved splits). Never touches
student_checkpoints/, student_embeddings/, results/cv_ablation_metrics.csv,
hest_patches/, or manifests/ -- those belong to the concurrently-running
CNN/GNN job."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts.cv_folds import build_cv_grid, carve_validation_from_train, exclude_targeted_panel_samples
from sts.expression import TrainOnlyStandardizer, select_hvg_for_fold
from sts.metrics import patient_macro_fisher_z, per_patient_per_column_pearson, target_summary
from sts.patch_head import patch_head_loss
from sts.teacher_genbio import GenBioPathFMTeacher
from sts.utils import RunEnvironment, load_config, resolve_run_paths, save_json, seed_everything

GENBIO_EMBED_DIM = 4608
HIDDEN1, HIDDEN2 = 320, 128


# ------------------------------------------------------------------ import scripts/10's
# load_fold_expression WITHOUT copy-pasting it and WITHOUT modifying that file --
# `10_run_patient_cv.py` is not a valid module name for a normal `import`
# statement (leading digit), so it is loaded by file path instead. The module's
# own `if __name__ == "__main__": main()` guard means executing it as an
# import has no side effects here.
def _import_scripts10():
    path = Path(__file__).resolve().parent / "10_run_patient_cv.py"
    spec = importlib.util.spec_from_file_location("_scripts10_run_patient_cv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- model
class GenBioDownstreamHead(nn.Module):
    """Frozen-GenBio-embedding -> {HVG1000, PROGENy14} predictor. Only this
    module's parameters ever receive gradients; GenBio itself is never in
    the autograd graph (embeddings are loaded from disk as plain tensors)."""

    def __init__(self, gene_dim: int, pathway_dim: int, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(GENBIO_EMBED_DIM, HIDDEN1), nn.LayerNorm(HIDDEN1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(HIDDEN1, HIDDEN2), nn.GELU(),
        )
        self.gene_head = nn.Linear(HIDDEN2, gene_dim)
        self.pathway_head = nn.Linear(HIDDEN2, pathway_dim)

    def forward(self, h: torch.Tensor) -> dict:
        z = self.trunk(h)
        return {"latent": z, "gene": self.gene_head(z), "pathway": self.pathway_head(z)}


# --------------------------------------------------------------- GenBio feature extraction
def _valid_cache(path: Path, expected_n_spots: int) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            emb = f["embedding"]
            if emb.shape[0] != expected_n_spots or emb.shape[1] != GENBIO_EMBED_DIM:
                return False
            e = emb[:].astype(np.float32)
            return bool(np.isfinite(e).all())
    except Exception:
        return False


def extract_genbio_embeddings_for_sample(genbio: GenBioPathFMTeacher, sample_row: pd.Series, spot_manifest: pd.DataFrame,
                                          patch_shard_dir: Path, out_path: Path, batch_size: int) -> dict:
    """Runs the frozen GenBio-PathFM teacher once over every ST spot's
    already-extracted 224x224@0.5um/px patch (scripts/08's patch shard) and
    caches the 4608-D CLS embedding per spot. Returns timing/throughput
    stats. Resumable at the call site via `_valid_cache`."""
    sample_id = sample_row["sample_id"]
    shard_path = patch_shard_dir / f"{sample_id}.h5"
    with h5py.File(shard_path, "r") as f:
        shard_spot_ids = f["patch_id"][:].astype(str)
        patches = f["patches"][:]

    sub = spot_manifest[spot_manifest["sample_id"] == sample_id].set_index("spot_id")
    coords = sub.loc[shard_spot_ids, ["x_um", "y_um"]].values.astype(np.float32)

    t0 = time.time()
    embs = []
    with torch.inference_mode():
        for i in range(0, len(shard_spot_ids), batch_size):
            batch = patches[i:i + batch_size]
            with torch.autocast(device_type="cuda" if genbio.device.startswith("cuda") else "cpu", dtype=torch.bfloat16,
                                 enabled=genbio.device.startswith("cuda")):
                embs.append(genbio.embed_batch(batch))
    elapsed = time.time() - t0
    emb = np.concatenate(embs, axis=0)
    assert emb.shape == (len(shard_spot_ids), GENBIO_EMBED_DIM), \
        f"{sample_id}: expected GenBio embedding shape ({len(shard_spot_ids)}, {GENBIO_EMBED_DIM}), got {emb.shape}"
    assert np.isfinite(emb).all(), f"{sample_id}: NaN/Inf in GenBio embeddings"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("cancer", data=np.array([sample_row["cancer"]] * len(shard_spot_ids), dtype=object), dtype=str_dtype)
        f.create_dataset("sample_id", data=np.array([sample_id] * len(shard_spot_ids), dtype=object), dtype=str_dtype)
        f.create_dataset("patient_id", data=np.array([sample_row["patient_id"]] * len(shard_spot_ids), dtype=object), dtype=str_dtype)
        f.create_dataset("spot_id", data=np.array(shard_spot_ids.tolist(), dtype=object), dtype=str_dtype)
        f.create_dataset("x_um", data=coords[:, 0])
        f.create_dataset("y_um", data=coords[:, 1])
        f.create_dataset("embedding", data=emb.astype(np.float16))  # unit-norm CLS features -> fp16 is safe

    return {"sample_id": sample_id, "n_spots": len(shard_spot_ids), "seconds": elapsed,
            "patches_per_sec": len(shard_spot_ids) / elapsed if elapsed > 0 else float("nan")}


def load_cached_genbio_features(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        return {"spot_id": f["spot_id"][:].astype(str), "embedding": f["embedding"][:].astype(np.float32),
                "x_um": f["x_um"][:], "y_um": f["y_um"][:]}


# ------------------------------------------------------------------------ CSV read w/ retry
def _read_csv_with_retry(path: Path, retries: int = 5, delay: float = 2.0) -> pd.DataFrame:
    """manifests/hest_samples.csv and hest_spot_manifest.csv are periodically
    rewritten in full by the concurrently-running scripts/08 job (job 84327/
    84331) -- a read landing mid-rewrite can see a truncated/partial CSV.
    Retries a few times on parse failure rather than crashing this
    independent, read-only job."""
    last_err = None
    for _ in range(retries):
        try:
            return pd.read_csv(path)
        except Exception as e:  # pragma: no cover - concurrent-write race
            last_err = e
            time.sleep(delay)
    raise last_err


# --------------------------------------------------------------- wait for scripts/08 to finish
def wait_for_hest_manifest_complete(cfg: dict, paths: dict, poll_seconds: float = 60.0,
                                     stable_checks: int = 3, timeout_seconds: float = 20 * 3600) -> None:
    """scripts/08 (job 84327/84331) writes manifests/hest_samples.csv and
    hest_spot_manifest.csv in full at every checkpoint, but partway through
    its run only a subset of cancers (e.g. IDC first, then COAD, then
    CCRCC) have any QC-passed rows yet. Reading the manifest before that job
    finishes Step 8 would silently give this script an incomplete,
    single-cancer view. Polls until every cfg['cancers']['hest'] cancer has
    at least one QC-eligible sample AND the total eligible-sample count is
    unchanged across `stable_checks` consecutive polls (i.e. scripts/08 has
    stopped adding rows), then returns. Read-only -- never touches
    scripts/08's files. `timeout_seconds` only lets this proceed early once
    every cancer has at least one sample but the count hasn't gone stable yet
    (scripts/08 still trickling in more of the same cancers) -- it never
    proceeds while a whole cancer is still completely missing, since the
    downstream smoke test hard-requires a sample from every cancer and would
    just crash. If a cancer never appears this waits until the SLURM
    wall-clock kills the job, which is a clean failure rather than a wasted
    silent partial run."""
    expected_cancers = set(cfg["cancers"]["hest"])
    samples_path = paths["manifests"] / "hest_samples.csv"
    manifest_path = paths["manifests"] / "hest_spot_manifest.csv"
    start = time.monotonic()
    last_n, stable_streak = -1, 0
    while True:
        hest_samples = _read_csv_with_retry(samples_path)
        spot_manifest = _read_csv_with_retry(manifest_path)
        eligible_ids = set(spot_manifest["sample_id"].unique())
        eligible = hest_samples[hest_samples["sample_id"].isin(eligible_ids)]
        seen_cancers = set(eligible["cancer"].unique())
        n = len(eligible)
        missing = expected_cancers - seen_cancers
        stable_streak = stable_streak + 1 if (not missing and n == last_n) else 0
        print(f"[run_genbio_hest_baseline] waiting for scripts/08 manifest: {n} QC-eligible samples, "
              f"cancers seen={sorted(seen_cancers)}, missing={sorted(missing)}, "
              f"stable_streak={stable_streak}/{stable_checks}", flush=True)
        if not missing and stable_streak >= stable_checks:
            print(f"[run_genbio_hest_baseline] manifest complete: {n} samples across all "
                  f"{sorted(expected_cancers)}, stable across {stable_checks} checks -- proceeding", flush=True)
            return
        if not missing and time.monotonic() - start > timeout_seconds:
            print(f"[run_genbio_hest_baseline] WARNING: manifest wait timed out after {timeout_seconds:.0f}s "
                  f"before going stable (all cancers present, count still {n}) -- proceeding anyway", flush=True)
            return
        last_n = n
        time.sleep(poll_seconds)


# --------------------------------------------------------------------------- CV split reuse
def load_or_recompute_split(cfg: dict, paths: dict, hest_samples: pd.DataFrame, cancer: str, seed: int, fold: int) -> tuple[dict, str]:
    """Requirement: reuse the EXACT seed/fold patient split
    scripts/10_run_patient_cv.py uses -- never regenerate a new random
    split. Prefers loading the persisted manifests/hest_cv_folds.json that
    job 84327/84331's Step 10 writes (paths["manifests"] / "hest_cv_folds.json",
    see scripts/10_run_patient_cv.py's `save_json(cv_grid, ...)`); if that
    job hasn't reached Step 10 yet, falls back to calling the identical
    `sts.cv_folds.build_cv_grid` deterministically (GroupKFold has no RNG --
    it depends only on the sorted patient-id list) on the CURRENT
    manifests/hest_samples.csv. hest_samples.csv is written in full (not
    incrementally) at every one of Step 8's checkpoints, so its patient set
    is already stable well before Step 8 finishes extracting every sample's
    patches -- once Step 10 does run, it will derive the identical
    patients_by_cancer from the same stable file and therefore produce a
    bit-identical fold 0, so this fallback is not an approximation, just an
    earlier evaluation of the same deterministic function."""
    cv_path = paths["manifests"] / "hest_cv_folds.json"
    if cv_path.exists():
        cv_grid = json.loads(cv_path.read_text())
        source = "loaded_existing_hest_cv_folds_json"
    else:
        patients_by_cancer = {c: hest_samples[hest_samples["cancer"] == c]["patient_id"].unique().tolist()
                               for c in cfg["cancers"]["hest"]}
        cv_grid = build_cv_grid(patients_by_cancer, cfg)
        source = "recomputed_deterministic_build_cv_grid_fallback"
        print(f"[run_genbio_hest_baseline] WARNING: {cv_path} not found yet (CNN/GNN job hasn't reached Step 10) -- "
              f"recomputed the identical deterministic split via sts.cv_folds.build_cv_grid instead of loading it.")
    fold_info = cv_grid["per_cancer"][cancer]["folds"][fold]
    assert fold_info["fold"] == fold
    cv_seeds = cv_grid.get("seeds", cfg["cross_validation"]["seeds"])
    assert seed in cv_seeds, f"seed {seed} not in this experiment's configured seeds {cv_seeds}"
    return fold_info, source


# --------------------------------------------------------------------------- smoke test
def run_smoke_test(cfg: dict, paths: dict, genbio: GenBioPathFMTeacher, embed_dir: Path, patch_shard_dir: Path,
                    hest_samples: pd.DataFrame, spot_manifest: pd.DataFrame, batch_size: int) -> None:
    print("[smoke-test] running 1 sample per cancer ...")
    scripts10 = _import_scripts10()
    for cancer in cfg["cancers"]["hest"]:
        available = hest_samples[(hest_samples["cancer"] == cancer)]
        available = available[available["sample_id"].isin(set(spot_manifest["sample_id"]))]
        assert len(available) > 0, f"[smoke-test] no patch-extracted sample available yet for {cancer}"
        row = available.iloc[0]
        sample_id = row["sample_id"]
        out_path = embed_dir / f"{sample_id}.h5"
        n_spots = int((spot_manifest["sample_id"] == sample_id).sum())

        stats = extract_genbio_embeddings_for_sample(genbio, row, spot_manifest, patch_shard_dir, out_path, batch_size)
        print(f"[smoke-test] {cancer}/{sample_id}: extracted {stats['n_spots']} spots in {stats['seconds']:.1f}s "
              f"({stats['patches_per_sec']:.1f} patches/sec)")

        reloaded = load_cached_genbio_features(out_path)
        assert reloaded["embedding"].shape[1] == GENBIO_EMBED_DIM
        assert reloaded["embedding"].shape[0] == n_spots
        assert np.isfinite(reloaded["embedding"]).all()
        assert set(reloaded["spot_id"]) == set(spot_manifest[spot_manifest["sample_id"] == sample_id]["spot_id"])
        print(f"[smoke-test] {cancer}/{sample_id}: cache reload OK, {reloaded['embedding'].shape}")

        adata_path = row["expression_path"]
        import anndata as ad
        adata = ad.read_h5ad(adata_path)
        expr_spot_ids = set(adata.obs_names.astype(str))
        overlap = expr_spot_ids & set(reloaded["spot_id"])
        assert len(overlap) > 0, f"[smoke-test] {sample_id}: GenBio spot_ids don't overlap expression spot_ids at all"
        print(f"[smoke-test] {cancer}/{sample_id}: {len(overlap)}/{len(reloaded['spot_id'])} spot_ids "
              f"align with ST expression spot IDs")

        head = GenBioDownstreamHead(gene_dim=1000, pathway_dim=14).to(genbio.device)
        opt = torch.optim.AdamW(head.parameters(), lr=cfg["patch_head"]["lr"])
        n = min(8, reloaded["embedding"].shape[0])
        xb = torch.tensor(reloaded["embedding"][:n], dtype=torch.float32, device=genbio.device)
        gb = torch.zeros(n, 1000, device=genbio.device)
        pb = torch.zeros(n, 14, device=genbio.device)
        opt.zero_grad()
        out = head(xb)
        loss = patch_head_loss(out, gb, pb, cfg["patch_head"]["pathway_loss_weight"])
        loss["loss"].backward()
        grad_norms = [p.grad.norm().item() for p in head.parameters() if p.grad is not None]
        assert len(grad_norms) > 0 and all(g == g for g in grad_norms), f"[smoke-test] head backward pass produced no/NaN gradients"
        opt.step()
        print(f"[smoke-test] {cancer}/{sample_id}: downstream head forward/backward OK "
              f"({len(grad_norms)} grad tensors, mean |grad|={np.mean(grad_norms):.4g})")
    print("[smoke-test] PASSED for all 3 cancers.\n")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--extract-only", action="store_true", help="only run GenBio feature extraction, then exit")
    ap.add_argument("--train-only", action="store_true", help="skip extraction, assume embeddings already cached")
    ap.add_argument("--skip-smoke-test", action="store_true", help="skip the 1-sample-per-cancer smoke test")
    ap.add_argument("--smoke-test-only", action="store_true", help="run only the smoke test, then exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    persist_dir = Path(cfg["paths"].get("persist_dir") or cfg["paths"]["work_dir"])
    embed_dir = persist_dir / "genbio_baseline_embeddings"
    patch_shard_dir = Path(cfg["paths"]["work_dir"]) / "hest_patches"  # read-only: owned by scripts/08
    results_dir = paths["results"] / "genbio_baseline"
    hvg_lists_dir = results_dir / "hvg_lists"
    splits_dir = results_dir / "splits_used"
    for d in (embed_dir, results_dir, hvg_lists_dir, splits_dir):
        d.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cap = torch.cuda.get_device_capability(0)
        print(f"[run_genbio_hest_baseline] GPU={torch.cuda.get_device_name(0)} capability={cap} "
              f"bf16_supported={torch.cuda.is_bf16_supported()}")

    wait_for_hest_manifest_complete(cfg, paths)
    hest_samples = _read_csv_with_retry(paths["manifests"] / "hest_samples.csv")
    spot_manifest = _read_csv_with_retry(paths["manifests"] / "hest_spot_manifest.csv")
    # "eligible" (requirement 4): passed scripts/08's alignment QC, i.e. has a real patch shard + spot_manifest rows.
    eligible_sample_ids = set(spot_manifest["sample_id"].unique())
    hest_samples = hest_samples[hest_samples["sample_id"].isin(eligible_sample_ids)].reset_index(drop=True)
    hest_samples = exclude_targeted_panel_samples(hest_samples)
    print(f"[run_genbio_hest_baseline] {len(hest_samples)} QC-eligible HEST samples across "
          f"{hest_samples['cancer'].value_counts().to_dict()}")

    print("[run_genbio_hest_baseline] loading frozen GenBio-PathFM teacher "
          f"({cfg['teacher']['hf_repo']}, dtype={cfg['teacher']['dtype']}, embedding_dim={cfg['teacher']['embedding_dim']}) ...")
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    genbio = GenBioPathFMTeacher(cfg, device=device)
    assert cfg["teacher"]["embedding_dim"] == GENBIO_EMBED_DIM
    for p in genbio.model.parameters():
        assert not p.requires_grad, "GenBio-PathFM must be fully frozen"
    assert not genbio.model.training, "GenBio-PathFM must be in eval() mode"
    print(f"[run_genbio_hest_baseline] GenBio-PathFM loaded, frozen, {genbio.n_parameters / 1e9:.2f}B params")

    if not args.skip_smoke_test:
        run_smoke_test(cfg, paths, genbio, embed_dir, patch_shard_dir, hest_samples, spot_manifest,
                        batch_size=cfg["teacher"]["batch_size"])
        if args.smoke_test_only:
            print("[run_genbio_hest_baseline] --smoke-test-only: done.")
            return

    # ------------------------------------------------------------ Step: GenBio feature extraction (resumable)
    extraction_stats_by_cancer = {}
    if not args.train_only:
        with RunEnvironment("genbio_baseline_extraction", paths["logs"] / "genbio_baseline_extraction_env.json"):
            for cancer in cfg["cancers"]["hest"]:
                samples = hest_samples[hest_samples["cancer"] == cancer]
                t0 = time.time()
                n_total_spots, n_extracted_now = 0, 0
                for _, row in samples.iterrows():
                    sample_id = row["sample_id"]
                    n_spots = int((spot_manifest["sample_id"] == sample_id).sum())
                    n_total_spots += n_spots
                    out_path = embed_dir / f"{sample_id}.h5"
                    if _valid_cache(out_path, n_spots):
                        continue  # resumable: already-cached sample, skip
                    extract_genbio_embeddings_for_sample(genbio, row, spot_manifest, patch_shard_dir, out_path,
                                                          batch_size=cfg["teacher"]["batch_size"])
                    n_extracted_now += 1
                elapsed = time.time() - t0
                extraction_stats_by_cancer[cancer] = {"n_samples": len(samples), "n_spots": n_total_spots,
                                                       "n_newly_extracted_samples": n_extracted_now,
                                                       "seconds": elapsed}
                print(f"[run_genbio_hest_baseline] {cancer}: {len(samples)} samples / {n_total_spots} spots, "
                      f"{n_extracted_now} newly extracted this run, {elapsed:.1f}s")
        if args.extract_only:
            print("[run_genbio_hest_baseline] --extract-only: done.")
            return

    # ------------------------------------------------------------ Step: downstream head, per cancer
    scripts10 = _import_scripts10()
    hest_root = Path(cfg["paths"]["hest_root"])
    mt = cfg["molecular_targets"]
    spot_by_uid = spot_manifest.assign(spot_uid=spot_manifest["sample_id"] + "::" + spot_manifest["spot_id"]).set_index("spot_uid")

    summary_rows, gene_metric_rows, pathway_metric_rows = [], [], []
    for cancer in cfg["cancers"]["hest"]:
        per_cancer_json = results_dir / "per_cancer" / f"{cancer}_seed{args.seed}_fold{args.fold}.json"
        if per_cancer_json.exists():
            print(f"[run_genbio_hest_baseline] {cancer}: per-cancer result already present at {per_cancer_json}, reusing.")
            cached = json.loads(per_cancer_json.read_text())
            summary_rows.append(cached["summary"])
            gene_metric_rows.extend(cached["gene_metrics"])
            pathway_metric_rows.extend(cached["pathway_metrics"])
            continue

        fold_info, split_source = load_or_recompute_split(cfg, paths, hest_samples, cancer, args.seed, args.fold)
        train_patients_all, test_patients = fold_info["train_patients"], fold_info["test_patients"]
        assert set(train_patients_all).isdisjoint(test_patients), \
            f"{cancer}: train/test patient overlap! {set(train_patients_all) & set(test_patients)}"
        train_p, val_p = carve_validation_from_train(train_patients_all, 0.15, args.seed)
        assert set(train_p).isdisjoint(test_patients) and set(val_p).isdisjoint(test_patients), \
            f"{cancer}: test patients leaked into train/val carve"

        save_json({"cancer": cancer, "seed": args.seed, "fold": args.fold, "source": split_source,
                    "train_patients": sorted(train_p), "val_patients": sorted(val_p),
                    "test_patients": sorted(test_patients)},
                   splits_dir / f"{cancer}_seed{args.seed}_fold{args.fold}.json")

        cancer_samples = hest_samples[hest_samples["cancer"] == cancer]

        def load_expr(patients):
            expr, progeny, genes_by_sample, net = scripts10.load_fold_expression(cancer_samples, spot_manifest,
                                                                                   hest_root, patients, cfg)
            return expr, progeny, genes_by_sample, net

        train_expr, train_progeny, genes_by_sample, net = load_expr(train_p)
        pathway_names = list(net["source"].unique())
        assert len(pathway_names) == 14, f"expected 14 PROGENy pathways, got {len(pathway_names)}"
        common_genes = sorted(set.intersection(*genes_by_sample.values())) if genes_by_sample else []

        hvg = select_hvg_for_fold(train_expr, common_genes, mt["hvg1000"], mt["hvg50"])
        assert set(hvg["hvg50"]).issubset(set(hvg["hvg1000"])), "HVG50 must be a subset of HVG1000"
        save_json(hvg["hvg1000"], hvg_lists_dir / f"{cancer}_hvg1000.json")
        save_json(hvg["hvg50"], hvg_lists_dir / f"{cancer}_hvg50.json")

        gene_std = TrainOnlyStandardizer().fit(train_expr[hvg["hvg1000"]].values, ids=train_expr.index)
        path_std = TrainOnlyStandardizer().fit(train_progeny.values, ids=train_progeny.index)

        def build_targets(patients, is_train):
            expr, progeny, _, _ = load_expr(patients)
            if not is_train:
                gene_std.assert_no_leakage(expr.index)  # test/val patients must never overlap the fit set
            gene_t = gene_std.transform(expr[hvg["hvg1000"]].reindex(columns=hvg["hvg1000"]).values)
            path_t = path_std.transform(progeny.reindex(columns=pathway_names).values)
            assert np.isfinite(gene_t).all() and np.isfinite(path_t).all(), \
                f"{cancer}: NaN/Inf in standardized targets for {'train' if is_train else 'eval'} patients"
            return gene_t, path_t, expr.index.astype(str)

        y_gene_train, y_path_train, ids_train = build_targets(train_p, is_train=True)
        y_gene_val, y_path_val, ids_val = build_targets(val_p, is_train=False)
        y_gene_test, y_path_test, ids_test = build_targets(test_patients, is_train=False)

        feat_cache = {}

        def load_feats(spot_uids):
            out = []
            for uid in spot_uids:
                sample_id, bare_spot_id = spot_by_uid.loc[uid, ["sample_id", "spot_id"]]
                if sample_id not in feat_cache:
                    feat_cache[sample_id] = load_cached_genbio_features(embed_dir / f"{sample_id}.h5")
                c = feat_cache[sample_id]
                pos = list(c["spot_id"]).index(bare_spot_id)
                out.append(c["embedding"][pos])
            return np.stack(out)

        x_train, x_val, x_test = load_feats(ids_train), load_feats(ids_val), load_feats(ids_test)
        for name, x in (("train", x_train), ("val", x_val), ("test", x_test)):
            assert x.shape[1] == GENBIO_EMBED_DIM
            assert np.isfinite(x).all(), f"{cancer}: NaN/Inf in {name} GenBio embeddings"

        print(f"[run_genbio_hest_baseline] {cancer}: train={len(train_p)}pat/{len(ids_train)}spots "
              f"val={len(val_p)}pat/{len(ids_val)}spots test={len(test_patients)}pat/{len(ids_test)}spots "
              f"(split source: {split_source})")

        seed_everything(args.seed)
        head = GenBioDownstreamHead(gene_dim=len(hvg["hvg1000"]), pathway_dim=14,
                                     dropout=cfg["patch_head"].get("dropout", 0.1)).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=cfg["patch_head"]["lr"], weight_decay=cfg["patch_head"]["weight_decay"])
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_train0 = time.time()
        best_val, patience_ctr, best_state = float("inf"), 0, None
        for epoch in range(cfg["patch_head"]["max_epochs"]):
            head.train()
            xb = torch.tensor(x_train, dtype=torch.float32, device=device)
            gb = torch.tensor(y_gene_train, dtype=torch.float32, device=device)
            pb = torch.tensor(y_path_train, dtype=torch.float32, device=device)
            opt.zero_grad()
            loss = patch_head_loss(head(xb), gb, pb, cfg["patch_head"]["pathway_loss_weight"])
            loss["loss"].backward()
            opt.step()

            head.eval()
            with torch.no_grad():
                xv = torch.tensor(x_val, dtype=torch.float32, device=device)
                gv = torch.tensor(y_gene_val, dtype=torch.float32, device=device)
                pv = torch.tensor(y_path_val, dtype=torch.float32, device=device)
                val_loss = patch_head_loss(head(xv), gv, pv, cfg["patch_head"]["pathway_loss_weight"])["loss"].item()
            if val_loss < best_val:
                best_val, patience_ctr, best_state = val_loss, 0, {k: v.clone() for k, v in head.state_dict().items()}
            else:
                patience_ctr += 1
            if patience_ctr >= cfg["patch_head"]["early_stop_patience"]:
                break
        head.load_state_dict(best_state)
        head_training_seconds = time.time() - t_train0
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")

        head.eval()
        with torch.no_grad():
            out = head(torch.tensor(x_test, dtype=torch.float32, device=device))
        gene_pred, path_pred = out["gene"].cpu().numpy(), out["pathway"].cpu().numpy()

        test_patient_ids = spot_by_uid.loc[ids_test, "patient_id"].values
        n_hvg50 = len(hvg["hvg50"])

        pp_hvg1000 = per_patient_per_column_pearson(y_gene_test, gene_pred, test_patient_ids, hvg["hvg1000"],
                                                      cfg["evaluation"]["min_regions_per_patient"])
        pp_hvg50 = per_patient_per_column_pearson(y_gene_test[:, :n_hvg50], gene_pred[:, :n_hvg50], test_patient_ids,
                                                    hvg["hvg50"], cfg["evaluation"]["min_regions_per_patient"])
        pp_pathway = per_patient_per_column_pearson(y_path_test, path_pred, test_patient_ids, pathway_names,
                                                      cfg["evaluation"]["min_regions_per_patient"])

        r_hvg1000 = target_summary(pp_hvg1000, tuple(cfg["evaluation"]["gene_r_thresholds"]))["fisher_z_mean_patient_R"]
        r_hvg50 = target_summary(pp_hvg50, tuple(cfg["evaluation"]["gene_r_thresholds"]))["fisher_z_mean_patient_R"]
        r_pathway = target_summary(pp_pathway, tuple(cfg["evaluation"]["gene_r_thresholds"]))["fisher_z_mean_patient_R"]

        pp_hvg1000.insert(0, "cancer", cancer)
        pp_pathway.insert(0, "cancer", cancer)
        gene_metric_rows.extend(pp_hvg1000.to_dict("records"))
        pathway_metric_rows.extend(pp_pathway.to_dict("records"))

        summary_row = {
            "cancer": cancer, "n_train_patients": len(train_p), "n_test_patients": len(test_patients),
            "n_train_spots": len(ids_train), "n_test_spots": len(ids_test),
            "HVG50_r": r_hvg50, "HVG1000_r": r_hvg1000, "PROGENy14_r": r_pathway,
            # mean_mse/mean_mae: same per-patient-per-column mse/mae per_patient_per_column_pearson already
            # computes (mirrors sts.metrics.evaluate_predictions's mean_mse/mean_mae, which target_summary
            # itself doesn't surface -- added per explicit user request to keep alongside Pearson r for
            # every target block, matching what cv_ablation_metrics.csv already has for the CNN/GNN run).
            "HVG50_mse": float(pp_hvg50["mse"].mean()) if len(pp_hvg50) else float("nan"),
            "HVG50_mae": float(pp_hvg50["mae"].mean()) if len(pp_hvg50) else float("nan"),
            "HVG1000_mse": float(pp_hvg1000["mse"].mean()) if len(pp_hvg1000) else float("nan"),
            "HVG1000_mae": float(pp_hvg1000["mae"].mean()) if len(pp_hvg1000) else float("nan"),
            "PROGENy14_mse": float(pp_pathway["mse"].mean()) if len(pp_pathway) else float("nan"),
            "PROGENy14_mae": float(pp_pathway["mae"].mean()) if len(pp_pathway) else float("nan"),
            "feature_extraction_time": extraction_stats_by_cancer.get(cancer, {}).get("seconds"),
            "head_training_time": head_training_seconds, "peak_vram": peak_vram_gb,
        }
        summary_rows.append(summary_row)

        per_cancer_json.parent.mkdir(parents=True, exist_ok=True)
        save_json({"summary": summary_row, "gene_metrics": pp_hvg1000.to_dict("records"),
                    "pathway_metrics": pp_pathway.to_dict("records")}, per_cancer_json)
        print(f"[run_genbio_hest_baseline] {cancer}: HVG50_r={r_hvg50:.4f} HVG1000_r={r_hvg1000:.4f} "
              f"PROGENy14_r={r_pathway:.4f} (head_training={head_training_seconds:.1f}s peak_vram={peak_vram_gb:.2f}GB)")

    # ------------------------------------------------------------ save & print final results
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "genbio_summary.csv", index=False)
    pd.DataFrame(gene_metric_rows).to_csv(results_dir / "genbio_gene_metrics.csv", index=False)
    pd.DataFrame(pathway_metric_rows).to_csv(results_dir / "genbio_pathway_metrics.csv", index=False)

    print("\n[run_genbio_hest_baseline] ================= FINAL RESULTS =================")
    print(f"{'Cancer':<8} {'HVG50 r':>10} {'HVG1000 r':>10} {'PROGENy14 r':>12}")
    for _, row in summary_df.iterrows():
        print(f"{row['cancer']:<8} {row['HVG50_r']:>10.4f} {row['HVG1000_r']:>10.4f} {row['PROGENy14_r']:>12.4f}")
    if len(summary_df):
        print(f"{'MEAN':<8} {summary_df['HVG50_r'].mean():>10.4f} {summary_df['HVG1000_r'].mean():>10.4f} "
              f"{summary_df['PROGENy14_r'].mean():>12.4f}")
    print(f"\nn_samples_processed={len(hest_samples)} n_spots_total={spot_manifest.shape[0]}")
    for cancer, stats in extraction_stats_by_cancer.items():
        pps = stats["n_spots"] / stats["seconds"] if stats["seconds"] > 0 else float("nan")
        print(f"  {cancer}: extraction {stats['seconds']:.1f}s over {stats['n_spots']} spots ({pps:.1f} patches/sec)")
    print(f"results written to {results_dir}")


if __name__ == "__main__":
    main()
