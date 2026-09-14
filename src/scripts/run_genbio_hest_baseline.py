#!/usr/bin/env python
"""Standalone H-Optimus-0 patch-only baseline for the HEST IDC/COAD/CCRCC
downstream experiment: the frozen teacher embeds every ST spot once
(cached, resumable), then a small trainable head is fit on the exact
train/fold patient split src/scripts/10_run_patient_cv.py uses, for
comparison against the CNN/GNN pipeline (never fine-tunes H-Optimus-0, builds
no graph, and never writes to that job's own output locations)."""
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
from epathst.data.splits import (
    build_cv_grid,
    carve_validation_from_train,
    exclude_targeted_panel_samples,
)
from epathst.data.expression import TrainOnlyStandardizer, select_hvg_for_fold
from epathst.evaluation.metrics import per_patient_per_column_pearson, target_summary
from epathst.models.patch_head import patch_head_loss
from epathst.models.teacher import HOptimusTeacher
from epathst.utils import (
    RunEnvironment,
    load_config,
    resolve_run_paths,
    save_json,
    seed_everything,
)

TEACHER_EMBED_DIM = 1536
HIDDEN1, HIDDEN2 = 320, 128


# ------------------------------------------------------------------ import src/scripts/10's
# load_fold_expression WITHOUT copy-pasting or modifying that file --
# `10_run_patient_cv.py` is not a valid module name for a normal `import`
# (leading digit), so it's loaded by file path instead. Its own
# `if __name__ == "__main__": main()` guard means this import has no side effects.
def _import_scripts10():
    path = Path(__file__).resolve().parent / "10_run_patient_cv.py"
    spec = importlib.util.spec_from_file_location("_scripts10_run_patient_cv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- model
class HOptimusDownstreamHead(nn.Module):
    """Frozen-H-Optimus-0-embedding -> {HVG1000, PROGENy14} predictor. Only this
    module's parameters ever receive gradients; H-Optimus-0 itself is never in
    the autograd graph (embeddings are loaded from disk as plain tensors)."""

    def __init__(self, gene_dim: int, pathway_dim: int, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(TEACHER_EMBED_DIM, HIDDEN1), nn.LayerNorm(HIDDEN1), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(HIDDEN1, HIDDEN2), nn.GELU(),
        )
        self.gene_head = nn.Linear(HIDDEN2, gene_dim)
        self.pathway_head = nn.Linear(HIDDEN2, pathway_dim)

    def forward(self, h: torch.Tensor) -> dict:
        z = self.trunk(h)
        return {"latent": z, "gene": self.gene_head(z), "pathway": self.pathway_head(z)}


# --------------------------------------------------------------- H-Optimus-0 feature extraction
def _valid_cache(path: Path, expected_n_spots: int) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            emb = f["embedding"]
            if emb.shape[0] != expected_n_spots or emb.shape[1] != TEACHER_EMBED_DIM:
                return False
            e = emb[:].astype(np.float32)
            return bool(np.isfinite(e).all())
    except Exception:
        return False


def extract_hoptimus_embeddings_for_sample(
    teacher_h: HOptimusTeacher, sample_row: pd.Series, spot_manifest: pd.DataFrame,
    patch_shard_dir: Path, out_path: Path, batch_size: int,
) -> dict:
    """Runs the frozen H-Optimus-0 teacher once over every ST spot's
    already-extracted 224x224@0.5um/px patch (src/scripts/08's patch shard) and
    caches the 4608-D CLS embedding per spot. Returns timing/throughput stats;
    resumable at the call site via `_valid_cache`."""
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
            is_cuda = teacher_h.device.startswith("cuda")
            with torch.autocast(
                device_type="cuda" if is_cuda else "cpu", dtype=torch.bfloat16, enabled=is_cuda
            ):
                embs.append(teacher_h.embed_batch(batch))
    elapsed = time.time() - t0
    emb = np.concatenate(embs, axis=0)
    assert emb.shape == (len(shard_spot_ids), TEACHER_EMBED_DIM), (
        f"{sample_id}: expected H-Optimus-0 embedding shape "
        f"({len(shard_spot_ids)}, {TEACHER_EMBED_DIM}), got {emb.shape}"
    )
    assert np.isfinite(emb).all(), f"{sample_id}: NaN/Inf in H-Optimus-0 embeddings"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    n_spots = len(shard_spot_ids)
    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "cancer", data=np.array([sample_row["cancer"]] * n_spots, dtype=object),
            dtype=str_dtype,
        )
        f.create_dataset(
            "sample_id", data=np.array([sample_id] * n_spots, dtype=object), dtype=str_dtype
        )
        f.create_dataset(
            "patient_id", data=np.array([sample_row["patient_id"]] * n_spots, dtype=object),
            dtype=str_dtype,
        )
        f.create_dataset(
            "spot_id", data=np.array(shard_spot_ids.tolist(), dtype=object), dtype=str_dtype
        )
        f.create_dataset("x_um", data=coords[:, 0])
        f.create_dataset("y_um", data=coords[:, 1])
        # unit-norm CLS features -> fp16 is safe
        f.create_dataset("embedding", data=emb.astype(np.float16))

    return {
        "sample_id": sample_id, "n_spots": n_spots, "seconds": elapsed,
        "patches_per_sec": n_spots / elapsed if elapsed > 0 else float("nan"),
    }


def load_cached_hoptimus_features(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        return {
            "spot_id": f["spot_id"][:].astype(str),
            "embedding": f["embedding"][:].astype(np.float32),
            "x_um": f["x_um"][:], "y_um": f["y_um"][:],
        }


# ------------------------------------------------------------------------ CSV read w/ retry
def _read_csv_with_retry(path: Path, retries: int = 5, delay: float = 2.0) -> pd.DataFrame:
    """The HEST manifests are periodically rewritten in full by the
    concurrently-running src/scripts/08 job -- a read landing mid-rewrite can
    see a truncated CSV. Retries on parse failure rather than crashing this
    independent, read-only job."""
    last_err = None
    for _ in range(retries):
        try:
            return pd.read_csv(path)
        except Exception as e:  # pragma: no cover - concurrent-write race
            last_err = e
            time.sleep(delay)
    raise last_err


# --------------------------------------------------------------- wait for src/scripts/08 to finish
def wait_for_hest_manifest_complete(
    cfg: dict, paths: dict, poll_seconds: float = 60.0,
    stable_checks: int = 3, timeout_seconds: float = 20 * 3600,
) -> None:
    """src/scripts/08 rewrites the HEST manifests in full at every checkpoint,
    but partway through its run only a subset of cancers have QC-passed rows
    -- reading before it finishes Step 8 would silently give this script an
    incomplete, single-cancer view. Polls until every configured cancer has
    at least one QC-eligible sample AND the eligible-sample count is
    unchanged across `stable_checks` consecutive polls. `timeout_seconds`
    only lets this proceed early once every cancer is present but the count
    hasn't gone stable -- it never proceeds while a whole cancer is still
    missing (the smoke test hard-requires all cancers and would crash), so a
    cancer that never appears stalls until SLURM kills the job -- a clean
    failure, not a silent partial run. Read-only."""
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
        print(
            f"[run_genbio_hest_baseline] waiting for src/scripts/08 manifest: "
            f"{n} QC-eligible samples, "
            f"cancers seen={sorted(seen_cancers)}, missing={sorted(missing)}, "
            f"stable_streak={stable_streak}/{stable_checks}",
            flush=True,
        )
        if not missing and stable_streak >= stable_checks:
            print(
                f"[run_genbio_hest_baseline] manifest complete: {n} samples across all "
                f"{sorted(expected_cancers)}, stable across {stable_checks} checks -- proceeding",
                flush=True,
            )
            return
        if not missing and time.monotonic() - start > timeout_seconds:
            print(
                f"[run_genbio_hest_baseline] WARNING: manifest wait timed out after "
                f"{timeout_seconds:.0f}s before going stable (all cancers present, count "
                f"still {n}) -- proceeding anyway",
                flush=True,
            )
            return
        last_n = n
        time.sleep(poll_seconds)


# --------------------------------------------------------------------------- CV split reuse
def load_or_recompute_split(
    cfg: dict, paths: dict, hest_samples: pd.DataFrame, cancer: str, seed: int, fold: int,
) -> tuple[dict, str]:
    """Reuses the EXACT seed/fold patient split src/scripts/10_run_patient_cv.py
    uses -- never regenerates a new random split. Prefers loading the
    persisted manifests/hest_cv_folds.json that script 10 writes; if that
    job hasn't reached its Step 10 yet, falls back to calling the identical
    `epathst.data.splits.build_cv_grid` on the current hest_samples.csv. This
    fallback is not an approximation: GroupKFold has no RNG (it depends only
    on the sorted patient-id list), and hest_samples.csv is written in full
    at every checkpoint, so script 10 will later derive the same
    patients_by_cancer from the same stable file and produce a bit-identical
    fold."""
    cv_path = paths["manifests"] / "hest_cv_folds.json"
    if cv_path.exists():
        cv_grid = json.loads(cv_path.read_text())
        source = "loaded_existing_hest_cv_folds_json"
    else:
        patients_by_cancer = {
            c: hest_samples[hest_samples["cancer"] == c]["patient_id"].unique().tolist()
            for c in cfg["cancers"]["hest"]
        }
        cv_grid = build_cv_grid(patients_by_cancer, cfg)
        source = "recomputed_deterministic_build_cv_grid_fallback"
        print(
            f"[run_genbio_hest_baseline] WARNING: {cv_path} not found yet "
            f"(CNN/GNN job hasn't reached Step 10) -- recomputed the identical deterministic "
            f"split via epathst.data.splits.build_cv_grid instead of loading it."
        )
    fold_info = cv_grid["per_cancer"][cancer]["folds"][fold]
    assert fold_info["fold"] == fold
    cv_seeds = cv_grid.get("seeds", cfg["cross_validation"]["seeds"])
    assert seed in cv_seeds, f"seed {seed} not in this experiment's configured seeds {cv_seeds}"
    return fold_info, source


# --------------------------------------------------------------------------- smoke test
def run_smoke_test(
    cfg: dict, paths: dict, teacher_h: HOptimusTeacher, embed_dir: Path, patch_shard_dir: Path,
    hest_samples: pd.DataFrame, spot_manifest: pd.DataFrame, batch_size: int,
) -> None:
    print("[smoke-test] running 1 sample per cancer ...")
    for cancer in cfg["cancers"]["hest"]:
        available = hest_samples[(hest_samples["cancer"] == cancer)]
        available = available[available["sample_id"].isin(set(spot_manifest["sample_id"]))]
        assert len(available) > 0, f"[smoke-test] no patch-extracted sample available yet for {cancer}"
        row = available.iloc[0]
        sample_id = row["sample_id"]
        out_path = embed_dir / f"{sample_id}.h5"
        n_spots = int((spot_manifest["sample_id"] == sample_id).sum())

        stats = extract_hoptimus_embeddings_for_sample(
            teacher_h, row, spot_manifest, patch_shard_dir, out_path, batch_size
        )
        print(
            f"[smoke-test] {cancer}/{sample_id}: extracted {stats['n_spots']} spots "
            f"in {stats['seconds']:.1f}s ({stats['patches_per_sec']:.1f} patches/sec)"
        )

        reloaded = load_cached_hoptimus_features(out_path)
        assert reloaded["embedding"].shape[1] == TEACHER_EMBED_DIM
        assert reloaded["embedding"].shape[0] == n_spots
        assert np.isfinite(reloaded["embedding"]).all()
        expected_spot_ids = set(spot_manifest[spot_manifest["sample_id"] == sample_id]["spot_id"])
        assert set(reloaded["spot_id"]) == expected_spot_ids
        print(f"[smoke-test] {cancer}/{sample_id}: cache reload OK, {reloaded['embedding'].shape}")

        adata_path = row["expression_path"]
        import anndata as ad
        adata = ad.read_h5ad(adata_path)
        expr_spot_ids = set(adata.obs_names.astype(str))
        overlap = expr_spot_ids & set(reloaded["spot_id"])
        assert len(overlap) > 0, (
            f"[smoke-test] {sample_id}: H-Optimus-0 spot_ids don't overlap expression spot_ids at all"
        )
        print(
            f"[smoke-test] {cancer}/{sample_id}: {len(overlap)}/{len(reloaded['spot_id'])} "
            f"spot_ids align with ST expression spot IDs"
        )

        head = HOptimusDownstreamHead(gene_dim=1000, pathway_dim=14).to(teacher_h.device)
        opt = torch.optim.AdamW(head.parameters(), lr=cfg["patch_head"]["lr"])
        n = min(8, reloaded["embedding"].shape[0])
        x_smoke_batch = torch.tensor(
            reloaded["embedding"][:n], dtype=torch.float32, device=teacher_h.device
        )
        gene_dummy_target = torch.zeros(n, 1000, device=teacher_h.device)
        pathway_dummy_target = torch.zeros(n, 14, device=teacher_h.device)
        opt.zero_grad()
        out = head(x_smoke_batch)
        loss = patch_head_loss(
            out, gene_dummy_target, pathway_dummy_target, cfg["patch_head"]["pathway_loss_weight"]
        )
        loss["loss"].backward()
        grad_norms = [p.grad.norm().item() for p in head.parameters() if p.grad is not None]
        assert len(grad_norms) > 0 and all(g == g for g in grad_norms), (
            "[smoke-test] head backward pass produced no/NaN gradients"
        )
        opt.step()
        print(
            f"[smoke-test] {cancer}/{sample_id}: downstream head forward/backward OK "
            f"({len(grad_norms)} grad tensors, mean |grad|={np.mean(grad_norms):.4g})"
        )
    print("[smoke-test] PASSED for all 3 cancers.\n")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument(
        "--extract-only", action="store_true",
        help="only run H-Optimus-0 feature extraction, then exit",
    )
    ap.add_argument(
        "--train-only", action="store_true",
        help="skip extraction, assume embeddings already cached",
    )
    ap.add_argument(
        "--skip-smoke-test", action="store_true",
        help="skip the 1-sample-per-cancer smoke test",
    )
    ap.add_argument(
        "--smoke-test-only", action="store_true", help="run only the smoke test, then exit"
    )
    ap.add_argument(
        "--head", choices=["mlp", "ridge"], default="mlp",
        help="downstream head on the frozen H-Optimus-0 embeddings: 'mlp' (default, this project's "
             "small trained neural head) or 'ridge' (closed-form RidgeCV linear probe, the "
             "standard downstream head most published HEST-1k-style benchmarks use)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    # writes only under embed_dir/results_dir below -- never touches student_checkpoints/,
    # student_embeddings/, results/cv_ablation_metrics.csv, hest_patches/, or manifests/,
    # which belong to the concurrently-running CNN/GNN job (src/scripts/10)
    persist_dir = Path(cfg["paths"].get("persist_dir") or cfg["paths"]["work_dir"])
    embed_dir = persist_dir / "hoptimus0_baseline_embeddings"
    # read-only: owned by src/scripts/08
    patch_shard_dir = Path(cfg["paths"]["work_dir"]) / "hest_patches"
    # separate results dir per --head so a ridge run never clobbers the mlp run's cached results
    results_subdir = "hoptimus0_baseline" if args.head == "mlp" else f"hoptimus0_baseline_{args.head}"
    results_dir = paths["results"] / results_subdir
    hvg_lists_dir = results_dir / "hvg_lists"
    splits_dir = results_dir / "splits_used"
    for d in (embed_dir, results_dir, hvg_lists_dir, splits_dir):
        d.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cap = torch.cuda.get_device_capability(0)
        print(
            f"[run_genbio_hest_baseline] GPU={torch.cuda.get_device_name(0)} capability={cap} "
            f"bf16_supported={torch.cuda.is_bf16_supported()}"
        )

    wait_for_hest_manifest_complete(cfg, paths)
    hest_samples = _read_csv_with_retry(paths["manifests"] / "hest_samples.csv")
    spot_manifest = _read_csv_with_retry(paths["manifests"] / "hest_spot_manifest.csv")
    # "eligible" = passed src/scripts/08's alignment QC, i.e. has a real patch shard + spot_manifest rows.
    eligible_sample_ids = set(spot_manifest["sample_id"].unique())
    hest_samples = hest_samples[
        hest_samples["sample_id"].isin(eligible_sample_ids)
    ].reset_index(drop=True)
    hest_samples = exclude_targeted_panel_samples(hest_samples)
    print(
        f"[run_genbio_hest_baseline] {len(hest_samples)} QC-eligible HEST samples across "
        f"{hest_samples['cancer'].value_counts().to_dict()}"
    )

    print(
        "[run_genbio_hest_baseline] loading frozen H-Optimus-0 teacher "
        f"({cfg['teacher']['hf_repo']}, dtype={cfg['teacher']['dtype']}, "
        f"embedding_dim={cfg['teacher']['embedding_dim']}) ..."
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    teacher_h = HOptimusTeacher(cfg, device=device)  # load frozen teacher
    assert cfg["teacher"]["embedding_dim"] == TEACHER_EMBED_DIM
    for p in teacher_h.model.parameters():
        assert not p.requires_grad, "H-Optimus-0 must be fully frozen"
    assert not teacher_h.model.training, "H-Optimus-0 must be in eval() mode"
    print(
        f"[run_genbio_hest_baseline] H-Optimus-0 loaded, frozen, "
        f"{teacher_h.n_parameters / 1e9:.2f}B params"
    )

    if not args.skip_smoke_test:
        run_smoke_test(
            cfg, paths, teacher_h, embed_dir, patch_shard_dir, hest_samples, spot_manifest,
            batch_size=cfg["teacher"]["batch_size"],
        )
        if args.smoke_test_only:
            print("[run_genbio_hest_baseline] --smoke-test-only: done.")
            return

    # ------------------------------------------------------------ Step: H-Optimus-0 feature extraction (resumable)
    extraction_stats_by_cancer = {}
    if not args.train_only:
        extraction_env_log = paths["logs"] / "hoptimus0_baseline_extraction_env.json"
        with RunEnvironment("hoptimus0_baseline_extraction", extraction_env_log):
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
                    extract_hoptimus_embeddings_for_sample(
                        teacher_h, row, spot_manifest, patch_shard_dir, out_path,
                        batch_size=cfg["teacher"]["batch_size"],
                    )
                    n_extracted_now += 1
                elapsed = time.time() - t0
                extraction_stats_by_cancer[cancer] = {
                    "n_samples": len(samples), "n_spots": n_total_spots,
                    "n_newly_extracted_samples": n_extracted_now,
                    "seconds": elapsed,
                }
                print(
                    f"[run_genbio_hest_baseline] {cancer}: {len(samples)} samples / "
                    f"{n_total_spots} spots, {n_extracted_now} newly extracted this run, "
                    f"{elapsed:.1f}s"
                )
        if args.extract_only:
            print("[run_genbio_hest_baseline] --extract-only: done.")
            return

    # ------------------------------------------------------------ Step: downstream head, per cancer
    scripts10 = _import_scripts10()
    hest_root = Path(cfg["paths"]["hest_root"])
    mt = cfg["molecular_targets"]
    spot_uid = spot_manifest["sample_id"] + "::" + spot_manifest["spot_id"]
    spot_by_uid = spot_manifest.assign(spot_uid=spot_uid).set_index("spot_uid")

    summary_rows, gene_metric_rows, pathway_metric_rows = [], [], []
    for cancer in cfg["cancers"]["hest"]:
        per_cancer_json = results_dir / "per_cancer" / f"{cancer}_seed{args.seed}_fold{args.fold}.json"
        if per_cancer_json.exists():
            print(
                f"[run_genbio_hest_baseline] {cancer}: per-cancer result already present "
                f"at {per_cancer_json}, reusing."
            )
            cached = json.loads(per_cancer_json.read_text())
            summary_rows.append(cached["summary"])
            gene_metric_rows.extend(cached["gene_metrics"])
            pathway_metric_rows.extend(cached["pathway_metrics"])
            continue

        # load CV split
        fold_info, split_source = load_or_recompute_split(
            cfg, paths, hest_samples, cancer, args.seed, args.fold
        )
        train_patients_all, test_patients = fold_info["train_patients"], fold_info["test_patients"]
        assert set(train_patients_all).isdisjoint(test_patients), (
            f"{cancer}: train/test patient overlap! {set(train_patients_all) & set(test_patients)}"
        )
        train_p, val_p = carve_validation_from_train(train_patients_all, 0.15, args.seed)
        assert set(train_p).isdisjoint(test_patients) and set(val_p).isdisjoint(test_patients), (
            f"{cancer}: test patients leaked into train/val carve"
        )

        save_json(
            {
                "cancer": cancer, "seed": args.seed, "fold": args.fold, "source": split_source,
                "train_patients": sorted(train_p), "val_patients": sorted(val_p),
                "test_patients": sorted(test_patients),
            },
            splits_dir / f"{cancer}_seed{args.seed}_fold{args.fold}.json",
        )

        cancer_samples = hest_samples[hest_samples["cancer"] == cancer]

        def load_expr(patients):
            expr, progeny, genes_by_sample, net = scripts10.load_fold_expression(
                cancer_samples, spot_manifest, hest_root, patients, cfg
            )
            return expr, progeny, genes_by_sample, net

        # load train expression
        train_expr, train_progeny, genes_by_sample, net = load_expr(train_p)
        pathway_names = list(net["source"].unique())
        assert len(pathway_names) == 14, f"expected 14 PROGENy pathways, got {len(pathway_names)}"
        common_genes = sorted(set.intersection(*genes_by_sample.values())) if genes_by_sample else []

        # select HVGs
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
            # NaN is expected here, not a bug: decoupler returns NaN for spots it drops as degenerate
            # (epathst.data.pathways.compute_progeny_scores), and patch_head_loss already masks NaN via
            # gene_mask/path_mask. Only Inf would indicate a real standardization bug.
            split_name = "train" if is_train else "eval"
            assert not np.isinf(gene_t).any() and not np.isinf(path_t).any(), (
                f"{cancer}: Inf in standardized targets for {split_name} patients"
            )
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
                    feat_cache[sample_id] = load_cached_hoptimus_features(embed_dir / f"{sample_id}.h5")
                c = feat_cache[sample_id]
                pos = list(c["spot_id"]).index(bare_spot_id)
                out.append(c["embedding"][pos])
            return np.stack(out)

        # load cached H-Optimus-0 embeddings
        x_train, x_val, x_test = load_feats(ids_train), load_feats(ids_val), load_feats(ids_test)
        for name, x in (("train", x_train), ("val", x_val), ("test", x_test)):
            assert x.shape[1] == TEACHER_EMBED_DIM
            assert np.isfinite(x).all(), f"{cancer}: NaN/Inf in {name} H-Optimus-0 embeddings"

        print(
            f"[run_genbio_hest_baseline] {cancer}: train={len(train_p)}pat/{len(ids_train)}spots "
            f"val={len(val_p)}pat/{len(ids_val)}spots "
            f"test={len(test_patients)}pat/{len(ids_test)}spots (split source: {split_source})"
        )

        # train head
        seed_everything(args.seed)
        t_train0 = time.time()
        if args.head == "ridge":
            # Closed-form linear-probe baseline, the standard choice for frozen
            # foundation-model embeddings and less prone to MLP over/underfitting
            # (docs/RUN_STATUS.md). RidgeCV picks its regularization via built-in
            # leave-one-out CV on TRAIN+VAL (never test), so no early-stopping loop
            # is needed and train+val can be combined -- unlike the MLP path below,
            # which needs val held out for early-stopping.
            from sklearn.linear_model import RidgeCV
            x_fit = np.concatenate([x_train, x_val], axis=0)
            y_gene_fit = np.concatenate([y_gene_train, y_gene_val], axis=0)
            y_path_fit = np.concatenate([y_path_train, y_path_val], axis=0)
            alphas = np.logspace(-2, 4, 13)
            gene_ridge = RidgeCV(alphas=alphas).fit(x_fit, y_gene_fit)
            path_ridge = RidgeCV(alphas=alphas).fit(x_fit, y_path_fit)
            gene_pred = gene_ridge.predict(x_test)
            path_pred = path_ridge.predict(x_test)
            peak_vram_gb = float("nan")  # sklearn, CPU-only, no CUDA allocation to report
        else:
            head = HOptimusDownstreamHead(
                gene_dim=len(hvg["hvg1000"]), pathway_dim=14,
                dropout=cfg["patch_head"].get("dropout", 0.1),
            ).to(device)
            opt = torch.optim.AdamW(
                head.parameters(), lr=cfg["patch_head"]["lr"],
                weight_decay=cfg["patch_head"]["weight_decay"],
            )
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            best_val, patience_ctr, best_state = float("inf"), 0, None
            for epoch in range(cfg["patch_head"]["max_epochs"]):
                head.train()
                x_train_batch = torch.tensor(x_train, dtype=torch.float32, device=device)
                gene_train_batch = torch.tensor(y_gene_train, dtype=torch.float32, device=device)
                pathway_train_batch = torch.tensor(y_path_train, dtype=torch.float32, device=device)
                opt.zero_grad()
                loss = patch_head_loss(
                    head(x_train_batch), gene_train_batch, pathway_train_batch,
                    cfg["patch_head"]["pathway_loss_weight"],
                )
                loss["loss"].backward()
                opt.step()

                head.eval()
                with torch.no_grad():
                    x_val_batch = torch.tensor(x_val, dtype=torch.float32, device=device)
                    gene_val_batch = torch.tensor(y_gene_val, dtype=torch.float32, device=device)
                    pathway_val_batch = torch.tensor(y_path_val, dtype=torch.float32, device=device)
                    pathway_weight = cfg["patch_head"]["pathway_loss_weight"]
                    val_loss = patch_head_loss(
                        head(x_val_batch), gene_val_batch, pathway_val_batch, pathway_weight
                    )["loss"].item()
                if val_loss < best_val:
                    best_val = val_loss
                    patience_ctr = 0
                    best_state = {k: v.clone() for k, v in head.state_dict().items()}
                else:
                    patience_ctr += 1
                if patience_ctr >= cfg["patch_head"]["early_stop_patience"]:
                    break
            head.load_state_dict(best_state)
            peak_vram_gb = (
                torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
            )

            head.eval()
            with torch.no_grad():
                out = head(torch.tensor(x_test, dtype=torch.float32, device=device))
            gene_pred, path_pred = out["gene"].cpu().numpy(), out["pathway"].cpu().numpy()
        head_training_seconds = time.time() - t_train0

        # evaluate on test
        test_patient_ids = spot_by_uid.loc[ids_test, "patient_id"].values
        n_hvg50 = len(hvg["hvg50"])

        min_regions = cfg["evaluation"]["min_regions_per_patient"]
        pp_hvg1000 = per_patient_per_column_pearson(
            y_gene_test, gene_pred, test_patient_ids, hvg["hvg1000"], min_regions
        )
        # HVG50 is a strict column slice of the HVG1000 predictions/targets -- no separate model
        pp_hvg50 = per_patient_per_column_pearson(
            y_gene_test[:, :n_hvg50], gene_pred[:, :n_hvg50], test_patient_ids,
            hvg["hvg50"], min_regions,
        )
        pp_pathway = per_patient_per_column_pearson(
            y_path_test, path_pred, test_patient_ids, pathway_names, min_regions
        )

        r_thresholds = tuple(cfg["evaluation"]["gene_r_thresholds"])
        r_hvg1000 = target_summary(pp_hvg1000, r_thresholds)["fisher_z_mean_patient_R"]
        r_hvg50 = target_summary(pp_hvg50, r_thresholds)["fisher_z_mean_patient_R"]
        r_pathway = target_summary(pp_pathway, r_thresholds)["fisher_z_mean_patient_R"]

        pp_hvg1000.insert(0, "cancer", cancer)
        pp_pathway.insert(0, "cancer", cancer)
        gene_metric_rows.extend(pp_hvg1000.to_dict("records"))
        pathway_metric_rows.extend(pp_pathway.to_dict("records"))

        summary_row = {
            "cancer": cancer, "n_train_patients": len(train_p),
            "n_test_patients": len(test_patients),
            "n_train_spots": len(ids_train), "n_test_spots": len(ids_test),
            "HVG50_r": r_hvg50, "HVG1000_r": r_hvg1000, "PROGENy14_r": r_pathway,
            # mirrors epathst.evaluation.metrics.evaluate_predictions's mean_mse/mean_mae
            # (target_summary doesn't surface these) so this table has the same columns
            # as cv_ablation_metrics.csv.
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
        save_json(
            {
                "summary": summary_row, "gene_metrics": pp_hvg1000.to_dict("records"),
                "pathway_metrics": pp_pathway.to_dict("records"),
            },
            per_cancer_json,
        )
        print(
            f"[run_genbio_hest_baseline] {cancer}: HVG50_r={r_hvg50:.4f} "
            f"HVG1000_r={r_hvg1000:.4f} PROGENy14_r={r_pathway:.4f} "
            f"(head_training={head_training_seconds:.1f}s peak_vram={peak_vram_gb:.2f}GB)"
        )

    # ------------------------------------------------------------ save & print final results
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "hoptimus0_summary.csv", index=False)
    pd.DataFrame(gene_metric_rows).to_csv(results_dir / "hoptimus0_gene_metrics.csv", index=False)
    pd.DataFrame(pathway_metric_rows).to_csv(results_dir / "hoptimus0_pathway_metrics.csv", index=False)

    print("\n[run_genbio_hest_baseline] ================= FINAL RESULTS =================")
    print(f"{'Cancer':<8} {'HVG50 r':>10} {'HVG1000 r':>10} {'PROGENy14 r':>12}")
    for _, row in summary_df.iterrows():
        print(
            f"{row['cancer']:<8} {row['HVG50_r']:>10.4f} {row['HVG1000_r']:>10.4f} "
            f"{row['PROGENy14_r']:>12.4f}"
        )
    if len(summary_df):
        print(
            f"{'MEAN':<8} {summary_df['HVG50_r'].mean():>10.4f} "
            f"{summary_df['HVG1000_r'].mean():>10.4f} {summary_df['PROGENy14_r'].mean():>12.4f}"
        )
    print(f"\nn_samples_processed={len(hest_samples)} n_spots_total={spot_manifest.shape[0]}")
    for cancer, stats in extraction_stats_by_cancer.items():
        pps = stats["n_spots"] / stats["seconds"] if stats["seconds"] > 0 else float("nan")
        print(
            f"  {cancer}: extraction {stats['seconds']:.1f}s over {stats['n_spots']} spots "
            f"({pps:.1f} patches/sec)"
        )
    print(f"results written to {results_dir}")


if __name__ == "__main__":
    main()
