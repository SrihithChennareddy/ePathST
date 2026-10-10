"""Frozen ePathST-E, ePathST and H-Optimus-0 prediction of each cancer's official HEST 50-gene
panel for every extracted TCGA H&E patch ("pseudo-spot"). Nothing is trained here.

Also hosts the shared helpers (bundle loading, `predict_from_embeddings`, slide QC) used by
`hexgrid.py` and `worker.py`, i.e. the `src/scripts/24_tcga_validation.py predict` and `worker`
stages.

Model bundles are the official-fold `models/{cancer}_fold{n}/` directories from the Leiden official
panel run. TCGA project -> panel/models:
    TCGA-BRCA -> IDC    TCGA-COAD -> COAD    TCGA-KIRC -> CCRCC
Per slide:
  1. read the patches (`patches/<project>/<case>.h5`) and embed them with the frozen distilled
     student (320-d) and frozen H-Optimus-0 (1536-d CLS, then each fold's train-fit PCA);
  2. build the multiscale graph (150/300/600 um, caps 8/12/16) from patch coordinates converted to
     microns with the slide's native mpp (patch x/y are level-0 pixels);
  3. for every fold bundle: ePathST-E = patch head; ePathST = full ResidualGraphPredictor;
     H-Optimus-0 = PCA + head; un-standardize with that fold's mu/sigma (log1p scale), then
     average across folds.
Writes `<out>/<tag>/<project>/<case>.npz` (spot ids, um coordinates, the 50 genes, three (n, 50)
prediction matrices) and a per-slide QC row. Resumable and shardable for SLURM arrays.

Caveat recorded in the QC file: the extracted patches are a thinned regular grid with stride tied
to the slide's native mpp, so spacing differs from Visium/Xenium and the graph model sees different
neighbour geometry than in training. `median_nn_um` / `frac_nodes_lt8_local` quantify it per slide.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from epathst.ablations import build_graph_for_ablation, scale_aware_edge_data
from epathst.data.feature_cache import extract_student_features
from epathst.data.wsi_patches import open_slide
from epathst.graph.training import make_pyg_data, predict_graph_model
from epathst.models.student import load_locked_encoder
from epathst.models.teacher import HOptimusTeacher
from epathst.utils import load_config, resolve_run_paths

TCGA_ROOT = Path("/lustre/nvwulf/scratch/schennareddy/TCGA")
PERSIST = Path("/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/tcga")
PANEL_OF = {"TCGA-BRCA": "IDC", "TCGA-COAD": "COAD", "TCGA-KIRC": "CCRCC"}
DEFAULT_MODELS = (
    "/lustre/nvwulf/projects/YurovskyGroup-nvwulf/srihith/cptac_hoptimus0_distill_hest_pna_work/"
    "results/leiden_official_hvg50/models"
)
SYSTEMS = ("ePathST-E", "ePathST", "H-Optimus-0")


def load_bundles(models_dir: Path, cancer: str, device: str) -> list[dict]:
    bundles = []
    for d in sorted(glob.glob(str(models_dir / f"{cancer}_fold*"))):
        d = Path(d)
        meta = json.loads((d / "meta.json").read_text())
        sd, pca = np.load(d / "standardizer.npz"), np.load(d / "hoptimus_pca.npz")
        ld = lambda n: torch.load(
            d / n, map_location=device, weights_only=False
        ).eval()  # noqa: E731
        bundles.append(
            {
                "name": d.name,
                "meta": meta,
                "head_e": ld("epathst_e_patch_head.pt"),
                "head_t": ld("hoptimus_patch_head.pt"),
                "full": ld("epathst_full.pt"),
                "mu": sd["mu"].astype(np.float32),
                "scale": (sd["sigma"] + float(sd["eps"])).astype(np.float32),
                "pca_mean": pca["mean"].astype(np.float32),
                "pca_comp": pca["components"].astype(np.float32),
            }
        )
    assert bundles, f"no model bundles for {cancer} under {models_dir}"
    genes = bundles[0]["meta"]["genes"]
    assert all(b["meta"]["genes"] == genes for b in bundles), "fold bundles disagree on gene order"
    assert len(genes) == 50 and len(set(genes)) == 50
    return bundles


@torch.no_grad()
def head_predict(head, x: np.ndarray, device: str) -> np.ndarray:
    return head(torch.tensor(x, dtype=torch.float32, device=device))["gene"].cpu().numpy()


def slide_qc(coords_um: np.ndarray, cfg: dict) -> dict:
    d, _ = cKDTree(coords_um).query(coords_um, k=2)
    tree = cKDTree(coords_um)
    n_local = np.array(
        [len(x) - 1 for x in tree.query_ball_point(coords_um, r=cfg["graph"]["local_radius_um"])]
    )
    return {
        "median_nn_um": float(np.median(d[:, 1])),
        "p5_nn_um": float(np.percentile(d[:, 1], 5)),
        "p95_nn_um": float(np.percentile(d[:, 1], 95)),
        "frac_nodes_lt8_local": float((n_local < cfg["graph"]["max_local_neighbors"]).mean()),
    }


def predict_from_embeddings(bundles, x_s, x_t, coords_um, case, cfg, device, seed) -> dict:
    """Frozen-model predictions for one slide from its embeddings and micron coordinates: the
    multiscale graph is built once, every fold bundle is applied, outputs are un-standardized (log1p
    scale) and averaged across folds.
    """
    n = len(x_s)
    edges = build_graph_for_ablation(coords_um, "multiscale", cfg, seed)
    ei, ea, es = scale_aware_edge_data(edges, cfg)
    g = make_pyg_data(
        x_s,
        ei,
        ea,
        np.zeros((n, 50), np.float32),
        np.full((n, 1), np.nan, np.float32),
        case,
        edge_scale=es,
    )
    acc = {s: np.zeros((n, 50), np.float64) for s in SYSTEMS}
    for b in bundles:
        z_e = head_predict(b["head_e"], x_s, device)
        z_t = head_predict(b["head_t"], (x_t - b["pca_mean"]) @ b["pca_comp"].T, device)
        out = predict_graph_model(b["full"], [g], device, backend="scale_aware")
        assert np.allclose(
            out["gene_patch_only"], z_e, atol=1e-4
        ), f"{case} {b['name']}: ePathST's patch head differs from the saved ePathST-E head"
        for s, z_ in (("ePathST-E", z_e), ("ePathST", out["gene_pred"]), ("H-Optimus-0", z_t)):
            acc[s] += z_ * b["scale"] + b["mu"]
    preds = {s: (acc[s] / len(bundles)).astype(np.float32) for s in SYSTEMS}
    assert all(np.isfinite(p).all() for p in preds.values()), f"{case}: non-finite predictions"
    # A8 (FINAL_GUIDE.md Step 1): E, P and T all predict the same `n` rows of `coords_um` built
    # above, so their spot sets cannot diverge; assert it so a future refactor can't break it silently.
    assert all(p.shape[0] == n for p in preds.values()), f"{case}: systems produced different spot counts"
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--projects", nargs="+", default=list(PANEL_OF))
    ap.add_argument("--models-dir", default=DEFAULT_MODELS)
    ap.add_argument(
        "--tag", default=None, help="output subfolder; default = name of the models dir's parent"
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument(
        "--max-slides", type=int, default=None, help="smoke test: first N slides per project"
    )
    ap.add_argument("--min-spots", type=int, default=200)
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--seed", type=int, default=0, help="graph builder seed (edges are deterministic)"
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_run_paths(cfg)
    device = args.device if torch.cuda.is_available() else "cpu"
    models_dir = Path(args.models_dir)
    tag = args.tag or models_dir.parent.name
    out_root = PERSIST / "predictions" / tag
    emb_root = TCGA_ROOT / "embeddings"
    print(f"[prediction] device={device} models={models_dir} -> {out_root}", flush=True)

    ckpt = torch.load(
        paths["student_checkpoints"] / f"locked_encoder_seed{args.feature_seed}.pt",
        map_location=device,
    )
    student = load_locked_encoder(ckpt["encoder_state_dict"], device)
    teacher = HOptimusTeacher(cfg, device)
    bs = cfg["teacher"]["batch_size"]

    for project in args.projects:
        cancer = PANEL_OF[project]
        bundles = load_bundles(models_dir, cancer, device)
        genes = bundles[0]["meta"]["genes"]
        print(f"[prediction] {project}: {len(bundles)} fold bundles ({cancer} panel)", flush=True)

        slides = pd.read_csv(TCGA_ROOT / "metadata" / f"{project}_slides.csv")
        slides = slides[slides["status"] == "downloaded"].copy()
        slides["case"] = slides["file_name"].str[:12]
        path_of = dict(zip(slides["case"], slides["local_path"]))
        cases = sorted(
            p.stem for p in (TCGA_ROOT / "patches" / project).glob("*.h5") if p.stem in path_of
        )
        if args.max_slides:
            cases = cases[: args.max_slides]
        cases = cases[args.shard :: args.n_shards]
        (out_root / project).mkdir(parents=True, exist_ok=True)
        (emb_root / project).mkdir(parents=True, exist_ok=True)
        qc_path = out_root / f"qc_{project}_shard{args.shard}.csv"
        qc_rows = pd.read_csv(qc_path).to_dict("records") if qc_path.exists() else []
        done = {r["case"] for r in qc_rows}

        for i, case in enumerate(cases):
            out_path = out_root / project / f"{case}.npz"
            if out_path.exists() and case in done:
                continue
            t0 = time.time()
            try:
                with h5py.File(TCGA_ROOT / "patches" / project / f"{case}.h5", "r") as f:
                    pids = np.array([p.decode() for p in f["patch_id"][:]])
                    xy_px = np.stack([f["x"][:], f["y"][:]], axis=1).astype(np.float64)
                    patches = f["patches"][:]
                if len(pids) < args.min_spots:
                    qc_rows.append(
                        {
                            "case": case,
                            "project": project,
                            "status": f"skipped_<{args.min_spots}_spots",
                            "n_spots": len(pids),
                        }
                    )
                    continue
                reader = open_slide(path_of[case])
                native_mpp, mpp_from_slide = reader.mpp, reader.mpp is not None
                if hasattr(reader, "close"):
                    reader.close()
                native_mpp = (
                    native_mpp or 0.5
                )  # same fallback as patch extraction (stride/crop size)
                coords_um = xy_px * native_mpp

                ep = emb_root / project / f"{case}.npz"
                if ep.exists():
                    z = np.load(ep)
                    assert np.array_equal(z["patch_id"], pids)
                    x_s, x_t = z["student"].astype(np.float32), z["teacher"].astype(np.float32)
                else:
                    x_s = extract_student_features(student, patches, device)
                    x_t = np.concatenate(
                        [
                            teacher.embed_batch(patches[j : j + bs])
                            for j in range(0, len(patches), bs)
                        ]
                    )
                    np.savez(
                        ep,
                        patch_id=pids,
                        student=x_s.astype(np.float16),
                        teacher=x_t.astype(np.float16),
                    )
                del patches

                preds = predict_from_embeddings(
                    bundles, x_s, x_t, coords_um, case, cfg, device, args.seed
                )

                np.savez_compressed(
                    out_path,
                    spot_id=pids,
                    x_um=coords_um[:, 0],
                    y_um=coords_um[:, 1],
                    genes=np.array(genes),
                    n_folds=len(bundles),
                    native_mpp=native_mpp,
                    **{f"pred__{s}": p for s, p in preds.items()},
                )
                qc_rows.append(
                    {
                        "case": case,
                        "project": project,
                        "status": "ok",
                        "n_spots": len(pids),
                        "native_mpp": native_mpp,
                        "mpp_from_slide": mpp_from_slide,
                        "seconds": round(time.time() - t0, 1),
                        **slide_qc(coords_um, cfg),
                    }
                )
                print(
                    f"[prediction] {project} {i + 1}/{len(cases)} {case}: {len(pids)} spots, "
                    f"mpp={native_mpp}, "
                    f"median NN {qc_rows[-1]['median_nn_um']:.0f}um, {qc_rows[-1]['seconds']}s",
                    flush=True,
                )
            except Exception as e:  # keep going; failures are recorded, not hidden
                qc_rows.append(
                    {
                        "case": case,
                        "project": project,
                        "status": f"FAILED {type(e).__name__}: {e}"[:200],
                    }
                )
                print(
                    f"[prediction] {project} {case}: FAILED {type(e).__name__}: {e}",
                    file=sys.stderr,
                    flush=True,
                )
            pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
        pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
    print("[prediction] done", flush=True)
