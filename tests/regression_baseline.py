"""Records deterministic digests of the numeric library code (graphs, statistics, metrics, splits,
models). Run before and after an organisational change and diff the two JSON files; any difference
is a behaviour change.

    python tests/regression_baseline.py --out before.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from epathst import ablations  # noqa: E402
from epathst.data import splits  # noqa: E402
from epathst.efficiency import efficiency  # noqa: E402
from epathst.evaluation import metrics  # noqa: E402
from epathst.graph import builder, destruction  # noqa: E402
from epathst.models import graph_layers, graph_model, patch_head, student  # noqa: E402
from epathst.statistics import stats_tests  # noqa: E402
from epathst.utils import load_config  # noqa: E402


def digest(obj) -> str:
    """Stable hash of arrays, tensors and nested containers (floats rounded to 6 decimals)."""
    h = hashlib.sha256()

    def feed(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            if x.dtype.kind == "f":
                x = np.round(x.astype(np.float64), 6) + 0.0
            h.update(str(x.shape).encode() + str(x.dtype.kind).encode())
            h.update(np.ascontiguousarray(x.astype(str) if x.dtype.kind in "OU" else x).tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode())
                feed(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                feed(v)
        elif isinstance(x, float):
            h.update(repr(round(x, 6) + 0.0).encode())
        else:
            h.update(repr(x).encode())

    feed(obj)
    return h.hexdigest()[:16]


def record_config(out):
    cfg = load_config(str(ROOT / "configs/base.yaml"))
    out["config_digest"] = digest(json.loads(json.dumps(cfg, default=str)))
    return load_config(str(ROOT / "configs/base.yaml"))


def record_graphs(out, cfg):
    g = cfg["graph"]
    rng = np.random.RandomState(0)
    coords = rng.uniform(0, 2000, size=(300, 2))
    args = (
        g["local_radius_um"],
        g["meso_radius_um"],
        g["macro_radius_um"],
        g["max_local_neighbors"],
        g["max_meso_neighbors"],
        g["max_macro_neighbors"],
    )
    edges = builder.build_multiscale_edges(coords, *args)
    out["graph_edges"] = {
        "n_edges": int(edges["edge_index"].shape[1]),
        "digest": digest(edges),
        "per_scale": {k: int((edges["scale"] == k).sum()) for k in builder.SCALE_NAMES},
    }
    out["graph_union_features"] = digest(
        builder.union_edge_features(
            edges["distance_um"], edges["scale"], g["macro_radius_um"], g["n_rbf"], g["rbf_gamma"]
        )
    )
    out["graph_per_scale_features"] = digest(
        builder.per_scale_edge_features(
            edges["distance_um"], g["macro_radius_um"], g["n_rbf"], g["rbf_gamma"]
        )
    )
    out["graph_split_by_scale"] = digest(builder.split_edges_by_scale(edges))
    out["graph_local_only"] = digest(
        builder.build_local_only_edges(coords, g["local_radius_um"], g["max_local_neighbors"])
    )
    out["destruction"] = {
        "random": digest(destruction.random_graph(edges["edge_index"], 300, 3)),
        "rewire": digest(destruction.degree_preserving_rewire(edges["edge_index"], 3)),
        "partial": digest(
            destruction.partial_edge_replacement(
                edges["edge_index"], 300, 0.4, 3, distance_um=edges["distance_um"]
            )
        ),
    }
    out["ablation_graphs"] = {
        src: digest(ablations.build_graph_for_ablation(coords, src, cfg, 5))
        for src in ("multiscale", "local", "random", "shuffled", "none")
    }
    ms = ablations.build_graph_for_ablation(coords, "multiscale", cfg, 5)
    out["ablation_edge_data"] = {
        "union": digest(ablations.edge_features_for_ablation(ms, "multiscale", cfg)),
        "scale_aware": digest(ablations.scale_aware_edge_data(ms, cfg)),
    }
    out["ablation_grid"] = [
        (a.name, a.feature_source, a.graph_source, a.backend, a.priority)
        for a in ablations.ABLATION_GRID
    ]
    out["matched_mlp_hidden"] = [
        ablations.matched_mlp_hidden_dim(n, 320, 64) for n in (50_000, 123_456, 900_000)
    ]
    return edges


def record_statistics(out):
    rng = np.random.RandomState(1)
    deltas = rng.normal(0.02, 0.1, size=40)
    out["bootstrap_ci"] = digest(
        stats_tests.patient_bootstrap_ci(deltas, n_iterations=2000, seed=4)
    )
    out["sign_flip"] = digest(stats_tests.paired_sign_flip_test(deltas, n_iterations=2000, seed=4))
    out["bh_fdr"] = digest(stats_tests.bh_fdr(rng.uniform(0, 1, size=60)))


def record_metrics(out, cfg):
    rng = np.random.RandomState(2)
    n, n_cols = 600, 8
    patients = np.repeat([f"P{i}" for i in range(12)], n // 12)
    cancers = np.repeat(["IDC", "COAD", "CCRCC"], n // 3)
    y_true = rng.normal(size=(n, n_cols))
    y_pred = 0.5 * y_true + rng.normal(size=(n, n_cols))
    cols = [f"g{i}" for i in range(n_cols)]
    out["evaluate_predictions"] = digest(
        metrics.evaluate_predictions(y_true, y_pred, patients, cancers, cols, cfg)
    )
    pp = metrics.per_patient_per_column_pearson(y_true, y_pred, patients, cols, 5)
    out["fisher_z"] = digest(metrics.patient_macro_fisher_z(pp).to_dict("list"))
    out["morans_i"] = digest(metrics.morans_i(y_true[:, 0], rng.uniform(0, 1000, size=(n, 2))))


def record_splits(out, cfg):
    manifest = pd.DataFrame(
        {
            "patient_id": [f"p{i:03d}" for i in range(90)],
            "cancer_type": ["BRCA"] * 30 + ["COAD"] * 30 + ["CCRCC"] * 30,
        }
    )
    out["cptac_split"] = digest(splits.cptac_patient_split(manifest, cfg, seed=7).to_dict("list"))
    by_cancer = {
        c: manifest[manifest.cancer_type == c].patient_id.tolist() for c in ("BRCA", "COAD")
    }
    out["cv_grid"] = digest(splits.build_cv_grid(by_cancer, cfg))
    out["carve_validation"] = digest(
        splits.carve_validation_from_train(manifest.patient_id.tolist()[:40], 0.2, 3)
    )


def record_models(out, cfg, edges):
    gene_dim, pathway_dim, in_dim = 50, 14, 320
    torch.manual_seed(0)
    x = torch.randn(300, in_dim)
    out["models"] = {}

    head = patch_head.build_patch_head(cfg, in_dim, gene_dim, pathway_dim).eval()
    out["models"]["patch_head"] = {
        "params": efficiency.count_model_params(head),
        "forward": digest(head(x)),
    }

    ei = torch.as_tensor(edges["edge_index"], dtype=torch.long)
    deg = graph_layers.compute_degree_histogram([ei], [300])
    g = cfg["graph"]
    union_attr = torch.as_tensor(
        builder.union_edge_features(
            edges["distance_um"], edges["scale"], g["macro_radius_um"], g["n_rbf"], g["rbf_gamma"]
        )
    )
    scale_attr = torch.as_tensor(
        builder.per_scale_edge_features(
            edges["distance_um"], g["macro_radius_um"], g["n_rbf"], g["rbf_gamma"]
        )
    )
    edge_scale = torch.as_tensor([ablations.SCALE_TO_INT[s] for s in edges["scale"]])
    deg_by_scale = [
        graph_layers.compute_degree_histogram([ei[:, edge_scale == k]], [300]) for k in range(3)
    ]
    out["models"]["degree_histogram"] = digest(deg)

    candidates = {
        "pna_union": lambda: graph_model.GraphCorrectionModel(
            in_dim, cfg, deg, union_attr.shape[1], gene_dim, pathway_dim
        ),
        "gatv2": lambda: graph_model.GATv2CorrectionModel(
            in_dim, cfg, union_attr.shape[1], gene_dim, pathway_dim
        ),
        "scale_aware": lambda: graph_model.ScaleAwareGraphCorrectionModel(
            in_dim, cfg, *deg_by_scale, scale_attr.shape[1], gene_dim, pathway_dim
        ),
    }
    for name, make in candidates.items():
        torch.manual_seed(0)
        model = make().eval()
        args = (ei, union_attr) if name != "scale_aware" else (ei, scale_attr, edge_scale)
        out["models"][name] = {
            "params": efficiency.count_model_params(model),
            "forward": digest(model(x, *args)),
        }
        torch.manual_seed(1)
        residual = graph_model.ResidualGraphPredictor(
            patch_head.build_patch_head(cfg, in_dim, gene_dim, pathway_dim), make()
        ).eval()
        out["models"][name]["residual_forward"] = digest(residual(x, *args))

    torch.manual_seed(0)
    enc = student.ConvNeXtV2AttoStudent(pretrained=None).eval()
    img = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out["models"]["student_encoder"] = {
            "params": efficiency.count_model_params(enc),
            "forward": digest(enc(img)),
        }
    torch.manual_seed(0)
    distill = student.DistillationStudent(cfg, pretrained=None).eval()
    out["models"]["distillation_student_params"] = efficiency.count_model_params(distill)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = {}
    cfg = record_config(out)
    edges = record_graphs(out, cfg)
    record_statistics(out)
    record_metrics(out, cfg)
    record_splits(out, cfg)
    record_models(out, cfg, edges)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print(f"wrote {args.out} ({len(out)} sections)")


if __name__ == "__main__":
    main()
