"""Fast, torch-free tests for the deterministic numeric code -- runnable
right now, before any real data lands."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathology_st import destruction, graph_builder, metrics, stats_tests


def test_multiscale_edges_within_radius():
    rng = np.random.RandomState(0)
    coords = rng.uniform(0, 1000, size=(50, 2))
    edges = graph_builder.build_multiscale_edges(coords, 150, 300, 600, 8, 12, 16)
    assert edges["edge_index"].shape[1] == len(edges["distance_um"])
    assert (edges["distance_um"] <= 600.0).all()
    assert set(np.unique(edges["scale"])) <= {"local", "meso", "macro"}


def test_union_edge_features_shape():
    # V2 Phase 9 correction: 1 distance + 16 RBF + 3-D scale one-hot = 20-D (V1 omitted the distance scalar -> 19-D, a bug).
    d = np.array([10.0, 200.0, 500.0], dtype=np.float32)
    scale = np.array(["local", "meso", "macro"], dtype=object)
    feats = graph_builder.union_edge_features(d, scale, 600.0, n_rbf=16)
    assert feats.shape == (3, 20)
    # back-compat alias must produce the identical corrected output
    assert np.allclose(feats, graph_builder.rbf_edge_features(d, scale, 600.0, n_rbf=16))


def test_per_scale_edge_features_shape():
    # V2 Phase 9: scale-aware branches don't need the one-hot (branch identity IS the scale) -> 1 distance + 16 RBF = 17-D.
    d = np.array([10.0, 200.0, 500.0], dtype=np.float32)
    feats = graph_builder.per_scale_edge_features(d, 600.0, n_rbf=16)
    assert feats.shape == (3, 17)


def test_split_edges_by_scale():
    rng = np.random.RandomState(2)
    coords = rng.uniform(0, 1000, size=(40, 2))
    edges = graph_builder.build_multiscale_edges(coords, 150, 300, 600, 8, 12, 16)
    split = graph_builder.split_edges_by_scale(edges)
    assert set(split.keys()) == {"local", "meso", "macro"}
    total_split_edges = sum(e["edge_index"].shape[1] for e in split.values())
    assert total_split_edges == edges["edge_index"].shape[1]
    for name, e in split.items():
        assert set(e["scale"].tolist()) <= {name}


def test_degree_preserving_rewire_keeps_degree_and_edge_count():
    rng = np.random.RandomState(1)
    coords = rng.uniform(0, 1000, size=(40, 2))
    edges = graph_builder.build_multiscale_edges(coords, 150, 300, 600, 8, 12, 16)
    n = 40
    rewired = destruction.degree_preserving_rewire(edges["edge_index"], seed=0)

    def deg(edge_index):
        d = np.zeros(n, dtype=int)
        for j in edge_index[1]:
            d[j] += 1
        return d

    assert np.array_equal(np.sort(deg(edges["edge_index"])), np.sort(deg(rewired)))
    assert edges["edge_index"].shape[1] == rewired.shape[1]


def test_random_graph_preserves_edge_count():
    edge_index2 = np.array([[0, 1, 2, 3], [1, 0, 3, 2]])
    out2 = destruction.random_graph(edge_index2, n_nodes=10, seed=0)
    assert out2.shape[1] == edge_index2.shape[1]


def test_patient_fisher_z_pearson_perfect_correlation():
    y_true = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_pred = np.array([[1.0], [2.0], [3.0], [4.0]])
    patient_ids = np.array(["p1", "p1", "p1", "p1"])
    df = metrics.per_patient_per_column_pearson(
        y_true, y_pred, patient_ids, ["gene0"], min_regions=3
    )
    assert np.isclose(df["r"].iloc[0], 1.0)


def test_bootstrap_ci_positive_when_deltas_positive():
    deltas = np.array([0.1, 0.2, 0.15, 0.05, 0.3])
    ci = stats_tests.patient_bootstrap_ci(deltas, n_iterations=2000, seed=0)
    assert ci["ci_low"] > 0


def test_bh_fdr_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.02, 0.5, np.nan])
    reject, q = stats_tests.bh_fdr(p, q_threshold=0.05)
    assert reject[0]  # smallest p-value should survive FDR correction
    assert np.isnan(q[-1])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK  {name}")
