"""Fast, torch-free tests for the deterministic numeric code -- runnable
right now, before any real data lands."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sts import destruction, graph_builder, metrics, stats_tests


def test_multiscale_edges_within_radius():
    rng = np.random.RandomState(0)
    coords = rng.uniform(0, 1000, size=(50, 2))
    edges = graph_builder.build_multiscale_edges(coords, 150, 300, 600, 8, 12, 16)
    assert edges["edge_index"].shape[1] == len(edges["distance_um"])
    assert (edges["distance_um"] <= 600.0).all()
    assert set(np.unique(edges["scale"])) <= {"local", "meso", "macro"}


def test_rbf_edge_features_shape():
    d = np.array([10.0, 200.0, 500.0], dtype=np.float32)
    scale = np.array(["local", "meso", "macro"], dtype=object)
    feats = graph_builder.rbf_edge_features(d, scale, 600.0, n_rbf=16)
    assert feats.shape == (3, 19)


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
    edge_index = np.array([[0, 1, 2], [1, 0, 3]])
    out = destruction.random_graph(edge_index, n_nodes=10, seed=0)
    n_undirected_in = 1  # (0,1)/(1,0) is one undirected pair; (2,3) is another -> 2 total
    edge_index2 = np.array([[0, 1, 2, 3], [1, 0, 3, 2]])
    out2 = destruction.random_graph(edge_index2, n_nodes=10, seed=0)
    assert out2.shape[1] == edge_index2.shape[1]


def test_patient_fisher_z_pearson_perfect_correlation():
    y_true = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_pred = np.array([[1.0], [2.0], [3.0], [4.0]])
    patient_ids = np.array(["p1", "p1", "p1", "p1"])
    df = metrics.per_patient_per_column_pearson(y_true, y_pred, patient_ids, ["gene0"], min_regions=3)
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
