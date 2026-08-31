"""Evaluates whether distillation worked, on completely unseen CPTAC test
patients. All metrics computed overall and per cancer type (BRCA/COAD/
CCRCC), with bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np


def cosine_similarity_distribution(student_proj: np.ndarray, teacher_emb: np.ndarray) -> np.ndarray:
    s = student_proj / np.clip(np.linalg.norm(student_proj, axis=1, keepdims=True), 1e-8, None)
    t = teacher_emb / np.clip(np.linalg.norm(teacher_emb, axis=1, keepdims=True), 1e-8, None)
    return np.sum(s * t, axis=1)


def bootstrap_ci_1d(values: np.ndarray, n_iter: int = 10000, seed: int = 0, ci=(2.5, 97.5)) -> dict:
    rng = np.random.RandomState(seed)
    n = len(values)
    boots = np.array([values[rng.randint(0, n, size=n)].mean() for _ in range(n_iter)])
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "ci_low": float(np.percentile(boots, ci[0])), "ci_high": float(np.percentile(boots, ci[1])),
            "n": n}


def retrieval_overlap_at_k(student_proj: np.ndarray, teacher_emb: np.ndarray, ks=(1, 5, 10),
                            max_query: int = 2000, seed: int = 0) -> dict:
    """For each test patch, compares teacher-space nearest neighbors to
    student-space nearest neighbors and reports overlap@K. Uses a random
    subsample of queries (max_query) against the full test set as the
    search pool -- exact brute-force cosine kNN, capped for tractability
    on large test sets."""
    from sklearn.neighbors import NearestNeighbors

    s = student_proj / np.clip(np.linalg.norm(student_proj, axis=1, keepdims=True), 1e-8, None)
    t = teacher_emb / np.clip(np.linalg.norm(teacher_emb, axis=1, keepdims=True), 1e-8, None)
    n = s.shape[0]
    rng = np.random.RandomState(seed)
    query_idx = rng.choice(n, size=min(max_query, n), replace=False)

    max_k = max(ks) + 1  # +1 to drop self-match
    nn_t = NearestNeighbors(n_neighbors=max_k, metric="cosine").fit(t)
    nn_s = NearestNeighbors(n_neighbors=max_k, metric="cosine").fit(s)
    _, idx_t = nn_t.kneighbors(t[query_idx])
    _, idx_s = nn_s.kneighbors(s[query_idx])

    results = {}
    for k in ks:
        overlaps = []
        for row_t, row_s, q in zip(idx_t, idx_s, query_idx):
            set_t = set(row_t[row_t != q][:k])
            set_s = set(row_s[row_s != q][:k])
            overlaps.append(len(set_t & set_s) / k)
        results[f"overlap_at_{k}"] = float(np.mean(overlaps))
    results["n_queries"] = len(query_idx)
    return results


def linear_probe_sanity(student_proj: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict:
    """Linear-probe sanity test -- e.g. cancer-type classification from the
    student's projected embedding, 5-fold CV accuracy. A sanity signal
    only, not primary evidence."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    scores = cross_val_score(clf, student_proj, labels, cv=5, scoring="accuracy")
    return {"mean_accuracy": float(scores.mean()), "std_accuracy": float(scores.std()), "fold_scores": scores.tolist()}


def umap_embedding(features: np.ndarray, seed: int = 0):
    """UMAP for visualization only, never as primary proof."""
    import umap
    reducer = umap.UMAP(n_components=2, random_state=seed)
    return reducer.fit_transform(features)


def evaluate_distillation(student_proj: np.ndarray, teacher_emb: np.ndarray, cancer_types: np.ndarray,
                           cfg: dict) -> dict:
    """Runs the full distillation-eval battery overall + per cancer type."""
    out = {}
    groups = {"overall": np.ones(len(cancer_types), dtype=bool)}
    for c in np.unique(cancer_types):
        groups[c] = cancer_types == c

    for name, mask in groups.items():
        sp, te = student_proj[mask], teacher_emb[mask]
        cos = cosine_similarity_distribution(sp, te)
        entry = {
            "n_patches": int(mask.sum()),
            "cosine_similarity": bootstrap_ci_1d(cos, n_iter=cfg["statistics"]["n_bootstrap"]),
        }
        if mask.sum() >= 20:
            entry["retrieval"] = retrieval_overlap_at_k(sp, te)
        out[name] = entry
    return out
