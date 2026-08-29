"""Final_Project_V2.txt Phase 21: fair GenBio-vs-student comparison via
training-fold-only PCA, replacing V1's ambiguous "feed the raw 4608-D
GenBio embedding into the same downstream head" approach.

V2's reasoning, quoted: "don't feed raw 4608-D vectors into a much larger
downstream head... GenBio 4608-D -> training-only PCA -> 256-D... PCA must
be fitted using training patients only." This mirrors HEST-Benchmark's own
PCA-256 + Ridge convention (V2 Phase 21), so the comparison is both fair
(matched output dimensionality: GenBio-PCA256 vs. student's 320->256
projection) and methodologically standard (no test-set leakage into the
PCA basis).

Produces the four-way comparison V2 Phase 21 requires:
    1. GenBio PCA256 -> Ridge              vs.  Student 256 -> Ridge
    2. GenBio PCA256 -> molecular MLP       vs.  Student 256 -> molecular MLP
    3. GenBio PCA256 -> residual multiscale PNA  vs.  Student 256 -> residual multiscale PNA
Run only AFTER the student model matrix (A0-A9) is locked -- never used to
retune CPTAC distillation or the downstream architecture (Section 24's
"never use HEST performance to repeatedly tune" principle applied here too).
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


class TrainOnlyPCA:
    """Fits sklearn PCA on TRAIN-patient rows only; transforms train/val/
    test identically. Tracks fit spot_ids so leakage can be asserted, same
    convention as sts.expression.TrainOnlyStandardizer."""

    def __init__(self, n_components: int = 256, random_state: int = 0):
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self._fit_ids: set = set()

    def fit(self, X_train: np.ndarray, ids=None) -> "TrainOnlyPCA":
        self.pca.fit(X_train)
        if ids is not None:
            self._fit_ids = set(ids)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)

    def assert_no_leakage(self, other_ids) -> None:
        overlap = self._fit_ids & set(other_ids)
        assert not overlap, f"PCA was fit using {len(overlap)} ids that also appear in this eval/test set"

    @property
    def explained_variance_ratio_sum(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum())


def fit_ridge_baseline(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> Ridge:
    """The simplest of the three required heads (V2 Phase 21 item 1) --
    Ridge regression directly on the (PCA-reduced or student) embedding,
    fit on training patients only."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def genbio_pca_comparison_report(genbio_ridge_r: float, student_ridge_r: float, genbio_mlp_r: float,
                                  student_mlp_r: float, genbio_pna_r: float, student_pna_r: float,
                                  pca_explained_variance: float) -> dict:
    """Assembles the V2 Phase 21 four-way (well, three-head x two-encoder)
    comparison table plus the retention question it exists to answer:
    'How much of a 1.1B-model representation survives in a 3.7M student?'"""
    return {
        "pca_explained_variance_ratio": pca_explained_variance,
        "ridge": {"genbio_pca256": genbio_ridge_r, "student_256": student_ridge_r,
                  "retention": student_ridge_r / genbio_ridge_r if genbio_ridge_r else float("nan")},
        "molecular_mlp": {"genbio_pca256": genbio_mlp_r, "student_256": student_mlp_r,
                          "retention": student_mlp_r / genbio_mlp_r if genbio_mlp_r else float("nan")},
        "residual_multiscale_pna": {"genbio_pca256": genbio_pna_r, "student_256": student_pna_r,
                                     "retention": student_pna_r / genbio_pna_r if genbio_pna_r else float("nan")},
    }
