"""Fair teacher-vs-student comparison: raw 1536-D H-Optimus-0 embedding reduced via training-fold-only
PCA to 256-D (matches student's 320->256 projection) so both feed identical Ridge/MLP/PNA heads."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


class TrainOnlyPCA:
    """Fits sklearn PCA on TRAIN-patient rows only, transforms train/val/test identically; tracks fit spot_ids so leakage can be asserted (same convention as expression.TrainOnlyStandardizer)."""

    def __init__(self, n_components: int = 256, random_state: int = 0):
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self._fit_ids: set = set()

    def fit(self, X_train: np.ndarray, ids=None) -> "TrainOnlyPCA":
        # train only -- fitting on pooled train+test would leak test-set structure into the basis
        self.pca.fit(X_train)
        if ids is not None:
            self._fit_ids = set(ids)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)

    def assert_no_leakage(self, other_ids) -> None:
        overlap = self._fit_ids & set(other_ids)
        assert not overlap, (
            f"PCA was fit using {len(overlap)} ids that also appear in this eval/test set"
        )

    @property
    def explained_variance_ratio_sum(self) -> float:
        return float(self.pca.explained_variance_ratio_.sum())


def fit_ridge_baseline(X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> Ridge:
    """Simplest of the three comparison heads: Ridge regression directly on the (PCA-reduced or student) embedding, fit on training patients only."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def hoptimus_pca_comparison_report(
    hoptimus_ridge_r: float,
    student_ridge_r: float,
    hoptimus_mlp_r: float,
    student_mlp_r: float,
    hoptimus_pna_r: float,
    student_pna_r: float,
    pca_explained_variance: float,
) -> dict:
    """Assembles the three-head x two-encoder comparison table plus the retention question it answers: how much of the 1.1B-parameter teacher survives in the ~3.4M-parameter student?"""
    return {
        "pca_explained_variance_ratio": pca_explained_variance,
        "ridge": {
            "hoptimus_pca256": hoptimus_ridge_r,
            "student_256": student_ridge_r,
            "retention": student_ridge_r / hoptimus_ridge_r if hoptimus_ridge_r else float("nan"),
        },
        "molecular_mlp": {
            "hoptimus_pca256": hoptimus_mlp_r,
            "student_256": student_mlp_r,
            "retention": student_mlp_r / hoptimus_mlp_r if hoptimus_mlp_r else float("nan"),
        },
        "residual_multiscale_pna": {
            "hoptimus_pca256": hoptimus_pna_r,
            "student_256": student_pna_r,
            "retention": student_pna_r / hoptimus_pna_r if hoptimus_pna_r else float("nan"),
        },
    }
