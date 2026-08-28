"""Step 6 (Section 9): distillation objective.

L = L_cos + lambda * L_feature, with L_cos the primary term and a small
normalized feature-regression term as a secondary signal. Never uses
unnormalized raw 4608-D MSE as the sole objective (explicitly forbidden)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_distillation_loss(student_projected: torch.Tensor, teacher_embedding: torch.Tensor) -> torch.Tensor:
    """L_cos = 1 - cosine_similarity(normalize(student_proj), normalize(teacher))."""
    s = F.normalize(student_projected, dim=-1)
    t = F.normalize(teacher_embedding, dim=-1)
    cos_sim = (s * t).sum(dim=-1)
    return (1.0 - cos_sim).mean()


def feature_regression_loss(student_projected: torch.Tensor, teacher_embedding: torch.Tensor,
                             kind: str = "smooth_l1") -> torch.Tensor:
    """Normalized feature-regression term (never raw unnormalized MSE)."""
    s = F.normalize(student_projected, dim=-1)
    t = F.normalize(teacher_embedding, dim=-1)
    if kind == "smooth_l1":
        return F.smooth_l1_loss(s, t)
    if kind == "normalized_mse":
        return F.mse_loss(s, t)
    raise ValueError(f"unknown feature loss kind: {kind}")


def distillation_loss(student_projected: torch.Tensor, teacher_embedding: torch.Tensor, lam: float,
                       feature_kind: str = "smooth_l1") -> dict:
    l_cos = cosine_distillation_loss(student_projected, teacher_embedding)
    l_feat = feature_regression_loss(student_projected, teacher_embedding, feature_kind)
    total = l_cos + lam * l_feat
    return {"loss": total, "l_cos": l_cos.detach(), "l_feature": l_feat.detach()}
