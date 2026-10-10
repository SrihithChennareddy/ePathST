"""ConvNeXt V2-Atto student backbone and its training-only projection head to the teacher space.

The head maps to the 1536-D H-Optimus-0 embedding. It is excluded from the reported final-encoder
parameter count and physically deleted after distillation (see `strip_to_encoder`).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvNeXtV2AttoStudent(nn.Module):
    """224x224 H&E -> ConvNeXt V2-Atto -> 320-D pooled feature.

    `pretrained="imagenet"` gives S0/S1 (timm ImageNet weights); `pretrained=None` gives S2
    (random init, then distilled).
    """

    def __init__(self, pretrained: str | None = "imagenet"):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "convnextv2_atto",
            pretrained=(pretrained == "imagenet"),
            num_classes=0,
        )
        self.backbone_dim = self.backbone.num_features
        assert self.backbone_dim > 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # (B, backbone_dim), 320 for Atto


class ProjectionHead(nn.Module):
    """Training-only MLP (320 -> 1024 -> GELU -> 1536) onto the teacher embedding dimension."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownstreamProjection(nn.Module):
    """Optional 320 -> 256 projection learned in HEST downstream training.

    Separate from the (deleted) teacher-space projection head.
    """

    def __init__(self, in_dim: int = 320, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DistillationStudent(nn.Module):
    """Encoder + projection head, used only during distillation training and eval.

    After locking, only `.encoder` survives: `strip_to_encoder()` returns a
    ConvNeXtV2AttoStudent-compatible state_dict with the projection head removed.
    """

    def __init__(self, cfg: dict, pretrained: str | None = "imagenet"):
        super().__init__()
        self.encoder = ConvNeXtV2AttoStudent(pretrained=pretrained)
        head_dims = cfg["student"]["projection_head"]
        self.projection_head = ProjectionHead(head_dims[0], head_dims[1], head_dims[2])

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.encoder(x)
        proj = self.projection_head(feat)
        return {"backbone_feature": feat, "projected": proj}

    def strip_to_encoder(self) -> dict:
        """Drop the 1536-D teacher projection head; return a state_dict of the backbone only."""
        return {
            k[len("encoder.") :]: v
            for k, v in self.state_dict().items()
            if k.startswith("encoder.")
        }


def load_locked_encoder(state_dict: dict, device: str = "cuda") -> ConvNeXtV2AttoStudent:
    """Rebuild the encoder from a `strip_to_encoder()` state_dict, in eval mode."""
    model = ConvNeXtV2AttoStudent(pretrained=None)
    model.load_state_dict(state_dict)
    return model.to(device).eval()
