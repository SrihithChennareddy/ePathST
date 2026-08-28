"""Step 14 (Section 17): patch-only molecular bottleneck. This model is
critical: it defines ŷ_patch,i, the baseline that the graph model must
never destroy (Section 20's residual design depends on it)."""
from __future__ import annotations

import torch
import torch.nn as nn


class MolecularBottleneckHead(nn.Module):
    """h_i (320 or 256) -> bottleneck -> {gene head (1000), pathway head (14)}.
    Loss: L = MSE_gene + 0.2 * MSE_pathway (Section 17)."""

    def __init__(self, in_dim: int, bottleneck_dims: list[int], gene_dim: int, pathway_dim: int):
        super().__init__()
        layers = []
        d = in_dim
        for h in bottleneck_dims[1:]:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.molecular_latent_dim = d
        self.gene_head = nn.Linear(d, gene_dim)
        self.pathway_head = nn.Linear(d, pathway_dim)

    def forward(self, h: torch.Tensor) -> dict:
        z = self.trunk(h)
        return {"latent": z, "gene": self.gene_head(z), "pathway": self.pathway_head(z)}


def patch_head_loss(pred: dict, gene_target: torch.Tensor, pathway_target: torch.Tensor,
                     pathway_weight: float = 0.2) -> dict:
    gene_mask = ~torch.isnan(gene_target)
    path_mask = ~torch.isnan(pathway_target)
    g_t = torch.where(gene_mask, gene_target, torch.zeros_like(gene_target))
    p_t = torch.where(path_mask, pathway_target, torch.zeros_like(pathway_target))
    mse_gene = (((pred["gene"] - g_t) ** 2) * gene_mask).sum() / gene_mask.sum().clamp(min=1)
    mse_pathway = (((pred["pathway"] - p_t) ** 2) * path_mask).sum() / path_mask.sum().clamp(min=1)
    total = mse_gene + pathway_weight * mse_pathway
    return {"loss": total, "mse_gene": mse_gene.detach(), "mse_pathway": mse_pathway.detach()}


def build_patch_head(cfg: dict, in_dim: int, gene_dim: int, pathway_dim: int) -> MolecularBottleneckHead:
    ph = cfg["patch_head"]
    return MolecularBottleneckHead(in_dim, ph["bottleneck"], gene_dim, pathway_dim)
