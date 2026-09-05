"""Patch-only molecular bottleneck head -- defines ŷ_patch,i, the anchor
prediction that the residual graph model must never destroy (the graph
correction is added on top of this, not a replacement for it).

Primary loss is gene-only by default (pathway_weight=0.0): the pathway
auxiliary term showed no consistent benefit in earlier pilot runs, so it's
swept separately as a post-lock ablation (lambda in {0, 0.05, 0.10, 0.20})
rather than baked into the primary objective. The pathway head itself stays
in the model for that sweep and for post-hoc pathway analysis."""
from __future__ import annotations

import torch
import torch.nn as nn


class MolecularBottleneckHead(nn.Module):
    """h_i (320 or 256) -> bottleneck -> {gene head (HVG1000), pathway
    head (14)}. Bottleneck: Linear(in->128) -> LayerNorm -> GELU ->
    Dropout -> Linear(128->64) -> GELU -> heads. LayerNorm+Dropout is
    applied only to the first (widest, most overfit-prone) layer."""

    def __init__(self, in_dim: int, bottleneck_dims: list[int], gene_dim: int, pathway_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        # first (widest) layer gets LayerNorm + Dropout, remaining layers do not
        dims = [in_dim] + list(bottleneck_dims[1:])
        layers = [nn.Linear(dims[0], dims[1]), nn.LayerNorm(dims[1]), nn.GELU(), nn.Dropout(dropout)]
        for d_in, d_out in zip(dims[1:-1], dims[2:]):
            layers += [nn.Linear(d_in, d_out), nn.GELU()]
        self.trunk = nn.Sequential(*layers)
        self.molecular_latent_dim = dims[-1]
        self.gene_head = nn.Linear(dims[-1], gene_dim)
        self.pathway_head = nn.Linear(dims[-1], pathway_dim)

    def forward(self, h: torch.Tensor) -> dict:
        z = self.trunk(h)
        return {"latent": z, "gene": self.gene_head(z), "pathway": self.pathway_head(z)}


def patch_head_loss(pred: dict, gene_target: torch.Tensor, pathway_target: torch.Tensor,
                     pathway_weight: float = 0.0) -> dict:
    """Primary loss is gene-only (pathway_weight=0.0 by default); pass a
    nonzero pathway_weight for the post-lock ablation sweep."""
    # mask missing targets, zero-fill so they don't contribute to the loss
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
    return MolecularBottleneckHead(in_dim, ph["bottleneck"], gene_dim, pathway_dim, dropout=ph.get("dropout", 0.1))
