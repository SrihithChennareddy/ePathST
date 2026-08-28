"""Steps 16 & 20 (Sections 19-20): PNA graph model with RESIDUAL correction
on top of the (frozen) patch-only predictor.

    y_patch,i  = patch_predictor(h_i)                         [frozen, Step 14]
    dy_graph,i = GNN(h_i, neighborhood_i)                     [this module]
    y_final,i  = y_patch,i + dy_graph,i

The graph correction heads are initialized near zero so y_final ~= y_patch
at initialization (Section 20) -- this is what makes A0-A9 interpretable as
"how much does spatial context add on top of what the patch already
predicts," rather than the graph model re-deriving the whole prediction
from scratch and potentially destroying a working patch-only signal."""
from __future__ import annotations

import torch
import torch.nn as nn

from sts.pna_block import MultiLayerPNA


class GraphCorrectionModel(nn.Module):
    def __init__(self, in_dim: int, cfg: dict, deg: torch.Tensor, edge_dim: int, gene_dim: int, pathway_dim: int):
        super().__init__()
        p = cfg["pna"]
        self.input_proj = nn.Linear(in_dim, p["hidden_dim"])
        self.pna = MultiLayerPNA(p["hidden_dim"], p["n_layers"], p["aggregators"], p["scalers"], deg, edge_dim,
                                  p["dropout"])
        self.to_latent = nn.Linear(p["hidden_dim"], p["molecular_latent_dim"])
        self.act = nn.GELU()
        self.gene_delta_head = nn.Linear(p["molecular_latent_dim"], gene_dim)
        self.pathway_delta_head = nn.Linear(p["molecular_latent_dim"], pathway_dim)

        if p.get("residual_zero_init", True):
            nn.init.zeros_(self.gene_delta_head.weight)
            nn.init.zeros_(self.gene_delta_head.bias)
            nn.init.zeros_(self.pathway_delta_head.weight)
            nn.init.zeros_(self.pathway_delta_head.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> dict:
        h = self.input_proj(x)
        h = self.pna(h, edge_index, edge_attr)
        z = self.act(self.to_latent(h))
        return {"latent": z, "gene_delta": self.gene_delta_head(z), "pathway_delta": self.pathway_delta_head(z)}


class ResidualGraphPredictor(nn.Module):
    """Wraps a FROZEN patch-only MolecularBottleneckHead (Step 14) with a
    trainable GraphCorrectionModel; forward() returns final = patch + delta.
    The patch predictor's parameters never receive gradients here --
    isolating "what does spatial context add" is the entire point of the
    residual design (Section 20)."""

    def __init__(self, patch_head, graph_correction: GraphCorrectionModel):
        super().__init__()
        self.patch_head = patch_head
        for p in self.patch_head.parameters():
            p.requires_grad_(False)
        self.graph_correction = graph_correction

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> dict:
        with torch.no_grad():
            patch_out = self.patch_head(x)
        corr = self.graph_correction(x, edge_index, edge_attr)
        gene_final = patch_out["gene"] + corr["gene_delta"]
        pathway_final = patch_out["pathway"] + corr["pathway_delta"]
        return {
            "gene": gene_final, "pathway": pathway_final,
            "gene_patch_only": patch_out["gene"], "pathway_patch_only": patch_out["pathway"],
            "gene_delta": corr["gene_delta"], "pathway_delta": corr["pathway_delta"],
        }


def graph_loss(pred: dict, gene_target: torch.Tensor, pathway_target: torch.Tensor,
               pathway_weight: float = 0.2) -> dict:
    from sts.patch_head import patch_head_loss
    fake_pred = {"gene": pred["gene"], "pathway": pred["pathway"]}
    return patch_head_loss(fake_pred, gene_target, pathway_target, pathway_weight)
