"""Edge-aware GATv2 block for the A9 architecture-control ablation. Mirrors
pathology_st.pna_block.PNABlock's structure exactly (edge-aware conv -> LayerNorm ->
GELU -> Dropout -> residual) so A7/A8 (PNA) and A9 (GATv2) differ only in
the message-passing operator -- everything else (features, graph, head,
folds) stays identical."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GATv2Block(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads for GATv2Conv's concat output"
        self.conv = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim,
                               dropout=dropout, concat=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        return x + h  # residual addition, matching PNABlock


class MultiLayerGATv2(nn.Module):
    """Same depth as pathology_st.pna_block.MultiLayerPNA for a fair, approximately
    parameter-matched architecture comparison."""

    def __init__(self, hidden_dim: int, n_layers: int, edge_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([GATv2Block(hidden_dim, edge_dim, heads, dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, edge_index, edge_attr)
        return x
