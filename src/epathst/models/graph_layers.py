"""Edge-aware message-passing blocks for the graph correction model: PNA (A4-A8) and
GATv2 (A9), structurally mirrored so the A7/A8-vs-A9 ablation isolates only the operator."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, PNAConv
from torch_geometric.utils import degree


def compute_degree_histogram(
    edge_indices: list[torch.Tensor], num_nodes_list: list[int]
) -> torch.Tensor:
    """PNA's amplification/attenuation scalers need the average in-degree of the TRAINING graphs; computed once from the training fold and reused for val/test as a fixed constant (not fit per-graph), so it introduces no leakage."""
    # first pass: find max degree across all graphs
    max_degree = 0
    for edge_index, n in zip(edge_indices, num_nodes_list):
        if edge_index.numel() == 0:
            continue
        d = degree(edge_index[1], num_nodes=n, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # second pass: accumulate degree histogram
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for edge_index, n in zip(edge_indices, num_nodes_list):
        if edge_index.numel() == 0:
            deg[0] += n
            continue
        d = degree(edge_index[1], num_nodes=n, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg


class PNABlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        aggregators: list[str],
        scalers: list[str],
        deg: torch.Tensor,
        edge_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = PNAConv(
            hidden_dim, hidden_dim, aggregators=aggregators, scalers=scalers, deg=deg,
            edge_dim=edge_dim, towers=1, pre_layers=1, post_layers=1,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        return x + h


class MultiLayerPNA(nn.Module):
    """Two PNA blocks by default -- do not start with more layers."""

    def __init__(
        self,
        hidden_dim: int,
        n_layers: int,
        aggregators: list[str],
        scalers: list[str],
        deg: torch.Tensor,
        edge_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            PNABlock(hidden_dim, aggregators, scalers, deg, edge_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, edge_index, edge_attr)
        return x


class GATv2Block(nn.Module):
    """For the A9 architecture-control ablation; mirrors PNABlock's structure exactly (edge-aware conv -> LayerNorm -> GELU -> Dropout -> residual)."""

    def __init__(self, hidden_dim: int, edge_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % heads == 0, (
            "hidden_dim must be divisible by heads for GATv2Conv's concat output"
        )
        self.conv = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, edge_dim=edge_dim,
                               dropout=dropout, concat=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        return x + h  # residual addition, matching PNABlock


class MultiLayerGATv2(nn.Module):
    """Same depth as MultiLayerPNA for a fair, approximately parameter-matched architecture comparison."""

    def __init__(
        self, hidden_dim: int, n_layers: int, edge_dim: int, heads: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [GATv2Block(hidden_dim, edge_dim, heads, dropout) for _ in range(n_layers)]
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, edge_index, edge_attr)
        return x
