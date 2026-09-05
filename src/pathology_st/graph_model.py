"""PNA / Scale-Aware PNA / GATv2 graph models, all wrapped in the same
RESIDUAL correction on top of the (frozen) patch-only predictor:

    y_patch,i  = patch_predictor(h_i)          [frozen]
    dy_graph,i = GNN(h_i, neighborhood_i)      [this module]
    y_final,i  = y_patch,i + dy_graph,i

The graph correction heads are zero-initialized so y_final == y_patch at
initialization -- this is what makes the A0-A9 grid interpretable as "how
much does spatial context add on top of the patch prediction," rather
than the graph model re-deriving everything from scratch. An earlier
project iteration (PoC_5) once regressed below patch-only performance
because this wasn't enforced; `assert_zero_residual_at_init` below is the
init-time check that now guards against that class of bug on every
graph-training call.

Three interchangeable graph-correction backends, all producing the same
{"latent", "gene_delta", "pathway_delta"} output so `ResidualGraphPredictor`
is backend-agnostic:
  - `GraphCorrectionModel` -- union multiscale PNA (A4/A5/A6/A7): one PNA
    sees all edges (of possibly mixed scale) at once, 20-D union edge
    features (pathology_st.graph_builder.union_edge_features).
  - `ScaleAwareGraphCorrectionModel` -- the proposed final model (A8):
    three independent PNA branches (local/meso/macro), each over ONLY its
    own scale's edges (17-D per_scale_edge_features), fused by a learned
    softmax gate.
  - `GATv2CorrectionModel` -- architecture control (A9): identical to
    GraphCorrectionModel except the message-passing layer is GATv2
    instead of PNA -- same input, edge features, head, and residual
    formulation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pathology_st.gatv2_block import MultiLayerGATv2
from pathology_st.pna_block import MultiLayerPNA


class GraphCorrectionModel(nn.Module):
    """Union multiscale PNA correction (A4 random / A5 shuffled / A6 local
    / A7 union-multiscale -- same module, different edge sets/features fed
    in from outside)."""

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
        _zero_init_delta_heads(self, p)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> dict:
        h = self.input_proj(x)
        h = self.pna(h, edge_index, edge_attr)
        z = self.act(self.to_latent(h))
        return {"latent": z, "gene_delta": self.gene_delta_head(z), "pathway_delta": self.pathway_delta_head(z)}


class GATv2CorrectionModel(nn.Module):
    """A9: identical structure to GraphCorrectionModel with GATv2 in place
    of PNA -- the architecture-isolation control. No degree histogram
    needed (GATv2 doesn't use PNA's degree scalers)."""

    def __init__(self, in_dim: int, cfg: dict, edge_dim: int, gene_dim: int, pathway_dim: int):
        super().__init__()
        p = cfg["pna"]
        g = cfg.get("gatv2", {})
        self.input_proj = nn.Linear(in_dim, p["hidden_dim"])
        self.gnn = MultiLayerGATv2(p["hidden_dim"], p["n_layers"], edge_dim,
                                    heads=g.get("heads", 4), dropout=p["dropout"])
        self.to_latent = nn.Linear(p["hidden_dim"], p["molecular_latent_dim"])
        self.act = nn.GELU()
        self.gene_delta_head = nn.Linear(p["molecular_latent_dim"], gene_dim)
        self.pathway_delta_head = nn.Linear(p["molecular_latent_dim"], pathway_dim)
        _zero_init_delta_heads(self, p)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> dict:
        h = self.input_proj(x)
        h = self.gnn(h, edge_index, edge_attr)
        z = self.act(self.to_latent(h))
        return {"latent": z, "gene_delta": self.gene_delta_head(z), "pathway_delta": self.pathway_delta_head(z)}


class ScaleAwareGraphCorrectionModel(nn.Module):
    """The proposed final architecture (A8). Three independent PNA
    branches, one per physical scale, each seeing ONLY its own scale's edges:

        h_i^0 = W_stem(h_i^CNN)
        h_i^L = PNA_L(h_i^0, E_local)
        h_i^M = PNA_M(h_i^0, E_meso)
        h_i^G = PNA_G(h_i^0, E_macro)
        c_i   = [h_i^0; h_i^L; h_i^M; h_i^G]
        [a_L, a_M, a_G] = softmax(MLP_gate(c_i))
        h_i^spatial = a_L*h_i^L + a_M*h_i^M + a_G*h_i^G

    followed by the same delta heads, zero-initialized exactly as the
    union model. `deg_local/meso/macro` are separate train-fold-only
    degree histograms, one per branch, since each branch's own PNA needs
    its own scalers.

    Interface note: rather than a parallel 3-edge-set data pipeline, this
    reuses the same single-`Data`-object PyG plumbing every other backend
    uses (src/pathology_st/graph_training.py). `edge_index`/`edge_attr` carry the
    UNION of all three scales' edges (17-D per_scale_edge_features, no
    one-hot needed), and a parallel `edge_scale` LongTensor (0=local,
    1=meso, 2=macro) tags which branch each edge belongs to. forward()
    masks by `edge_scale` internally to route each edge to its branch."""

    def __init__(self, in_dim: int, cfg: dict, deg_local: torch.Tensor, deg_meso: torch.Tensor,
                 deg_macro: torch.Tensor, edge_dim: int, gene_dim: int, pathway_dim: int):
        super().__init__()
        p = cfg["pna"]
        hd = p["hidden_dim"]
        self.input_proj = nn.Linear(in_dim, hd)
        self.pna_local = MultiLayerPNA(hd, p["n_layers"], p["aggregators"], p["scalers"], deg_local, edge_dim, p["dropout"])
        self.pna_meso = MultiLayerPNA(hd, p["n_layers"], p["aggregators"], p["scalers"], deg_meso, edge_dim, p["dropout"])
        self.pna_macro = MultiLayerPNA(hd, p["n_layers"], p["aggregators"], p["scalers"], deg_macro, edge_dim, p["dropout"])

        gate_cfg = cfg.get("scale_aware_pna", {})
        gate_hidden = gate_cfg.get("gate_hidden_dim", hd)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hd * 4, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 3),
        )

        self.to_latent = nn.Linear(hd, p["molecular_latent_dim"])
        self.act = nn.GELU()
        self.gene_delta_head = nn.Linear(p["molecular_latent_dim"], gene_dim)
        self.pathway_delta_head = nn.Linear(p["molecular_latent_dim"], pathway_dim)
        _zero_init_delta_heads(self, p)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                edge_scale: torch.Tensor) -> dict:
        """`edge_scale` is a (E,) LongTensor with values in {0, 1, 2} =
        {local, meso, macro}, aligned to `edge_index`'s columns."""
        h0 = self.input_proj(x)
        mask_l, mask_m, mask_g = edge_scale == 0, edge_scale == 1, edge_scale == 2
        hL = self.pna_local(h0, edge_index[:, mask_l], edge_attr[mask_l])
        hM = self.pna_meso(h0, edge_index[:, mask_m], edge_attr[mask_m])
        hG = self.pna_macro(h0, edge_index[:, mask_g], edge_attr[mask_g])

        c = torch.cat([h0, hL, hM, hG], dim=-1)
        alpha = torch.softmax(self.gate_mlp(c), dim=-1)  # (N, 3)
        h_spatial = (alpha[:, 0:1] * hL + alpha[:, 1:2] * hM + alpha[:, 2:3] * hG)

        z = self.act(self.to_latent(h_spatial))
        return {"latent": z, "gene_delta": self.gene_delta_head(z), "pathway_delta": self.pathway_delta_head(z),
                "scale_weights": alpha.detach()}


def _zero_init_delta_heads(model: nn.Module, pna_cfg: dict) -> None:
    if pna_cfg.get("residual_zero_init", True):
        nn.init.zeros_(model.gene_delta_head.weight)
        nn.init.zeros_(model.gene_delta_head.bias)
        nn.init.zeros_(model.pathway_delta_head.weight)
        nn.init.zeros_(model.pathway_delta_head.bias)


class ResidualGraphPredictor(nn.Module):
    """Wraps a FROZEN patch-only MolecularBottleneckHead with any
    trainable graph-correction backend (GraphCorrectionModel /
    GATv2CorrectionModel / ScaleAwareGraphCorrectionModel); forward()
    returns final = patch + delta. The patch predictor's parameters never
    receive gradients here -- isolating "what does spatial context add"
    is the entire point of the residual design.

    Backend-agnostic via *graph_args passthrough: union/GATv2 backends
    take (edge_index, edge_attr); the scale-aware backend also takes
    edge_scale -- either way this class just forwards whatever was passed
    to `self.graph_correction`."""

    def __init__(self, patch_head, graph_correction: nn.Module):
        super().__init__()
        self.patch_head = patch_head
        for p in self.patch_head.parameters():
            p.requires_grad_(False)
        # frozen -> always eval mode; without this, patch_head's Dropout layer would make the
        # "frozen" patch predictor non-deterministic across calls (caught by sanity check S1).
        self.patch_head.eval()
        self.graph_correction = graph_correction

    def train(self, mode: bool = True):
        """Keep patch_head in eval() even when the parent module's
        .train() is called -- it's frozen and must reproduce y_patch
        exactly, so it can never see BatchNorm/Dropout train-mode
        stochasticity."""
        super().train(mode)
        self.patch_head.eval()
        return self

    def forward(self, x: torch.Tensor, *graph_args) -> dict:
        with torch.no_grad():
            patch_out = self.patch_head(x)
        corr = self.graph_correction(x, *graph_args)
        gene_final = patch_out["gene"] + corr["gene_delta"]
        pathway_final = patch_out["pathway"] + corr["pathway_delta"]
        return {
            "gene": gene_final, "pathway": pathway_final,
            "gene_patch_only": patch_out["gene"], "pathway_patch_only": patch_out["pathway"],
            "gene_delta": corr["gene_delta"], "pathway_delta": corr["pathway_delta"],
            **({"scale_weights": corr["scale_weights"]} if "scale_weights" in corr else {}),
        }


def assert_zero_residual_at_init(model: ResidualGraphPredictor, x: torch.Tensor, *graph_args,
                                  tol: float = 1e-5) -> None:
    """Run before every graph-training call: with the graph correction
    heads freshly zero-initialized, y_final must exactly equal y_patch.
    Guards against the graph model silently starting out worse than
    patch-only (the PoC_5 failure mode this module's docstring describes)."""
    model.eval()
    with torch.no_grad():
        out = model(x, *graph_args)
        patch_only = model.patch_head(x)
    max_abs_gene = (out["gene"] - patch_only["gene"]).abs().max().item()
    max_abs_pathway = (out["pathway"] - patch_only["pathway"]).abs().max().item()
    assert max_abs_gene < tol, f"zero-residual check FAILED for genes: max_abs={max_abs_gene} >= tol={tol}"
    assert max_abs_pathway < tol, f"zero-residual check FAILED for pathways: max_abs={max_abs_pathway} >= tol={tol}"
    model.train()


def graph_loss(pred: dict, gene_target: torch.Tensor, pathway_target: torch.Tensor,
               pathway_weight: float = 0.0) -> dict:
    """Primary loss is gene-only by default (pathway_weight=0.0); pathway
    loss weight is swept separately as an ablation once the architecture
    is locked (see src/pathology_st/patch_head.py's matching default)."""
    from pathology_st.patch_head import patch_head_loss
    fake_pred = {"gene": pred["gene"], "pathway": pred["pathway"]}
    return patch_head_loss(fake_pred, gene_target, pathway_target, pathway_weight)
