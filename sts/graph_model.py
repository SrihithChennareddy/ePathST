"""Steps 16 & 20 (Sections 19-20), revised per Final_Project_V2.txt Phases
10-15: PNA / Scale-Aware PNA / GATv2 graph models, all wrapped in the same
RESIDUAL correction on top of the (frozen) patch-only predictor.

    y_patch,i  = patch_predictor(h_i)                         [frozen, Step 14]
    dy_graph,i = GNN(h_i, neighborhood_i)                     [this module]
    y_final,i  = y_patch,i + dy_graph,i

The graph correction heads are initialized near zero so y_final ~= y_patch
at initialization (Section 20 / V2 Phase 11) -- this is what makes
A0-A9 interpretable as "how much does spatial context add on top of what
the patch already predicts," rather than the graph model re-deriving the
whole prediction from scratch. V2 Phase 11 directly attributes PoC_5's
failure mode (patch CNN 0.1875 vs. multiscale PNA falling to 0.1268,
i.e. the graph model made things WORSE) to not having this residual
design tightly enforced with an explicit init-time check -- see
`assert_zero_residual_at_init` below, which V2 requires as a unit test
run before every graph-training call.

Three interchangeable graph-correction backends, all producing the same
{"latent", "gene_delta", "pathway_delta"} output so `ResidualGraphPredictor`
is backend-agnostic:
  - `GraphCorrectionModel`      -- union multiscale PNA (A4/A5/A6/A7): one
    PNA sees all edges (of possibly mixed scale) at once, 20-D union edge
    features (sts.graph_builder.union_edge_features).
  - `ScaleAwareGraphCorrectionModel` -- V2 Phase 12 (A8, the proposed
    final model): three independent PNA branches (local/meso/macro), each
    over ONLY its own scale's edges (17-D per_scale_edge_features), fused
    by a learned softmax gate.
  - `GATv2CorrectionModel`      -- V2 Phase 15 (A9, architecture control):
    identical to GraphCorrectionModel except the message-passing layer is
    GATv2 instead of PNA (sts.gatv2_block.MultiLayerGATv2) -- same input,
    same edge features, same head, same residual formulation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from sts.gatv2_block import MultiLayerGATv2
from sts.pna_block import MultiLayerPNA


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
    """V2 Phase 15 (A9): identical structure to GraphCorrectionModel with
    GATv2 in place of PNA -- the architecture-isolation control. No degree
    histogram needed (GATv2 doesn't use PNA's degree scalers)."""

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
    """V2 Phase 12 -- the proposed final architecture (A8). Three
    independent PNA branches, one per physical scale, each seeing ONLY its
    own scale's edges:

        h_i^0 = W_stem(h_i^CNN)
        h_i^L = PNA_L(h_i^0, E_local)
        h_i^M = PNA_M(h_i^0, E_meso)
        h_i^G = PNA_G(h_i^0, E_macro)
        c_i   = [h_i^0; h_i^L; h_i^M; h_i^G]
        [a_L, a_M, a_G] = softmax(MLP_gate(c_i))
        h_i^spatial = a_L*h_i^L + a_M*h_i^M + a_G*h_i^G

    followed by the same 128->64->{gene,pathway} delta heads,
    zero-initialized exactly as the union model. `deg_local/meso/macro`
    are separate TRAINING-fold degree histograms, one per branch (each
    branch's own PNA needs its own scalers -- V2 Phase 13's per-fold,
    train-only degree-histogram requirement applies to each branch
    independently).

    Interface note: rather than a parallel 3-edge-set data pipeline, this
    reuses the SAME single-`Data`-object PyG plumbing every other backend
    uses (sts/graph_training.py's make_pyg_data/train_graph_model/
    predict_graph_model, unchanged) -- `edge_index`/`edge_attr` carry the
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
    """Wraps a FROZEN patch-only MolecularBottleneckHead (Step 14) with any
    trainable graph-correction backend (GraphCorrectionModel /
    GATv2CorrectionModel / ScaleAwareGraphCorrectionModel); forward()
    returns final = patch + delta. The patch predictor's parameters never
    receive gradients here -- isolating "what does spatial context add" is
    the entire point of the residual design (Section 20).

    Backend-agnostic via *graph_args passthrough: union/GATv2 backends
    take (edge_index, edge_attr); the scale-aware backend takes
    (edges_local, edges_meso, edges_macro) -- either way this class just
    forwards whatever was passed to `self.graph_correction`."""

    def __init__(self, patch_head, graph_correction: nn.Module):
        super().__init__()
        self.patch_head = patch_head
        for p in self.patch_head.parameters():
            p.requires_grad_(False)
        self.patch_head.eval()  # frozen -> always inference mode (disables Dropout/BatchNorm
        # stochasticity); without this, patch_head.py's V2-added Dropout layer made the "frozen"
        # patch predictor non-deterministic across calls -- caught by scripts/16_sanity_checks.py's S1.
        self.graph_correction = graph_correction

    def train(self, mode: bool = True):
        """Keep patch_head in eval() even when the parent module's
        .train() is called (e.g. by the training loop toggling the whole
        ResidualGraphPredictor to train mode for the graph_correction
        branch) -- it must never see BatchNorm/Dropout train-mode
        stochasticity since it's frozen and meant to reproduce
        `ŷ_patch,i` exactly (Section 20 / S1)."""
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
    """V2 Phase 11's required unit test, run once before every graph-
    training call (see scripts/16_sanity_checks.py S3): with the graph
    correction heads freshly zero-initialized, y_final must exactly equal
    y_patch. This directly targets the PoC_5 failure mode where an
    untested graph model could silently start WORSE than patch-only
    (observed there: patch CNN 0.1875 vs. multiscale PNA 0.1268)."""
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
    """V2 Phase 7/19: primary loss is gene-only (pathway_weight=0.0
    default, changed from V1's mandatory 0.2 -- pathway loss becomes an
    ablation swept AFTER the architecture is locked, see
    sts/patch_head.py's matching default change)."""
    from sts.patch_head import patch_head_loss
    fake_pred = {"gene": pred["gene"], "pathway": pred["pathway"]}
    return patch_head_loss(fake_pred, gene_target, pathway_target, pathway_weight)
