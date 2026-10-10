"""PNA, Scale-Aware PNA and GATv2 graph-correction models, each a zero-initialized residual.

The residual sits on the frozen patch-only predictor: y_final = y_patch + GNN(h, neighborhood), so
spatial context is added to the patch prediction rather than re-derived from scratch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from epathst.models.graph_layers import MultiLayerGATv2, MultiLayerPNA


class GraphCorrectionModel(nn.Module):
    """Union multiscale PNA correction.

    A4 (random), A5 (shuffled), A6 (local) and A7 (union multiscale) share this module; only the
    edge sets and features fed in from outside differ.
    """

    def __init__(
        self,
        in_dim: int,
        cfg: dict,
        deg: torch.Tensor,
        edge_dim: int,
        gene_dim: int,
        pathway_dim: int,
    ):
        super().__init__()
        pna_cfg = cfg["pna"]
        self.input_proj = nn.Linear(in_dim, pna_cfg["hidden_dim"])
        self.pna = MultiLayerPNA(
            pna_cfg["hidden_dim"],
            pna_cfg["n_layers"],
            pna_cfg["aggregators"],
            pna_cfg["scalers"],
            deg,
            edge_dim,
            pna_cfg["dropout"],
        )
        self.to_latent = nn.Linear(pna_cfg["hidden_dim"], pna_cfg["molecular_latent_dim"])
        self.act = nn.GELU()
        self.gene_delta_head = nn.Linear(pna_cfg["molecular_latent_dim"], gene_dim)
        self.pathway_delta_head = nn.Linear(pna_cfg["molecular_latent_dim"], pathway_dim)
        _zero_init_delta_heads(self, pna_cfg)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> dict:
        h = self.input_proj(x)
        h = self.pna(h, edge_index, edge_attr)
        z = self.act(self.to_latent(h))
        return {
            "latent": z,
            "gene_delta": self.gene_delta_head(z),
            "pathway_delta": self.pathway_delta_head(z),
        }


class GATv2CorrectionModel(nn.Module):
    """A9: GraphCorrectionModel with GATv2 in place of PNA, the architecture-isolation control.

    No degree histogram is needed since GATv2 has no PNA degree scalers.
    """

    def __init__(self, in_dim: int, cfg: dict, edge_dim: int, gene_dim: int, pathway_dim: int):
        super().__init__()
        pna_cfg = cfg["pna"]
        g = cfg.get("gatv2", {})
        self.input_proj = nn.Linear(in_dim, pna_cfg["hidden_dim"])
        self.gnn = MultiLayerGATv2(
            pna_cfg["hidden_dim"],
            pna_cfg["n_layers"],
            edge_dim,
            heads=g.get("heads", 4),
            dropout=pna_cfg["dropout"],
        )
        self.to_latent = nn.Linear(pna_cfg["hidden_dim"], pna_cfg["molecular_latent_dim"])
        self.act = nn.GELU()
        self.gene_delta_head = nn.Linear(pna_cfg["molecular_latent_dim"], gene_dim)
        self.pathway_delta_head = nn.Linear(pna_cfg["molecular_latent_dim"], pathway_dim)
        _zero_init_delta_heads(self, pna_cfg)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> dict:
        h = self.input_proj(x)
        h = self.gnn(h, edge_index, edge_attr)
        z = self.act(self.to_latent(h))
        return {
            "latent": z,
            "gene_delta": self.gene_delta_head(z),
            "pathway_delta": self.pathway_delta_head(z),
        }


class ScaleAwareGraphCorrectionModel(nn.Module):
    """The proposed final architecture (A8): three independent PNA branches, one per physical
    scale, each seeing ONLY its own scale's edges:

        h_i^0 = W_stem(h_i^CNN)
        h_i^L = PNA_L(h_i^0, E_local)
        h_i^M = PNA_M(h_i^0, E_meso)
        h_i^G = PNA_G(h_i^0, E_macro)
        c_i   = [h_i^0; h_i^L; h_i^M; h_i^G]
        [a_L, a_M, a_G] = softmax(MLP_gate(c_i))
        h_i^spatial = a_L*h_i^L + a_M*h_i^M + a_G*h_i^G

    followed by the same delta heads, zero-initialized exactly as in the union model.
    `deg_local/meso/macro` are separate train-fold-only degree histograms, one per branch, since
    each branch's own PNA needs its own scalers.

    Reuses the same single-`Data`-object PyG plumbing every other backend uses (graph_training.py)
    instead of a parallel 3-edge-set pipeline: `edge_index`/`edge_attr` carry the UNION of all
    three scales' edges (17-D per_scale_edge_features, no one-hot needed), and a parallel
    `edge_scale` LongTensor (0=local, 1=meso, 2=macro) tags each edge's branch; forward() masks by
    `edge_scale` to route edges to branches.
    """

    def __init__(
        self,
        in_dim: int,
        cfg: dict,
        deg_local: torch.Tensor,
        deg_meso: torch.Tensor,
        deg_macro: torch.Tensor,
        edge_dim: int,
        gene_dim: int,
        pathway_dim: int,
    ):
        super().__init__()
        pna_cfg = cfg["pna"]
        hd = pna_cfg["hidden_dim"]
        self.input_proj = nn.Linear(in_dim, hd)
        self.pna_local = MultiLayerPNA(
            hd,
            pna_cfg["n_layers"],
            pna_cfg["aggregators"],
            pna_cfg["scalers"],
            deg_local,
            edge_dim,
            pna_cfg["dropout"],
        )
        self.pna_meso = MultiLayerPNA(
            hd,
            pna_cfg["n_layers"],
            pna_cfg["aggregators"],
            pna_cfg["scalers"],
            deg_meso,
            edge_dim,
            pna_cfg["dropout"],
        )
        self.pna_macro = MultiLayerPNA(
            hd,
            pna_cfg["n_layers"],
            pna_cfg["aggregators"],
            pna_cfg["scalers"],
            deg_macro,
            edge_dim,
            pna_cfg["dropout"],
        )

        gate_cfg = cfg.get("scale_aware_pna", {})
        gate_hidden = gate_cfg.get("gate_hidden_dim", hd)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hd * 4, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 3),
        )

        self.to_latent = nn.Linear(hd, pna_cfg["molecular_latent_dim"])
        self.act = nn.GELU()
        self.gene_delta_head = nn.Linear(pna_cfg["molecular_latent_dim"], gene_dim)
        self.pathway_delta_head = nn.Linear(pna_cfg["molecular_latent_dim"], pathway_dim)
        _zero_init_delta_heads(self, pna_cfg)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_scale: torch.Tensor,
    ) -> dict:
        """`edge_scale` is an (E,) LongTensor in {0, 1, 2} = {local, meso, macro}.

        It is aligned to `edge_index`'s columns.
        """
        h0 = self.input_proj(x)
        mask_l, mask_m, mask_g = edge_scale == 0, edge_scale == 1, edge_scale == 2
        hL = self.pna_local(h0, edge_index[:, mask_l], edge_attr[mask_l])
        hM = self.pna_meso(h0, edge_index[:, mask_m], edge_attr[mask_m])
        hG = self.pna_macro(h0, edge_index[:, mask_g], edge_attr[mask_g])

        c = torch.cat([h0, hL, hM, hG], dim=-1)
        alpha = torch.softmax(self.gate_mlp(c), dim=-1)  # (N, 3)
        h_spatial = alpha[:, 0:1] * hL + alpha[:, 1:2] * hM + alpha[:, 2:3] * hG

        z = self.act(self.to_latent(h_spatial))
        return {
            "latent": z,
            "gene_delta": self.gene_delta_head(z),
            "pathway_delta": self.pathway_delta_head(z),
            "scale_weights": alpha.detach(),
        }


def _zero_init_delta_heads(model: nn.Module, pna_cfg: dict) -> None:
    """Zero the delta heads so y_final == y_patch at init; off if residual_zero_init is false."""
    if pna_cfg.get("residual_zero_init", True):
        nn.init.zeros_(model.gene_delta_head.weight)
        nn.init.zeros_(model.gene_delta_head.bias)
        nn.init.zeros_(model.pathway_delta_head.weight)
        nn.init.zeros_(model.pathway_delta_head.bias)


class ResidualGraphPredictor(nn.Module):
    """A FROZEN patch-only MolecularBottleneckHead plus a trainable graph-correction backend.

    The backend is GraphCorrectionModel, GATv2CorrectionModel or ScaleAwareGraphCorrectionModel.
    forward() returns final = patch + delta; the patch predictor receives no gradients, which
    isolates what spatial context adds. `*graph_args` is passed through to the backend (union and
    GATv2 take edge_index, edge_attr; scale-aware also takes edge_scale).
    """

    def __init__(self, patch_head, graph_correction: nn.Module):
        super().__init__()
        self.patch_head = patch_head
        for p in self.patch_head.parameters():
            p.requires_grad_(False)
        # Frozen means always eval mode; otherwise Dropout makes the "frozen" patch predictor
        # non-deterministic across calls (caught by sanity check S1).
        self.patch_head.eval()
        self.graph_correction = graph_correction

    def train(self, mode: bool = True):
        """Keep patch_head in eval() even when the parent's .train() is called.

        A frozen head must reproduce y_patch exactly, so it can never see BatchNorm/Dropout
        train-mode stochasticity.
        """
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
            "gene": gene_final,
            "pathway": pathway_final,
            "gene_patch_only": patch_out["gene"],
            "pathway_patch_only": patch_out["pathway"],
            "gene_delta": corr["gene_delta"],
            "pathway_delta": corr["pathway_delta"],
            **({"scale_weights": corr["scale_weights"]} if "scale_weights" in corr else {}),
        }


def assert_zero_residual_at_init(
    model: ResidualGraphPredictor, x: torch.Tensor, *graph_args, tol: float = 1e-5
) -> None:
    """Run before every graph-training call: with freshly zero-initialized correction heads,
    y_final must equal y_patch to within `tol`.

    Guards against the graph model silently starting out worse than patch-only; PoC_5 once
    regressed below patch-only performance before this check existed.
    """
    model.eval()
    with torch.no_grad():
        out = model(x, *graph_args)
        patch_only = model.patch_head(x)
    max_abs_gene = (out["gene"] - patch_only["gene"]).abs().max().item()
    max_abs_pathway = (out["pathway"] - patch_only["pathway"]).abs().max().item()
    assert (
        max_abs_gene < tol
    ), f"zero-residual check FAILED for genes: max_abs={max_abs_gene} >= tol={tol}"
    assert (
        max_abs_pathway < tol
    ), f"zero-residual check FAILED for pathways: max_abs={max_abs_pathway} >= tol={tol}"
    model.train()


def graph_loss(
    pred: dict, gene_target: torch.Tensor, pathway_target: torch.Tensor, pathway_weight: float = 0.0
) -> dict:
    """Primary loss is gene-only by default (pathway_weight=0.0).

    The pathway loss weight is swept as a separate ablation once the architecture is locked; see
    the matching default in src/epathst/models/patch_head.py.
    """
    from epathst.models.patch_head import patch_head_loss

    fake_pred = {"gene": pred["gene"], "pathway": pred["pathway"]}
    return patch_head_loss(fake_pred, gene_target, pathway_target, pathway_weight)
