"""Residual-graph-model training loop; one shared train/eval/predict path for three interchangeable
backends (pna_union/scale_aware/gatv2, see graph_model.py) -- one graph per patient, 1-4 per optimizer step."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from pathology_st.graph_model import (
    GATv2CorrectionModel,
    GraphCorrectionModel,
    ResidualGraphPredictor,
    ScaleAwareGraphCorrectionModel,
    assert_zero_residual_at_init,
    graph_loss,
)
from pathology_st.pna_block import compute_degree_histogram
from pathology_st.utils import save_json, seed_everything


def make_pyg_data(
    features: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    gene_target: np.ndarray,
    pathway_target: np.ndarray,
    patient_id: str,
    edge_scale: np.ndarray | None = None,
) -> Data:
    """`edge_scale` (optional): (E,) int array {0,1,2}={local,meso,macro}, required only for backend="scale_aware"."""
    kwargs = {}
    if edge_scale is not None:
        kwargs["edge_scale"] = torch.tensor(edge_scale, dtype=torch.long)
    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y_gene=torch.tensor(gene_target, dtype=torch.float32),
        y_pathway=torch.tensor(pathway_target, dtype=torch.float32),
        patient_id=patient_id,
        **kwargs,
    )


def _build_correction_model(
    backend: str,
    in_dim: int,
    cfg: dict,
    train_graphs: list[Data],
    edge_dim: int,
    gene_dim: int,
    pathway_dim: int,
):
    """Constructs the right graph_correction module for `backend`; degree histogram(s) are computed from `train_graphs` only, never val/test."""
    if backend == "pna_union":
        deg = compute_degree_histogram(
            [g.edge_index for g in train_graphs], [g.num_nodes for g in train_graphs]
        )
        return GraphCorrectionModel(in_dim, cfg, deg, edge_dim, gene_dim, pathway_dim)
    if backend == "gatv2":
        return GATv2CorrectionModel(in_dim, cfg, edge_dim, gene_dim, pathway_dim)
    if backend == "scale_aware":
        def _deg_for_scale(s: int):
            eis = [g.edge_index[:, g.edge_scale == s] for g in train_graphs]
            ns = [g.num_nodes for g in train_graphs]
            return compute_degree_histogram(eis, ns)
        deg_local, deg_meso, deg_macro = _deg_for_scale(0), _deg_for_scale(1), _deg_for_scale(2)
        return ScaleAwareGraphCorrectionModel(
            in_dim, cfg, deg_local, deg_meso, deg_macro, edge_dim, gene_dim, pathway_dim
        )
    raise ValueError(f"unknown backend: {backend}")


def _forward_args(batch, backend: str) -> tuple:
    if backend == "scale_aware":
        return (batch.edge_index, batch.edge_attr, batch.edge_scale)
    return (batch.edge_index, batch.edge_attr)


def train_graph_model(
    patch_head,
    train_graphs: list[Data],
    val_graphs: list[Data],
    cfg: dict,
    out_path: Path,
    seed: int,
    device: str = "cuda",
    graphs_per_step: int = 4,
    backend: str = "pna_union",
) -> dict:
    seed_everything(seed)
    p = cfg["pna"]
    edge_dim = train_graphs[0].edge_attr.shape[1] if train_graphs[0].edge_attr.numel() else 20

    # build model
    in_dim = train_graphs[0].x.shape[1]
    gene_dim = train_graphs[0].y_gene.shape[1]
    pathway_dim = train_graphs[0].y_pathway.shape[1]
    graph_corr = _build_correction_model(
        backend, in_dim, cfg, train_graphs, edge_dim, gene_dim, pathway_dim
    )
    model = ResidualGraphPredictor(patch_head, graph_corr).to(device)

    # zero-residual must hold before any graph training step (see graph_model.assert_zero_residual_at_init)
    probe = Batch.from_data_list([train_graphs[0]]).to(device)
    assert_zero_residual_at_init(model, probe.x, *_forward_args(probe, backend))

    opt = torch.optim.AdamW(
        model.graph_correction.parameters(), lr=p["lr"], weight_decay=p["weight_decay"]
    )

    best_val, best_epoch, patience_ctr = float("inf"), -1, 0
    history = []
    rng = np.random.RandomState(seed)

    # train loop
    for epoch in range(p["max_epochs"]):
        model.train()
        order = rng.permutation(len(train_graphs))
        epoch_losses = []
        for start in range(0, len(order), graphs_per_step):
            batch_idx = order[start:start + graphs_per_step]
            batch = Batch.from_data_list([train_graphs[i] for i in batch_idx]).to(device)
            opt.zero_grad(set_to_none=True)
            out = model(batch.x, *_forward_args(batch, backend))
            pathway_weight = cfg["patch_head"]["pathway_loss_weight"]
            losses = graph_loss(out, batch.y_gene, batch.y_pathway, pathway_weight)
            losses["loss"].backward()
            opt.step()
            epoch_losses.append(float(losses["loss"].item()))

        val_loss = _eval_loss(model, val_graphs, cfg, device, backend)
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "val_loss": val_loss}
        )

        # checkpoint on improvement, else count toward early stop
        if val_loss < best_val:
            best_val, best_epoch, patience_ctr = val_loss, epoch, 0
            torch.save(
                {
                    "graph_correction_state_dict": model.graph_correction.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "seed": seed,
                    "backend": backend,
                },
                out_path,
            )
        else:
            patience_ctr += 1
        if patience_ctr >= p["early_stop_patience"]:
            break

    # save history
    save_json(
        {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "backend": backend,
        },
        str(out_path).replace(".pt", "_history.json"),
    )
    return {
        "checkpoint": str(out_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "backend": backend,
    }


@torch.no_grad()
def _eval_loss(
    model, graphs: list[Data], cfg: dict, device: str, backend: str = "pna_union"
) -> float:
    model.eval()
    losses = []
    for g in graphs:
        batch = Batch.from_data_list([g]).to(device)
        out = model(batch.x, *_forward_args(batch, backend))
        pathway_weight = cfg["patch_head"]["pathway_loss_weight"]
        l = graph_loss(out, batch.y_gene, batch.y_pathway, pathway_weight)
        losses.append(float(l["loss"].item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict_graph_model(
    model, graphs: list[Data], device: str, backend: str = "pna_union"
) -> dict:
    """Returns gene/pathway final+patch-only predictions concatenated in `graphs` iteration order; callers must reorder their own id/target arrays to match, or predictions silently misalign with targets."""
    model.eval()
    out = {
        "patient_id": [],
        "gene_pred": [],
        "pathway_pred": [],
        "gene_patch_only": [],
        "pathway_patch_only": [],
    }
    scale_weights = []
    for g in graphs:
        batch = Batch.from_data_list([g]).to(device)
        pred = model(batch.x, *_forward_args(batch, backend))
        n = g.num_nodes
        out["patient_id"] += [g.patient_id] * n
        out["gene_pred"].append(pred["gene"].cpu().numpy())
        out["pathway_pred"].append(pred["pathway"].cpu().numpy())
        out["gene_patch_only"].append(pred["gene_patch_only"].cpu().numpy())
        out["pathway_patch_only"].append(pred["pathway_patch_only"].cpu().numpy())
        if "scale_weights" in pred:
            scale_weights.append(pred["scale_weights"].cpu().numpy())

    # stack per-graph blocks
    for k in ("gene_pred", "pathway_pred", "gene_patch_only", "pathway_patch_only"):
        out[k] = np.concatenate(out[k], axis=0)
    out["patient_id"] = np.array(out["patient_id"])
    if scale_weights:
        # (N, 3) = [alpha_L, alpha_M, alpha_G] per node
        out["scale_weights"] = np.concatenate(scale_weights, axis=0)
    return out
