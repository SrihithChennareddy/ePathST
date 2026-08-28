"""Training loop for the residual graph model (Steps 16-17). One graph per
patient/slide; batches of 1-4 graphs per optimizer step (gradient
accumulation for larger effective batches), matching the linear-in-N
compute scaling argument for PNA (fixed degree k => |E| = O(N))."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from sts.graph_model import GraphCorrectionModel, ResidualGraphPredictor, graph_loss
from sts.pna_block import compute_degree_histogram
from sts.utils import save_json, seed_everything


def make_pyg_data(features: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray,
                   gene_target: np.ndarray, pathway_target: np.ndarray, patient_id: str) -> Data:
    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y_gene=torch.tensor(gene_target, dtype=torch.float32),
        y_pathway=torch.tensor(pathway_target, dtype=torch.float32),
        patient_id=patient_id,
    )


def train_graph_model(patch_head, train_graphs: list[Data], val_graphs: list[Data], cfg: dict, out_path: Path,
                       seed: int, device: str = "cuda", graphs_per_step: int = 4) -> dict:
    seed_everything(seed)
    p = cfg["pna"]
    edge_dim = train_graphs[0].edge_attr.shape[1] if train_graphs[0].edge_attr.numel() else p.get("edge_dim_fallback", 19)
    deg = compute_degree_histogram([g.edge_index for g in train_graphs], [g.num_nodes for g in train_graphs])

    in_dim = train_graphs[0].x.shape[1]
    gene_dim = train_graphs[0].y_gene.shape[1]
    pathway_dim = train_graphs[0].y_pathway.shape[1]
    graph_corr = GraphCorrectionModel(in_dim, cfg, deg, edge_dim, gene_dim, pathway_dim)
    model = ResidualGraphPredictor(patch_head, graph_corr).to(device)

    opt = torch.optim.AdamW(model.graph_correction.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])

    best_val, best_epoch, patience_ctr = float("inf"), -1, 0
    history = []
    rng = np.random.RandomState(seed)

    for epoch in range(p["max_epochs"]):
        model.train()
        order = rng.permutation(len(train_graphs))
        epoch_losses = []
        for start in range(0, len(order), graphs_per_step):
            batch_idx = order[start:start + graphs_per_step]
            batch = Batch.from_data_list([train_graphs[i] for i in batch_idx]).to(device)
            opt.zero_grad(set_to_none=True)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            losses = graph_loss(out, batch.y_gene, batch.y_pathway, cfg["patch_head"]["pathway_loss_weight"])
            losses["loss"].backward()
            opt.step()
            epoch_losses.append(float(losses["loss"].item()))

        val_loss = _eval_loss(model, val_graphs, cfg, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "val_loss": val_loss})

        if val_loss < best_val:
            best_val, best_epoch, patience_ctr = val_loss, epoch, 0
            torch.save({"graph_correction_state_dict": model.graph_correction.state_dict(),
                        "epoch": epoch, "val_loss": val_loss, "seed": seed}, out_path)
        else:
            patience_ctr += 1
        if patience_ctr >= p["early_stop_patience"]:
            break

    save_json({"history": history, "best_epoch": best_epoch, "best_val_loss": best_val},
              str(out_path).replace(".pt", "_history.json"))
    return {"checkpoint": str(out_path), "best_epoch": best_epoch, "best_val_loss": best_val}


@torch.no_grad()
def _eval_loss(model, graphs: list[Data], cfg: dict, device: str) -> float:
    model.eval()
    losses = []
    for g in graphs:
        batch = Batch.from_data_list([g]).to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        l = graph_loss(out, batch.y_gene, batch.y_pathway, cfg["patch_head"]["pathway_loss_weight"])
        losses.append(float(l["loss"].item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict_graph_model(model, graphs: list[Data], device: str) -> dict:
    """Returns per-patient arrays of gene/pathway final + patch-only
    predictions, for downstream metrics + the primary hypothesis test."""
    model.eval()
    out = {"patient_id": [], "gene_pred": [], "pathway_pred": [], "gene_patch_only": [], "pathway_patch_only": []}
    for g in graphs:
        batch = Batch.from_data_list([g]).to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        n = g.num_nodes
        out["patient_id"] += [g.patient_id] * n
        out["gene_pred"].append(pred["gene"].cpu().numpy())
        out["pathway_pred"].append(pred["pathway"].cpu().numpy())
        out["gene_patch_only"].append(pred["gene_patch_only"].cpu().numpy())
        out["pathway_patch_only"].append(pred["pathway_patch_only"].cpu().numpy())
    for k in ("gene_pred", "pathway_pred", "gene_patch_only", "pathway_patch_only"):
        out[k] = np.concatenate(out[k], axis=0)
    out["patient_id"] = np.array(out["patient_id"])
    return out
