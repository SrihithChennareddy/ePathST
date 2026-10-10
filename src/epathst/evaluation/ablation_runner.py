"""Trains and scores one ablation model on a cross-validation fold of the HEST cohort.
Shared by the patient-level CV, the official-protocol ablation grid and the Leiden analyses."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from epathst.ablations import (
    build_graph_for_ablation,
    edge_features_for_ablation,
    matched_mlp_hidden_dim,
    scale_aware_edge_data,
)
from epathst.data.expression import ensure_gene_symbol_index, normalize_log1p_slide
from epathst.data.pathways import compute_progeny_scores, fetch_and_cache_progeny_net
from epathst.evaluation.metrics import evaluate_predictions
from epathst.graph.training import (
    _build_correction_model,
    make_pyg_data,
    predict_graph_model,
    train_graph_model,
)
from epathst.models.graph_model import ResidualGraphPredictor
from epathst.models.patch_head import MolecularBottleneckHead, build_patch_head, patch_head_loss


def load_fold_expression(hest_samples, spot_manifest, hest_root, patients, cfg):
    """Loads expression and PROGENy scores for a fold's patients, indexed by spot_uid.

    spot_id alone collides across samples (two real breast samples share 144/176 ids), which
    would give a duplicate-keyed index and ambiguous cross-patient .loc[] lookups.
    """
    frames, progeny_frames, genes_by_sample = [], [], {}
    net = fetch_and_cache_progeny_net(
        Path(cfg["paths"]["work_dir"]) / "progeny_net.csv",
        cfg["molecular_targets"]["pathway_top_genes"],
    )
    for sample_id in hest_samples[hest_samples["patient_id"].isin(patients)]["sample_id"]:
        row = hest_samples[hest_samples["sample_id"] == sample_id].iloc[0]
        adata = ad.read_h5ad(row["expression_path"])
        adata = ensure_gene_symbol_index(adata)
        normalize_log1p_slide(adata, cfg["molecular_targets"]["normalize_total_target_sum"])
        uid_index = [f"{sample_id}::{sid}" for sid in adata.obs_names.astype(str)]
        expr = pd.DataFrame(
            adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
            index=uid_index,
            columns=adata.var_names.astype(str),
        )
        genes_by_sample[sample_id] = set(expr.columns)
        frames.append(expr)
        progeny_frames.append(compute_progeny_scores(expr, net))
    expr_all = pd.concat(frames)
    progeny_all = pd.concat(progeny_frames)
    assert (
        expr_all.index.is_unique
    ), "spot_uid collision within one fold's pooled expression -- should be impossible"
    return expr_all, progeny_all, genes_by_sample, net


def build_graph_data(
    feat_dict,
    coords,
    spot_ids,
    gene_target,
    pathway_target,
    patient_id,
    graph_source,
    backend,
    cfg,
    seed,
):
    edges = build_graph_for_ablation(coords, graph_source, cfg, seed)
    if backend == "scale_aware":
        edge_index, edge_attr, edge_scale = scale_aware_edge_data(edges, cfg)
        return make_pyg_data(
            feat_dict,
            edge_index,
            edge_attr,
            gene_target,
            pathway_target,
            patient_id,
            edge_scale=edge_scale,
        )
    edge_attr = edge_features_for_ablation(edges, graph_source, cfg)
    return make_pyg_data(
        feat_dict, edges["edge_index"], edge_attr, gene_target, pathway_target, patient_id
    )


def train_patch_head(
    x_train, x_val, y_gene_train, y_path_train, y_gene_val, y_path_val, cfg, device
):
    """Trains one MolecularBottleneckHead with the locked recipe. Shared so A1/A2 heads differ
    only in which frozen encoder's features they see, the difference Gate 1 isolates.
    """
    patch_head = build_patch_head(
        cfg, x_train.shape[1], y_gene_train.shape[1], y_path_train.shape[1]
    ).to(device)
    opt = torch.optim.AdamW(
        patch_head.parameters(),
        lr=cfg["patch_head"]["lr"],
        weight_decay=cfg["patch_head"]["weight_decay"],
    )

    best_val, patience_ctr, best_state = float("inf"), 0, None
    for epoch in range(cfg["patch_head"]["max_epochs"]):
        patch_head.train()
        x_train_batch = torch.tensor(x_train, dtype=torch.float32, device=device)
        gene_train_batch = torch.tensor(y_gene_train, dtype=torch.float32, device=device)
        pathway_train_batch = torch.tensor(y_path_train, dtype=torch.float32, device=device)
        opt.zero_grad()
        out = patch_head(x_train_batch)
        loss = patch_head_loss(
            out, gene_train_batch, pathway_train_batch, cfg["patch_head"]["pathway_loss_weight"]
        )
        loss["loss"].backward()
        opt.step()

        patch_head.eval()
        with torch.no_grad():
            x_val_batch = torch.tensor(x_val, dtype=torch.float32, device=device)
            gene_val_batch = torch.tensor(y_gene_val, dtype=torch.float32, device=device)
            pathway_val_batch = torch.tensor(y_path_val, dtype=torch.float32, device=device)
            pathway_weight = cfg["patch_head"]["pathway_loss_weight"]
            val_loss = patch_head_loss(
                patch_head(x_val_batch), gene_val_batch, pathway_val_batch, pathway_weight
            )["loss"].item()
        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in patch_head.state_dict().items()}
        else:
            patience_ctr += 1
        if patience_ctr >= cfg["patch_head"]["early_stop_patience"]:
            break
    patch_head.load_state_dict(best_state)
    return patch_head


def run_ablation(
    spec,
    x_train,
    x_val,
    x_test,
    ids_train,
    ids_val,
    ids_test,
    y_gene_train,
    y_path_train,
    y_gene_val,
    y_path_val,
    y_gene_test,
    y_path_test,
    patch_head,
    spot_manifest,
    cfg,
    seed,
    device,
    n_hvg50,
    pathway_names,
):
    # Dispatches on (graph_source, backend): A0 mean baseline, A1/A2 frozen patch_head, A3 matched
    # MLP (no graph), A4-A7/A9 pna_union/gatv2 graph correction, A8 scale_aware graph correction.
    # Returns (preds, metrics, eval_ids_test) in the order actually scored. Joins on spot_uid,
    # never bare spot_id (collides across samples, see load_fold_expression).
    spot_by_uid = spot_manifest.set_index("spot_uid")

    def coords_for(spot_uids):
        return spot_by_uid.loc[spot_uids, ["x_um", "y_um"]].values

    def patient_for(spot_uids):
        return spot_by_uid.loc[spot_uids, "patient_id"].values

    if spec.name == "A0_mean_expression":
        gene_mean = y_gene_train.mean(axis=0, keepdims=True)
        path_mean = y_path_train.mean(axis=0, keepdims=True)
        gene_pred = np.repeat(gene_mean, len(ids_test), axis=0)
        path_pred = np.repeat(path_mean, len(ids_test), axis=0)
    elif spec.name == "A3_matched_mlp":
        # parameter-matched MLP, same features, no neighbors: tests whether capacity alone explains
        # a gain
        pna_param_count = sum(p.numel() for p in patch_head.parameters())
        h = matched_mlp_hidden_dim(
            pna_param_count, x_train.shape[1], y_gene_train.shape[1] + y_path_train.shape[1]
        )
        mlp = MolecularBottleneckHead(
            x_train.shape[1], [x_train.shape[1], h, h], y_gene_train.shape[1], y_path_train.shape[1]
        ).to(device)
        opt = torch.optim.AdamW(
            mlp.parameters(),
            lr=cfg["patch_head"]["lr"],
            weight_decay=cfg["patch_head"]["weight_decay"],
        )
        best_val, patience_ctr, best_state = float("inf"), 0, None
        for epoch in range(cfg["patch_head"]["max_epochs"]):
            mlp.train()
            x_train_batch = torch.tensor(x_train, dtype=torch.float32, device=device)
            gene_train_batch = torch.tensor(y_gene_train, dtype=torch.float32, device=device)
            pathway_train_batch = torch.tensor(y_path_train, dtype=torch.float32, device=device)
            opt.zero_grad()
            loss = patch_head_loss(
                mlp(x_train_batch),
                gene_train_batch,
                pathway_train_batch,
                cfg["patch_head"]["pathway_loss_weight"],
            )
            loss["loss"].backward()
            opt.step()
            mlp.eval()
            with torch.no_grad():
                x_val_batch = torch.tensor(x_val, dtype=torch.float32, device=device)
                gene_val_batch = torch.tensor(y_gene_val, dtype=torch.float32, device=device)
                pathway_val_batch = torch.tensor(y_path_val, dtype=torch.float32, device=device)
                pathway_weight = cfg["patch_head"]["pathway_loss_weight"]
                val_loss = patch_head_loss(
                    mlp(x_val_batch), gene_val_batch, pathway_val_batch, pathway_weight
                )["loss"].item()
            if val_loss < best_val:
                best_val = val_loss
                patience_ctr = 0
                best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
            else:
                patience_ctr += 1
            if patience_ctr >= cfg["patch_head"]["early_stop_patience"]:
                break
        mlp.load_state_dict(best_state)
        mlp.eval()
        with torch.no_grad():
            out = mlp(torch.tensor(x_test, dtype=torch.float32, device=device))
        gene_pred, path_pred = out["gene"].cpu().numpy(), out["pathway"].cpu().numpy()
    elif spec.graph_source == "none":
        # A1/A2: frozen patch head forward pass; this is y_patch,i, the A2 anchor
        patch_head.eval()
        with torch.no_grad():
            out = patch_head(torch.tensor(x_test, dtype=torch.float32, device=device))
        gene_pred, path_pred = out["gene"].cpu().numpy(), out["pathway"].cpu().numpy()
    else:
        backend = spec.backend
        graphs_train = [
            build_graph_data(
                x_train[patient_for(ids_train) == p],
                coords_for(ids_train[patient_for(ids_train) == p]),
                ids_train[patient_for(ids_train) == p],
                y_gene_train[patient_for(ids_train) == p],
                y_path_train[patient_for(ids_train) == p],
                p,
                spec.graph_source,
                backend,
                cfg,
                seed,
            )
            for p in np.unique(patient_for(ids_train))
        ]
        graphs_val = [
            build_graph_data(
                x_val[patient_for(ids_val) == p],
                coords_for(ids_val[patient_for(ids_val) == p]),
                ids_val[patient_for(ids_val) == p],
                y_gene_val[patient_for(ids_val) == p],
                y_path_val[patient_for(ids_val) == p],
                p,
                spec.graph_source,
                backend,
                cfg,
                seed,
            )
            for p in np.unique(patient_for(ids_val))
        ]
        graphs_test = [
            build_graph_data(
                x_test[patient_for(ids_test) == p],
                coords_for(ids_test[patient_for(ids_test) == p]),
                ids_test[patient_for(ids_test) == p],
                y_gene_test[patient_for(ids_test) == p],
                y_path_test[patient_for(ids_test) == p],
                p,
                spec.graph_source,
                backend,
                cfg,
                seed,
            )
            for p in np.unique(patient_for(ids_test))
        ]

        ckpt_path = (
            Path(cfg["paths"]["work_dir"])
            / "student_checkpoints"
            / f"graph_{spec.name}_seed{seed}.pt"
        )
        train_graph_model(
            patch_head, graphs_train, graphs_val, cfg, ckpt_path, seed, device, backend=backend
        )
        edge_dim = graphs_train[0].edge_attr.shape[1]
        graph_corr = _build_correction_model(
            backend,
            x_train.shape[1],
            cfg,
            graphs_train,
            edge_dim,
            y_gene_train.shape[1],
            y_path_train.shape[1],
        )
        state = torch.load(ckpt_path, map_location=device)
        graph_corr.load_state_dict(state["graph_correction_state_dict"])
        model = ResidualGraphPredictor(patch_head, graph_corr).to(device)
        preds = predict_graph_model(model, graphs_test, device, backend=backend)
        gene_pred, path_pred = preds["gene_pred"], preds["pathway_pred"]
        # predict_graph_model returns per-patient-grouped order, so ids_test AND
        # y_gene_test/y_path_test must be reordered together. Reordering only ids_test silently
        # breaks every graph ablation (debug_idc_pipeline.py: r~1.0 -> r~0.08). Most important
        # correctness invariant here; do not simplify it away.
        order = np.concatenate(
            [np.where(patient_for(ids_test) == p)[0] for p in np.unique(patient_for(ids_test))]
        )
        ids_test = ids_test[order]
        y_gene_test = y_gene_test[order]
        y_path_test = y_path_test[order]

    # HVG50 is the first n_hvg50 columns of HVG1000 (nested, see select_hvg_for_fold), so it is a
    # column slice, not a separate model; HVG50/HVG1000/pathway are scored separately.
    pids, cancer_placeholder = patient_for(ids_test), np.array(["cancer"] * len(ids_test))
    m_hvg1000 = evaluate_predictions(
        y_gene_test,
        gene_pred,
        pids,
        cancer_placeholder,
        [str(i) for i in range(y_gene_test.shape[1])],
        cfg,
    ).get("overall", {})
    m_hvg50 = evaluate_predictions(
        y_gene_test[:, :n_hvg50],
        gene_pred[:, :n_hvg50],
        pids,
        cancer_placeholder,
        [str(i) for i in range(n_hvg50)],
        cfg,
    ).get("overall", {})
    m_pathway = evaluate_predictions(
        y_path_test, path_pred, pids, cancer_placeholder, pathway_names, cfg
    ).get("overall", {})
    metrics = {f"hvg1000_{k}": v for k, v in m_hvg1000.items()}
    metrics.update({f"hvg50_{k}": v for k, v in m_hvg50.items()})
    metrics.update({f"pathway_{k}": v for k, v in m_pathway.items()})
    # The splits completeness gate and Gate 1-3 tests key off this exact unprefixed name.
    metrics["fisher_z_mean_patient_R"] = m_hvg1000.get("fisher_z_mean_patient_R")
    return {"gene": gene_pred, "pathway": path_pred}, metrics, ids_test
