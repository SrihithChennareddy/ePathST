"""Model-efficiency accounting; reports exact programmatically-derived counts -- never manual estimates --
and always separates teacher-label-generation cost from student inference cost."""
from __future__ import annotations

import time

import numpy as np
import torch

from pathology_st.utils import count_parameters


def count_model_params(model) -> dict:
    return count_parameters(model)


def compression_ratio(teacher_params: dict, student_params: dict) -> dict:
    return {
        "teacher_total_parameters": teacher_params["total_parameters"],
        "student_total_parameters": student_params["total_parameters"],
        "compression_ratio_x": (
            teacher_params["total_parameters"] / max(1, student_params["total_parameters"])
        ),
    }


def measure_flops_macs(model, input_shape: tuple, device: str = "cuda") -> dict:
    """FLOPs/MACs per patch via fvcore if available; falls back to reporting None with an explanatory note if it isn't installed, rather than fabricating a number."""
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        x = torch.randn(1, *input_shape, device=device)
        flops = FlopCountAnalysis(model, x)
        return {
            "macs_per_patch": float(flops.total()),
            "flops_per_patch": float(flops.total()) * 2,
            "method": "fvcore",
        }
    except ImportError:
        return {
            "macs_per_patch": None,
            "flops_per_patch": None,
            "method": "fvcore not installed -- install to measure; not estimated",
        }


@torch.no_grad()
def measure_inference_throughput(
    model,
    patches_uint8: np.ndarray,
    device: str,
    batch_size: int = 64,
    n_warmup: int = 2,
) -> dict:
    """CNN feature-extraction time and patches/sec."""
    model.eval()
    n = len(patches_uint8)
    # warmup batches, excluded from the timed measurement
    for i in range(min(n_warmup, n // batch_size or 1)):
        batch = patches_uint8[i * batch_size:(i + 1) * batch_size]
        if len(batch) == 0:
            break
        x = _prep_batch(batch, device)
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # timed inference pass
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch = patches_uint8[i:i + batch_size]
        x = _prep_batch(batch, device)
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {
        "n_patches": n,
        "elapsed_seconds": elapsed,
        "patches_per_second": n / max(elapsed, 1e-9),
    }


def _prep_batch(batch_uint8: np.ndarray, device: str):
    from pathology_st.distill_train import IMAGENET_MEAN, IMAGENET_STD
    x = batch_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (0, 3, 1, 2)).astype(np.float32)
    return torch.from_numpy(x).to(device)


def measure_graph_inference_time(model, graphs: list, device: str) -> dict:
    from torch_geometric.data import Batch
    model.eval()
    t0 = time.time()
    n_nodes, n_edges = 0, 0
    with torch.no_grad():
        for g in graphs:
            batch = Batch.from_data_list([g]).to(device)
            model(batch.x, batch.edge_index, batch.edge_attr)
            n_nodes += g.num_nodes
            n_edges += g.edge_index.shape[1]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {"n_graphs": len(graphs), "total_nodes": n_nodes, "total_edges": n_edges,
            "elapsed_seconds": elapsed, "seconds_per_1000_nodes": elapsed / max(1, n_nodes) * 1000}


def stored_embedding_size_bytes(n_items: int, dim: int, dtype_bytes: int = 2) -> int:
    return n_items * dim * dtype_bytes
