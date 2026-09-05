"""Extracts frozen encoder features on HEST patches and caches them to
disk, with spot_id <-> patch <-> coordinate <-> embedding <-> expression
integrity assertions."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch


@torch.no_grad()
def extract_student_features(encoder, patches_uint8: np.ndarray, device: str, batch_size: int = 256,
                               downstream_proj=None) -> np.ndarray:
    """Runs the frozen encoder over patches in batches; image only, no
    expression data touches this path. Returns R^320 features (R^256 if
    `downstream_proj` is given)."""
    from pathology_st.distill_train import IMAGENET_MEAN, IMAGENET_STD

    encoder.eval()
    outs = []
    for i in range(0, len(patches_uint8), batch_size):
        batch = patches_uint8[i:i + batch_size].astype(np.float32) / 255.0
        batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
        batch = np.transpose(batch, (0, 3, 1, 2)).astype(np.float32)
        x = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            feat = encoder(x)
            if downstream_proj is not None:
                feat = downstream_proj(feat)
        outs.append(feat.cpu().numpy())
    return np.concatenate(outs, axis=0)


def cache_features(out_path: str | Path, spot_ids: list[str], features: np.ndarray, coords_um: np.ndarray,
                    encoder_name: str | None = None, checkpoint_source: str | None = None,
                    sample_id: str | None = None) -> None:
    """Stores `encoder_name`/`checkpoint_source`/`sample_id` as h5 attrs so
    `load_cached_features` can assert a cache belongs to the encoder the
    caller expects -- without this, A1 (ImageNet) and A2 (distilled) can
    silently load each other's features (see src/pathology_st/ablations.py)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("spot_id", data=np.array(spot_ids, dtype="S"))
        f.create_dataset("embedding", data=features.astype(np.float32))
        f.create_dataset("coords_um", data=coords_um.astype(np.float32))
        f.attrs["encoder_name"] = encoder_name or "unknown"
        f.attrs["checkpoint_source"] = checkpoint_source or "unknown"
        f.attrs["feature_dim"] = int(features.shape[1])
        f.attrs["sample_id"] = sample_id or "unknown"


def load_cached_features(path: str | Path, expected_encoder_name: str | None = None) -> dict:
    with h5py.File(path, "r") as f:
        out = {
            "spot_id": f["spot_id"][:].astype(str),
            "embedding": f["embedding"][:],
            "coords_um": f["coords_um"][:],
            "encoder_name": f.attrs.get("encoder_name", "unknown"),
            "checkpoint_source": f.attrs.get("checkpoint_source", "unknown"),
            "feature_dim": f.attrs.get("feature_dim", f["embedding"].shape[1]),
            "sample_id": f.attrs.get("sample_id", "unknown"),
        }
    if expected_encoder_name is not None:
        assert out["encoder_name"] == expected_encoder_name, (
            f"cache routing error: {path} was cached by encoder "
            f"'{out['encoder_name']}', but the caller required '{expected_encoder_name}' -- "
            f"this is the A1/A2 feature-mixup bug class, see src/pathology_st/ablations.py")
    return out


def assert_full_alignment(spot_ids: list[str], embedding: np.ndarray, coords_um: np.ndarray,
                           expr_df, expr_spot_col: str = None) -> dict:
    """Checks that spot_id <-> patch <-> coordinate <-> embedding <->
    expression stay in identical order. `expr_df` is indexed (or has a
    column) by spot_id."""
    expr_ids = expr_df.index.astype(str).tolist() if expr_spot_col is None else expr_df[expr_spot_col].astype(str).tolist()
    checks = {
        "n_spot_ids": len(spot_ids),
        "n_embeddings": embedding.shape[0],
        "n_coords": coords_um.shape[0],
        "n_expr_rows": len(expr_ids),
        "spot_embedding_count_match": len(spot_ids) == embedding.shape[0],
        "spot_coords_count_match": len(spot_ids) == coords_um.shape[0],
        "spot_expr_order_match": list(spot_ids) == expr_ids,
        "no_duplicate_spot_ids": len(spot_ids) == len(set(spot_ids)),
    }
    checks["all_ok"] = all(v for k, v in checks.items() if isinstance(v, bool))
    assert checks["all_ok"], f"feature-cache integrity check failed: {checks}"
    return checks
