"""Frozen H-optimus-0 teacher (https://huggingface.co/bioptimus/H-optimus-0) and embedding cache.

Inference only, no gradients: each CPTAC patch is embedded once and cached as FP16, since
re-running the teacher every epoch would be prohibitively expensive.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


class HOptimusTeacher:
    """H-optimus-0 loaded through timm's hf-hub loader (the model card's recommended path).

    Not transformers.AutoModel: `timm.create_model("hf-hub:bioptimus/H-optimus-0", ...)` returns
    the [B, 1536] CLS tensor directly, not a ModelOutput with `.last_hidden_state`.
    """

    HOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
    HOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

    def __init__(self, cfg: dict, device: str = "cuda"):
        import timm
        import torch

        self.device = device
        repo = cfg["teacher"]["hf_repo"]
        dtype = torch.bfloat16 if cfg["teacher"]["dtype"] == "bf16" else torch.float16
        self.dtype = dtype
        self.model = timm.create_model(
            f"hf-hub:{repo}", pretrained=True, init_values=1e-5, dynamic_img_size=False
        )
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embedding_dim = cfg["teacher"]["embedding_dim"]
        self._n_params = sum(p.numel() for p in self.model.parameters())
        self._mean = np.array(self.HOPTIMUS_MEAN, dtype=np.float32)
        self._std = np.array(self.HOPTIMUS_STD, dtype=np.float32)

    @property
    def n_parameters(self) -> int:
        return self._n_params

    def _preprocess(self, patches_uint8: np.ndarray):
        """(B, 224, 224, 3) uint8 RGB -> normalized (B, 3, 224, 224) fp32 on the model device.

        Patches that are not exactly 224x224 are resized as a safety net. Inputs and weights stay
        fp32: the model card recommends mixed precision via autocast at forward time, not casting
        to fp16 up front.
        """
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(patches_uint8).to(self.device).float() / 255.0
        # (B,H,W,3) -> (B,3,H,W); contiguous() is required because forward()'s view() fails on
        # permute()'s non-contiguous strides.
        x = x.permute(0, 3, 1, 2).contiguous()
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.from_numpy(self._mean).to(self.device).view(1, 3, 1, 1)
        std = torch.from_numpy(self._std).to(self.device).view(1, 3, 1, 1)
        return ((x - mean) / std).contiguous()

    def embed_batch(self, patches_uint8: np.ndarray) -> np.ndarray:
        """(B, H, W, 3) uint8 RGB -> (B, 1536) float32 L2-normalized CLS embeddings."""
        import torch

        x = self._preprocess(patches_uint8)
        with (
            torch.autocast(device_type=self.device.split(":")[0], dtype=self.dtype),
            torch.no_grad(),
        ):
            cls = self.model(x)  # (B, 1536) tensor directly, see class docstring
        cls = cls.float().cpu().numpy()
        assert (
            cls.shape[1] == self.embedding_dim
        ), f"expected teacher embedding dim {self.embedding_dim}, got {cls.shape[1]}"
        norm = np.linalg.norm(cls, axis=1, keepdims=True)
        norm = np.clip(norm, 1e-8, None)
        return (cls / norm).astype(np.float32)


def cache_teacher_embeddings(
    out_path: str | Path, patch_ids: list[str], embeddings: np.ndarray
) -> None:
    """Append patch_id/teacher_embedding rows to an HDF5 file.

    Embeddings are stored as FP16, which is adequate for unit-norm vectors. `patch_id` uses h5py's
    variable-length string dtype, not a fixed-width numpy `dtype="S"`: a fixed width freezes at
    the first call's longest string, and it previously truncated/collided later CPTAC ids (length
    varies with pixel-coordinate digits) and broke resumability.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "a") as f:
        if "teacher_embedding" not in f:
            f.create_dataset(
                "teacher_embedding",
                data=embeddings.astype(np.float16),
                maxshape=(None, embeddings.shape[1]),
            )
            f.create_dataset(
                "patch_id",
                data=np.array(patch_ids, dtype=object),
                dtype=str_dtype,
                maxshape=(None,),
            )
        else:
            n0 = f["teacher_embedding"].shape[0]
            n1 = n0 + len(patch_ids)
            f["teacher_embedding"].resize(n1, axis=0)
            f["teacher_embedding"][n0:n1] = embeddings.astype(np.float16)
            f["patch_id"].resize(n1, axis=0)
            f["patch_id"][n0:n1] = np.array(patch_ids, dtype=object)


def verify_teacher_cache(path: str | Path, expected_dim: int) -> dict:
    """Check a teacher cache: no NaNs, expected dimension, near-unit norms, unique patch_ids."""
    with h5py.File(path, "r") as f:
        emb = f["teacher_embedding"][:].astype(np.float32)
        pids = f["patch_id"][:].astype(str)

    n_nan = int(np.isnan(emb).sum())
    norms = np.linalg.norm(emb, axis=1)
    report = {
        "n_patches": len(pids),
        "n_unique_patch_ids": int(len(set(pids))),
        "embedding_dim": emb.shape[1],
        "expected_dim": expected_dim,
        "dim_ok": emb.shape[1] == expected_dim,
        "n_nan": n_nan,
        "no_nans": n_nan == 0,
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norms_near_unit": bool(abs(norms.mean() - 1.0) < 0.05),
        "no_duplicate_patch_ids": len(pids) == len(set(pids)),
    }
    report["passed"] = all(
        [
            report["dim_ok"],
            report["no_nans"],
            report["norms_near_unit"],
            report["no_duplicate_patch_ids"],
        ]
    )
    return report
