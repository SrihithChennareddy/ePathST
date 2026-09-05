"""Frozen GenBio-PathFM teacher (https://huggingface.co/genbio-ai/genbio-pathfm),
run once and cached. Inference only -- no gradients, 1.1B parameters never
updated. Every CPTAC patch is embedded exactly once; embeddings are
cached to disk (FP16) and reused across every distillation epoch, since
re-running the teacher per epoch would be prohibitively expensive."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


class GenBioPathFMTeacher:
    """Follows the model card's documented usage, which differs from the
    generic HF AutoModel/AutoImageProcessor pattern:
      - No `preprocessor_config.json` in the repo, so
        `AutoImageProcessor.from_pretrained` fails -- preprocessing is a
        plain Resize(224,224) -> ToTensor -> Normalize with GenBio-PathFM-
        specific stats (not ImageNet's).
      - `model(x)` returns the [B, 4608] CLS-feature tensor directly, not
        a HF-style ModelOutput with `.last_hidden_state`."""

    GENBIO_MEAN = (0.697, 0.575, 0.728)
    GENBIO_STD = (0.188, 0.240, 0.187)

    def __init__(self, cfg: dict, device: str = "cuda"):
        import torch
        from transformers import AutoModel

        self.device = device
        repo = cfg["teacher"]["hf_repo"]
        dtype = torch.bfloat16 if cfg["teacher"]["dtype"] == "bf16" else torch.float16
        self.dtype = dtype
        self.model = AutoModel.from_pretrained(repo, torch_dtype=dtype, trust_remote_code=True)
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embedding_dim = cfg["teacher"]["embedding_dim"]
        self._n_params = sum(p.numel() for p in self.model.parameters())
        self._mean = np.array(self.GENBIO_MEAN, dtype=np.float32)
        self._std = np.array(self.GENBIO_STD, dtype=np.float32)

    @property
    def n_parameters(self) -> int:
        return self._n_params

    def _preprocess(self, patches_uint8: np.ndarray):
        """patches_uint8: (B, 224, 224, 3) uint8 RGB. Patches are already
        224x224, but resize is applied anyway as a safety net for
        non-224 input."""
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(patches_uint8).to(self.device).float() / 255.0
        # (B, H, W, 3) -> (B, 3, H, W); must be .contiguous() -- the model's forward() does
        # pixel_values.view(...) internally, which fails on permute()'s non-contiguous strides.
        x = x.permute(0, 3, 1, 2).contiguous()
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.from_numpy(self._mean).to(self.device).view(1, 3, 1, 1)
        std = torch.from_numpy(self._std).to(self.device).view(1, 3, 1, 1)
        return ((x - mean) / std).contiguous().to(self.dtype)

    def embed_batch(self, patches_uint8: np.ndarray) -> np.ndarray:
        """patches_uint8: (B, H, W, 3) uint8 RGB. Returns (B, 4608) float32
        L2-normalized CLS embeddings."""
        import torch
        x = self._preprocess(patches_uint8)
        with torch.no_grad():
            cls = self.model(x)  # [B, 4608] tensor, returned directly (see class docstring)
        cls = cls.float().cpu().numpy()
        assert cls.shape[1] == self.embedding_dim, (
            f"expected teacher embedding dim {self.embedding_dim}, got {cls.shape[1]}"
        )
        norm = np.linalg.norm(cls, axis=1, keepdims=True)
        norm = np.clip(norm, 1e-8, None)
        return (cls / norm).astype(np.float32)


def cache_teacher_embeddings(out_path: str | Path, patch_ids: list[str], embeddings: np.ndarray) -> None:
    """Appends patch_id <-> teacher_embedding to disk (FP16 -- embeddings
    are unit-norm, so FP16 precision is adequate).

    Uses h5py's variable-length string dtype for `patch_id`, not a
    fixed-width numpy `dtype="S"` array: a fixed-width dtype freezes its
    per-element byte-width at the first call's longest string and
    silently truncates any longer id appended later. This previously
    corrupted CPTAC patch_ids (length varies with pixel-coordinate digit
    count) into collisions between unrelated patches, and broke
    resumability (a fresh id never matches its truncated form on disk)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "a") as f:
        if "teacher_embedding" not in f:
            f.create_dataset("teacher_embedding", data=embeddings.astype(np.float16), maxshape=(None, embeddings.shape[1]))
            f.create_dataset("patch_id", data=np.array(patch_ids, dtype=object), dtype=str_dtype, maxshape=(None,))
        else:
            n0 = f["teacher_embedding"].shape[0]
            n1 = n0 + len(patch_ids)
            f["teacher_embedding"].resize(n1, axis=0)
            f["teacher_embedding"][n0:n1] = embeddings.astype(np.float16)
            f["patch_id"].resize(n1, axis=0)
            f["patch_id"][n0:n1] = np.array(patch_ids, dtype=object)


def verify_teacher_cache(path: str | Path, expected_dim: int) -> dict:
    """Integrity checks: no NaNs, expected dimension, near-unit norms, and
    no duplicate patch_ids."""
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
    report["passed"] = all([report["dim_ok"], report["no_nans"], report["norms_near_unit"],
                             report["no_duplicate_patch_ids"]])
    return report
