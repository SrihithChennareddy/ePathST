"""Step 4 (Section 7): frozen GenBio-PathFM teacher, run once and cached.

GenBio-PathFM (https://huggingface.co/genbio-ai/genbio-pathfm) is loaded
strictly for inference -- gradients are never computed through it, and its
1.1B parameters are never updated. Every CPTAC patch is embedded exactly
once; embeddings are cached to disk (FP16) and reused for every subsequent
distillation epoch (CRITICAL: 'Do not repeatedly run the 1.1B teacher every
training epoch.')."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class GenBioPathFMTeacher:
    """Loading/inference exactly follows the model card's documented usage
    (confirmed live against the real, public, non-gated
    genbio-ai/genbio-pathfm repo on 2026-08-28) -- NOT the generic
    HF-AutoModel-with-AutoImageProcessor pattern this module originally
    assumed:

      - There is no `preprocessor_config.json` in the repo, so
        `AutoImageProcessor.from_pretrained` fails; the model card instead
        specifies a plain torchvision `Resize(224,224) -> ToTensor ->
        Normalize` with GenBio-PathFM-SPECIFIC stats (NOT ImageNet's),
        mean=(0.697, 0.575, 0.728), std=(0.188, 0.240, 0.187).
      - `model(x)` returns the [B, 4608] CLS-feature tensor DIRECTLY (a
        plain tensor, not a HF-style ModelOutput with `.last_hidden_state`)."""

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
        """patches_uint8: (B, 224, 224, 3) uint8 RGB, already 224x224
        (Section 5's patch extraction already produces 224x224, so the
        model card's Resize((224,224)) is a no-op here -- applied anyway
        for safety against any non-224 input)."""
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(patches_uint8).to(self.device).float() / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, H, W, 3) -> (B, 3, H, W); the model's own
        # forward() does `pixel_values.view(...)` internally, which requires a contiguous tensor --
        # permute() alone returns a view with non-contiguous strides.
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.from_numpy(self._mean).to(self.device).view(1, 3, 1, 1)
        std = torch.from_numpy(self._std).to(self.device).view(1, 3, 1, 1)
        return ((x - mean) / std).contiguous().to(self.dtype)

    def embed_batch(self, patches_uint8: np.ndarray) -> np.ndarray:
        """patches_uint8: (B, H, W, 3) uint8 RGB. Returns (B, 4608) float32
        L2-normalized CLS embeddings (Section 7: 'Use the model's
        4608-dimensional CLS representation ... Normalize teacher
        embeddings using L2 normalization')."""
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


import h5py  # noqa: E402


def cache_teacher_embeddings(out_path: str | Path, patch_ids: list[str], embeddings: np.ndarray) -> None:
    """Section 7: store patch_id <-> teacher_embedding, FP16 on disk to
    reduce footprint (values are unit-norm so FP16 precision is adequate).

    Uses h5py's VARIABLE-length string dtype for `patch_id`, not a fixed-
    width numpy `dtype="S"` array. This dataset is appended to across many
    separate calls over the life of a run (one per batch, potentially
    tens of thousands of calls); a fixed-width `S` dtype freezes its
    per-element byte-width at whatever the FIRST call's longest string
    happened to be, and every later `dataset[slice] = np.array(longer_ids,
    dtype='S')` silently TRUNCATES any id exceeding that width -- no error,
    no warning. This bit a real run: CPTAC patch_ids
    (`{dicom_series_uid}_{x}_{y}`) vary in length with the sampled (x, y)
    pixel coordinates' digit counts (up to ~6 chars difference between a
    slide's smallest and largest coordinates), so a dataset created from
    an early, short-coordinate batch truncated later, longer patch_ids --
    corrupting them into collisions with unrelated patches (two different
    truncated-to-the-same-prefix ids look identical) and permanently
    breaking this cache's own resumability (a fresh, untruncated patch_id
    string from patch_manifest.csv never string-matches its own truncated
    form already on disk, so `_already_cached_patch_ids` never recognizes
    it as done and re-embeds it every run). Variable-length strings have
    no such width limit."""
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
    """Section 7 integrity checks: no NaNs, expected dimension, sensible
    norms, and patch<->embedding alignment (unique, non-empty patch_ids)."""
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
