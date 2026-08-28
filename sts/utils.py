"""Shared utilities: config loading, seeding, JSON I/O, run-environment
capture (Section 3: git commit, package versions, CUDA/GPU, wall-clock,
GPU-hours, peak VRAM)."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from string import Template
from typing import Any

import numpy as np


def load_config(path: str | Path) -> dict:
    """Prefers a sibling .json (PyYAML is not guaranteed present on every
    cluster module) and falls back to real YAML parsing otherwise. Resolves
    ``${experiment_name}``-style templates against the config's own
    top-level scalar values (used for paths.work_dir)."""
    path = Path(path)
    json_sibling = path.with_suffix(".json")
    if json_sibling.exists():
        with open(json_sibling) as f:
            cfg = json.load(f)
    else:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)
    return _resolve_templates(cfg, cfg)


def _resolve_templates(node, root: dict):
    if isinstance(node, str) and "${" in node:
        flat = {k: v for k, v in root.items() if isinstance(v, (str, int, float))}
        return Template(node).safe_substitute(flat)
    if isinstance(node, dict):
        return {k: _resolve_templates(v, root) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_templates(v, root) for v in node]
    return node


def resolve_run_paths(cfg: dict) -> dict[str, Path]:
    """Turns cfg['paths'] into absolute Path objects under work_dir and
    creates every directory listed in Final_Project.txt Section 3 (configs/
    logs/ manifests/ qc/ teacher_embeddings/ student_checkpoints/
    student_embeddings/ graphs/ predictions/ results/ figures/).

    `teacher_embeddings` and `student_checkpoints` are the two expensive,
    reusable artifacts this project is explicitly asked to cache once and
    reuse (the frozen GenBio-PathFM teacher's cached embeddings, and the
    final locked ConvNeXt V2-Atto encoder). If cfg['paths'] sets
    `persist_dir`, those two subdirs are created there instead of under the
    (possibly ephemeral/scratch) work_dir, so they survive independently of
    any particular run and are trivially reusable by later runs/experiments."""
    work_dir = Path(cfg["paths"]["work_dir"])
    persist_dir = cfg["paths"].get("persist_dir")
    persist_dir = Path(persist_dir) if persist_dir else None

    ephemeral_subdirs = ["configs", "logs", "manifests", "qc", "student_embeddings",
                         "graphs", "predictions", "results", "figures"]
    persistable_subdirs = ["teacher_embeddings", "student_checkpoints"]

    out = {"work_dir": work_dir}
    for name in ephemeral_subdirs:
        p = work_dir / name
        p.mkdir(parents=True, exist_ok=True)
        out[name] = p
    for name in persistable_subdirs:
        p = (persist_dir if persist_dir else work_dir) / name
        p.mkdir(parents=True, exist_ok=True)
        out[name] = p
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class RunEnvironment:
    """Captures the Section 3 reproducibility record. Use as a context
    manager around any expensive stage; writes a JSON record with
    wall-clock seconds, peak VRAM (if CUDA available), package versions,
    CUDA version, GPU model, and the current git commit (if this directory
    is version-controlled -- it usually is not, so that field is simply
    omitted rather than failing the run)."""

    def __init__(self, stage_name: str, out_path: str | Path):
        self.stage_name = stage_name
        self.out_path = Path(out_path)
        self._t0 = None

    def __enter__(self):
        self._t0 = time.time()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        wall_clock_s = time.time() - self._t0
        record = {
            "stage": self.stage_name,
            "wall_clock_seconds": wall_clock_s,
            "wall_clock_hours": wall_clock_s / 3600.0,
            "success": exc_type is None,
            "python_version": sys.version,
            "git_commit": _git_commit(),
            "package_versions": _package_versions(),
        }
        record.update(_gpu_record())
        save_json(record, self.out_path)
        return False  # never swallow exceptions -- failures must surface (Section: CRITICAL RULE)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def _package_versions() -> dict:
    versions = {}
    for pkg in ["torch", "torchvision", "timm", "torch_geometric", "numpy", "pandas",
                "scipy", "sklearn", "scanpy", "anndata", "h5py", "pyarrow", "safetensors"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = None
    return versions


def _gpu_record() -> dict:
    out = {"cuda_version": None, "gpu_model": None, "peak_vram_gb": None}
    try:
        import torch
        if torch.cuda.is_available():
            out["cuda_version"] = torch.version.cuda
            out["gpu_model"] = torch.cuda.get_device_name(0)
            out["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    except ImportError:
        pass
    return out


def count_parameters(model) -> dict:
    """Section 27/Step 23: exact, programmatically-counted parameter
    counts. Never estimate manually."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_parameters": int(trainable), "total_parameters": int(total)}
