"""Distillation objective (cosine + normalized feature-regression) and the training loop
implementing the S0/S1/S2 baseline definitions: ImageNet-only, ImageNet-init distilled, random-init distilled."""
from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from epathst.models.student import DistillationStudent
from epathst.utils import save_json

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def cosine_distillation_loss(
    student_projected: torch.Tensor, teacher_embedding: torch.Tensor
) -> torch.Tensor:
    """L_cos = 1 - cosine_similarity(normalize(student_proj), normalize(teacher))."""
    s = F.normalize(student_projected, dim=-1)
    t = F.normalize(teacher_embedding, dim=-1)
    cos_sim = (s * t).sum(dim=-1)
    return (1.0 - cos_sim).mean()


def feature_regression_loss(student_projected: torch.Tensor, teacher_embedding: torch.Tensor,
                             kind: str = "smooth_l1") -> torch.Tensor:
    """Normalized feature-regression term (never raw unnormalized MSE)."""
    s = F.normalize(student_projected, dim=-1)
    t = F.normalize(teacher_embedding, dim=-1)
    if kind == "smooth_l1":
        return F.smooth_l1_loss(s, t)
    if kind == "normalized_mse":
        return F.mse_loss(s, t)
    raise ValueError(f"unknown feature loss kind: {kind}")


def distillation_loss(student_projected: torch.Tensor, teacher_embedding: torch.Tensor, lam: float,
                       feature_kind: str = "smooth_l1") -> dict:
    l_cos = cosine_distillation_loss(student_projected, teacher_embedding)
    l_feat = feature_regression_loss(student_projected, teacher_embedding, feature_kind)
    total = l_cos + lam * l_feat
    return {"loss": total, "l_cos": l_cos.detach(), "l_feature": l_feat.detach()}


def augment_train(patch_uint8: np.ndarray, rng: np.random.RandomState, cfg: dict) -> np.ndarray:
    """Flips, 90-deg rotations, moderate color/stain augmentation, mild brightness/contrast -- never destructive to tissue morphology."""
    aug = cfg["distillation"]["augmentation"]
    img = patch_uint8.copy()
    if aug["hflip"] and rng.rand() < 0.5:
        img = img[:, ::-1, :]
    if aug["vflip"] and rng.rand() < 0.5:
        img = img[::-1, :, :]
    if aug["rot90"]:
        k = rng.randint(0, 4)
        if k:
            img = np.rot90(img, k=k, axes=(0, 1)).copy()

    x = img.astype(np.float32) / 255.0
    bc = aug["brightness_contrast"]
    brightness = 1.0 + rng.uniform(-bc, bc)
    contrast = 1.0 + rng.uniform(-bc, bc)
    mean = x.mean(axis=(0, 1), keepdims=True)
    x = (x - mean) * contrast + mean * brightness
    cj = aug["color_jitter"]
    channel_scale = 1.0 + rng.uniform(-cj / 2, cj / 2, size=(1, 1, 3)).astype(np.float32)
    x = np.clip(x * channel_scale, 0.0, 1.0)
    return x


def to_tensor_normalized(img_float01: np.ndarray) -> np.ndarray:
    normed = (img_float01 - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(normed, (2, 0, 1)).astype(np.float32)


class CPTACDistillDataset(Dataset):
    """Reads patches on-demand from per-WSI HDF5 shards (epathst.data.wsi_patches format) and pairs each with its cached teacher embedding by patch_id; asserts every manifest patch has a cached embedding before training starts."""

    def __init__(
        self, patch_manifest, teacher_h5_path: str, shard_dir: str, train: bool, cfg: dict
    ):
        self.manifest = patch_manifest.reset_index(drop=True)
        self.shard_dir = Path(shard_dir)
        self.train = train
        self.cfg = cfg
        with h5py.File(teacher_h5_path, "r") as f:
            pids = f["patch_id"][:].astype(str)
            emb = f["teacher_embedding"][:].astype(np.float32)
        self._teacher_by_id = dict(zip(pids, emb))
        missing = set(self.manifest["patch_id"]) - set(self._teacher_by_id)
        assert not missing, f"{len(missing)} patches in manifest have no cached teacher embedding"

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        with h5py.File(self.shard_dir / f"{row['wsi_id']}.h5", "r") as f:
            pids = f["patch_id"][:].astype(str)
            pos = int(np.where(pids == row["patch_id"])[0][0])
            patch = f["patches"][pos]
        rng = np.random.RandomState(abs(hash((row["patch_id"], idx))) % (2 ** 31))
        img = (
            augment_train(patch, rng, self.cfg)
            if self.train
            else patch.astype(np.float32) / 255.0
        )
        x = to_tensor_normalized(img)
        teacher_emb = self._teacher_by_id[row["patch_id"]]
        return torch.from_numpy(x), torch.from_numpy(teacher_emb), row["patch_id"]


def build_optimizer(model: DistillationStudent, cfg: dict):
    d = cfg["distillation"]
    return torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": d["backbone_lr"]},
        {"params": model.projection_head.parameters(), "lr": d["projection_lr"]},
    ], weight_decay=d["weight_decay"])


def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate_val_cosine(model, loader, device) -> float:
    model.eval()
    sims = []
    for x, teacher_emb, _ in loader:
        x, teacher_emb = x.to(device), teacher_emb.to(device)
        out = model(x)
        s = torch.nn.functional.normalize(out["projected"], dim=-1)
        t = torch.nn.functional.normalize(teacher_emb, dim=-1)
        sims.append((s * t).sum(dim=-1).cpu().numpy())
    model.train()
    return float(np.concatenate(sims).mean()) if sims else float("nan")


def train_student(
    baseline: str,
    train_ds,
    val_ds,
    cfg: dict,
    out_dir: Path,
    seed: int,
    device: str = "cuda",
    time_budget_seconds: float | None = None,
) -> dict:
    """`baseline` in {"S1", "S2"} (S0 needs no training loop); saves the best checkpoint (by validation teacher-cosine similarity, never test) to out_dir/{baseline}_seed{seed}_best.pt, and is resumable/wall-clock-budgeted for SLURM: a full training-state snapshot (model+optimizer+scaler+step counters) is saved every epoch to out_dir/{baseline}_seed{seed}_resume.pt and reloaded automatically if present, so a job killed by the wall clock resumes on the next `sbatch`; if `time_budget_seconds` is set, training stops (after saving both checkpoints) once elapsed time exceeds it, returning `stopped_early_for_time=True` rather than raising."""
    import time
    from epathst.utils import seed_everything
    seed_everything(seed)

    # build model, optimizer, dataloaders
    pretrained = "imagenet" if baseline == "S1" else None
    model = DistillationStudent(cfg, pretrained=pretrained).to(device)
    d = cfg["distillation"]
    opt = build_optimizer(model, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=d["amp"])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=d["batch_size"], shuffle=True,
                                                num_workers=8, drop_last=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=d["batch_size"], shuffle=False, num_workers=4
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * d["max_epochs"]
    warmup_steps = steps_per_epoch * d["warmup_epochs"]

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{baseline}_seed{seed}_best.pt"
    resume_path = out_dir / f"{baseline}_seed{seed}_resume.pt"
    history_path = out_dir / f"{baseline}_seed{seed}_history.json"

    # resume from a prior wall-clock-budgeted run, if present
    start_epoch, best_val_cos, best_epoch, patience_ctr, global_step = 0, -1.0, -1, 0, 0
    history = []
    if resume_path.exists():
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        opt.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = state["epoch"] + 1
        best_val_cos, best_epoch = state["best_val_cos"], state["best_epoch"]
        patience_ctr, global_step = state["patience_ctr"], state["global_step"]
        history = state.get("history", [])
        print(f"[{baseline} seed{seed}] resumed from {resume_path} at epoch {start_epoch}")

    t0 = time.time()
    stopped_early_for_time = False

    for epoch in range(start_epoch, d["max_epochs"]):
        model.train()
        epoch_losses = []
        for x, teacher_emb, _ in train_loader:
            x, teacher_emb = x.to(device), teacher_emb.to(device)

            # lr warmup/cosine-decay schedule
            lr_scale = cosine_warmup_lr(global_step, total_steps, warmup_steps)
            for g, base_lr in zip(opt.param_groups, [d["backbone_lr"], d["projection_lr"]]):
                g["lr"] = base_lr * lr_scale

            # forward + distillation loss
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=d["amp"]):
                out = model(x)
                losses = distillation_loss(
                    out["projected"], teacher_emb, d["lambda_feature"], d["feature_loss"]
                )

            # backward + optimizer step
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), d["grad_clip_norm"])
            scaler.step(opt)
            scaler.update()
            epoch_losses.append(float(losses["loss"].item()))
            global_step += 1

            if (
                time_budget_seconds is not None
                and global_step % 200 == 0
                and (time.time() - t0) > time_budget_seconds
            ):
                stopped_early_for_time = True
                break

        # validation + logging
        val_cos = evaluate_val_cosine(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_cosine_sim": val_cos,
        })
        print(
            f"[{baseline} seed{seed}] epoch {epoch}: "
            f"train_loss={np.mean(epoch_losses):.4f} val_cos={val_cos:.4f}"
            + (" [time-budget hit mid-epoch]" if stopped_early_for_time else "")
        )

        # save best-so-far checkpoint
        if val_cos > best_val_cos:
            best_val_cos, best_epoch, patience_ctr = val_cos, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_cosine_sim": val_cos,
                "baseline": baseline,
                "seed": seed,
            }, ckpt_path)
        else:
            patience_ctr += 1

        # save resumable full training state
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_val_cos": best_val_cos,
            "best_epoch": best_epoch,
            "patience_ctr": patience_ctr,
            "global_step": global_step,
            "history": history,
        }, resume_path)

        # Milestone snapshots every 10 completed epochs (epoch 0-indexed, so epoch+1=completed count): permanent, never-overwritten, unlike `ckpt_path` (best-so-far) or `resume_path` (overwritten each epoch).
        completed = epoch + 1
        if completed % 10 == 0:
            milestone_path = out_dir / f"{baseline}_seed{seed}_epoch{completed}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_cosine_sim": val_cos,
                "train_loss": float(np.mean(epoch_losses)),
                "baseline": baseline,
                "seed": seed,
            }, milestone_path)
            print(
                f"[{baseline} seed{seed}] saved milestone checkpoint at epoch {completed} "
                f"-> {milestone_path}"
            )
        save_json({
            "history": history,
            "best_epoch": best_epoch,
            "best_val_cosine_sim": best_val_cos,
            "stopped_early_for_time": stopped_early_for_time,
        }, history_path)

        if stopped_early_for_time:
            print(
                f"[{baseline} seed{seed}] time budget ({time_budget_seconds}s) reached at "
                f"epoch {epoch} -- resume checkpoint saved, rerun to continue"
            )
            break

        if epoch + 1 >= d["min_epochs"] and patience_ctr >= d["early_stop_patience"]:
            print(f"[{baseline} seed{seed}] early stopping at epoch {epoch} (best={best_epoch})")
            break

    return {
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "best_val_cosine_sim": best_val_cos,
        "stopped_early_for_time": stopped_early_for_time,
    }
