"""
Training script.

Strategy
────────
Epoch 0  – warmup_epochs : backbone fully frozen, only temporal head trains.
                            LR ramps from 0 → cfg.lr (linear warmup).
Epoch unfreeze_epoch      : top 3 backbone blocks unfrozen.
                            New param group added to the EXISTING optimizer so
                            head momentum/variance accumulations are preserved.
                            LR switches to cosine annealing over remaining epochs.
Epoch unfreeze_epoch + N  : cosine annealing to zero over remaining epochs.

Production features
───────────────────
• torch.backends.cudnn.benchmark — ~10–15% speedup for fixed input size.
• Gradient accumulation          — effective batch = batch_size × grad_accum_steps.
• Early stopping                 — halts when val F1 stagnates for patience epochs.
• Checkpoint resumption          — --resume path/to/checkpoint.pth
• Per-epoch history JSON         — crash-safe; written after every epoch.
• LR logging                     — all param-group LRs printed each epoch.
• Temporally consistent augment  — see dataset.py _TemporalAugment.

Usage
─────
    conda run -n contrastive310 python train.py
    conda run -n contrastive310 python train.py --lr 1e-4 --batch_size 8
    conda run -n contrastive310 python train.py --resume checkpoints/best.pth
    conda run -n contrastive310 python train.py --grad_accum_steps 4
"""

import argparse
import json
import os
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import random
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from torch.cuda.amp import GradScaler, autocast  # type: ignore[import]
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

from config import Config
from dataset import build_dataloaders
from model import ViolenceDetector, get_param_groups
from utils import save_checkpoint, load_checkpoint, AverageMeter, compute_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Early stopping ─────────────────────────────────────────────────────────────

class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = 0.0
        self.counter = 0

    def step(self, metric: float) -> bool:
        """Returns True when training should stop."""
        if metric > self.best:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ── Scheduler ─────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """
    Linear warmup (by epoch) followed by cosine annealing.
    The warmup LambdaLR operates on step-level for smooth ramp.
    """
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps = cfg.num_epochs * steps_per_epoch

    def warmup_fn(step):
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-6)
        return 1.0

    warmup = LambdaLR(optimizer, lr_lambda=warmup_fn)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=(cfg.num_epochs - cfg.warmup_epochs) * steps_per_epoch,
        eta_min=1e-7,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


# ── Training loop ──────────────────────────────────────────────────────────────

def train_one_epoch(
    model: ViolenceDetector,
    loader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    criterion,
    device: torch.device,
    cfg: Config,
) -> dict:
    model.train()
    loss_m = AverageMeter()
    all_labels, all_probs = [], []

    # Zero before the loop; re-zero after each optimizer step inside.
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for i, batch in enumerate(pbar):
        frames = batch["frames"].to(device, non_blocking=True)  # (B,T,C,H,W)
        labels = batch["label"].to(device, non_blocking=True).float()

        with autocast():
            logits = model(frames)
            # Divide loss by accum steps so gradients sum to the correct scale.
            loss = criterion(logits, labels) / cfg.grad_accum_steps

        scaler.scale(loss).backward()

        is_update_step = ((i + 1) % cfg.grad_accum_steps == 0
                          or (i + 1) == len(loader))
        if is_update_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # Track the unscaled loss for display.
        loss_m.update(loss.item() * cfg.grad_accum_steps, frames.size(0))
        all_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        pbar.set_postfix(loss=f"{loss_m.avg:.4f}")

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = loss_m.avg
    return metrics


@torch.no_grad()
def validate(
    model: ViolenceDetector,
    loader,
    criterion,
    device: torch.device,
) -> dict:
    model.eval()
    loss_m = AverageMeter()
    all_labels, all_probs = [], []

    pbar = tqdm(loader, desc="  Val  ", leave=False, dynamic_ncols=True)
    for batch in pbar:
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).float()

        with autocast():
            logits = model(frames)
            loss = criterion(logits, labels)

        loss_m.update(loss.item(), frames.size(0))
        all_probs.extend(torch.sigmoid(logits).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        pbar.set_postfix(loss=f"{loss_m.avg:.4f}")

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = loss_m.avg
    return metrics


# ── Main training function ─────────────────────────────────────────────────────

def train(cfg: Config, resume_path: str = None):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        # cuDNN auto-selects the fastest algorithm for fixed input shapes.
        # deterministic=False trades reproducibility for ~10-15% throughput.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    print(f"Device: {device}  |  "
          f"GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'}")

    train_loader, val_loader, _ = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}  |  "
          f"Effective batch: {cfg.batch_size * cfg.grad_accum_steps}")

    model = ViolenceDetector(cfg).to(device)

    # Label smoothing applied manually so we don't couple it to BCEWithLogitsLoss.
    # Defined once here — not recreated inside the epoch loop.
    def smoothed_criterion(logits, labels, eps=cfg.label_smoothing):
        labels_smooth = labels * (1 - eps) + 0.5 * eps
        return nn.functional.binary_cross_entropy_with_logits(logits, labels_smooth)

    optimizer = AdamW(
        get_param_groups(model, cfg),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scaler = GradScaler()

    start_epoch = 1
    best_val_f1 = 0.0
    backbone_unfrozen = False
    history = []

    # ── Checkpoint resumption ──────────────────────────────────────────────────
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        prev_epoch = ckpt.get("epoch", 0)
        start_epoch = prev_epoch + 1
        best_val_f1 = ckpt.get("metrics", {}).get("f1", 0.0)

        # Restore backbone unfreeze state BEFORE loading optimizer state,
        # so the param groups match what was saved.
        if prev_epoch >= cfg.unfreeze_epoch:
            model.unfreeze_backbone(num_blocks=3)
            backbone_unfrozen = True

        # Rebuild optimizer with correct param groups, then restore momentum.
        optimizer = AdamW(
            get_param_groups(model, cfg),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Restore history from JSON if available.
        hist_path = os.path.join(cfg.checkpoint_dir, "history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)

        print(f"Resumed from {resume_path}  "
              f"(epoch {prev_epoch}  |  best F1 {best_val_f1:.4f})")

    # ── Scheduler (built / rebuilt after resume) ───────────────────────────────
    if backbone_unfrozen:
        remaining = max(1, cfg.num_epochs - (start_epoch - 1))
        scheduler = CosineAnnealingLR(
            optimizer, T_max=remaining * len(train_loader), eta_min=1e-7
        )
    else:
        scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    # Restore scheduler position when resuming.
    if resume_path and os.path.exists(resume_path):
        sd = ckpt.get("scheduler_state_dict")
        if sd is not None:
            try:
                scheduler.load_state_dict(sd)
            except Exception:
                pass  # type mismatch across unfreeze boundary — skip gracefully

    history_path = os.path.join(cfg.checkpoint_dir, "history.json")
    early_stopper = EarlyStopper(patience=cfg.early_stop_patience)
    # Seed early stopper with best seen so far so patience counts from reality.
    early_stopper.best = best_val_f1

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        t0 = time.time()

        # Progressive unfreezing: add backbone param group to the EXISTING
        # optimizer so the head's accumulated momentum is preserved.
        if epoch == cfg.unfreeze_epoch and not backbone_unfrozen:
            model.unfreeze_backbone(num_blocks=3)
            backbone_unfrozen = True
            backbone_params = [p for p in model.spatial.parameters()
                               if p.requires_grad]
            if backbone_params:
                optimizer.add_param_group({
                    "params": backbone_params,
                    "lr": cfg.lr * cfg.backbone_lr_scale,
                    "weight_decay": cfg.weight_decay,
                })
            remaining = max(1, cfg.num_epochs - epoch + 1)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=remaining * len(train_loader), eta_min=1e-7
            )
            print(f"  ↑ Backbone top-3 blocks unfrozen at epoch {epoch}")

        train_m = train_one_epoch(model, train_loader, optimizer, scheduler,
                                   scaler, smoothed_criterion, device, cfg)
        val_m   = validate(model, val_loader, smoothed_criterion, device)

        elapsed = time.time() - t0

        lrs = "/".join(f"{pg['lr']:.1e}" for pg in optimizer.param_groups)
        print(
            f"Epoch {epoch:03d}/{cfg.num_epochs}  "
            f"train loss={train_m['loss']:.4f} acc={train_m['acc']:.4f} "
            f"f1={train_m['f1']:.4f}  |  "
            f"val   loss={val_m['loss']:.4f} acc={val_m['acc']:.4f} "
            f"f1={val_m['f1']:.4f}  "
            f"lr={lrs}  ({elapsed:.1f}s)"
        )

        history.append({"epoch": epoch, "train": train_m, "val": val_m})

        # Persist history after every epoch — survives crashes.
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        save_checkpoint(model, optimizer, epoch, val_m, cfg,
                        tag="last", scheduler=scheduler)

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            save_checkpoint(model, optimizer, epoch, val_m, cfg,
                            tag="best", scheduler=scheduler)
            print(f"  ✓ New best val F1: {best_val_f1:.4f}")

        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_m, cfg,
                            tag=f"epoch{epoch}", scheduler=scheduler)

        # Early stopping — check AFTER checkpoint so best is always saved.
        if early_stopper.step(val_m["f1"]):
            print(f"\nEarly stopping triggered  "
                  f"(patience={cfg.early_stop_patience}, "
                  f"best F1={early_stopper.best:.4f})")
            break

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    return history


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr",               type=float)
    p.add_argument("--batch_size",       type=int)
    p.add_argument("--num_epochs",       type=int)
    p.add_argument("--unfreeze_epoch",   type=int)
    p.add_argument("--num_frames",       type=int)
    p.add_argument("--grad_accum_steps", type=int,
                   help="Gradient accumulation steps (effective batch multiplier)")
    p.add_argument("--early_stop_patience", type=int)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume training from")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()
    for k, v in vars(args).items():
        if v is not None and k != "resume":
            setattr(cfg, k, v)
    train(cfg, resume_path=args.resume)
