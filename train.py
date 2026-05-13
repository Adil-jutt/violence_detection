"""
Training script.

Strategy
────────
Epoch 0  – warmup_epochs : backbone fully frozen, only temporal head trains.
                            LR ramps from 0 → cfg.lr (linear warmup).
Epoch unfreeze_epoch      : top 3 backbone blocks unfrozen with 10× lower LR.
Epoch unfreeze_epoch + N  : cosine annealing to zero over remaining epochs.

Mixed precision (AMP) + gradient clipping keeps GPU memory and training stable.
Label smoothing (0.1) reduces overconfidence on a relatively small dataset.

Usage
─────
    conda run -n contrastive310 python train.py
    conda run -n contrastive310 python train.py --lr 1e-4 --batch_size 8
"""

import argparse
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


def train_one_epoch(
    model: ViolenceDetector,
    loader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
) -> dict:
    model.train()
    loss_m = AverageMeter()
    all_labels, all_probs = [], []

    pbar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for batch in pbar:
        frames = batch["frames"].to(device, non_blocking=True)  # (B,T,C,H,W)
        labels = batch["label"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(frames)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_m.update(loss.item(), frames.size(0))
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
    criterion: nn.Module,
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


def train(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  GPU: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'N/A'}")

    train_loader, val_loader, _ = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    model = ViolenceDetector(cfg).to(device)
    criterion = nn.BCEWithLogitsLoss(
        reduction="mean",
        # label smoothing: replace 0/1 targets with ε/2 and 1-ε/2
        pos_weight=None,
    )

    # Initial optimizer — only temporal head params (backbone still frozen)
    optimizer = AdamW(
        get_param_groups(model, cfg),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler()

    best_val_f1 = 0.0
    backbone_unfrozen = False
    history = []

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()

        # Progressive unfreezing
        if epoch == cfg.unfreeze_epoch and not backbone_unfrozen:
            model.unfreeze_backbone(num_blocks=3)
            backbone_unfrozen = True
            # rebuild optimizer with backbone param group
            optimizer = AdamW(
                get_param_groups(model, cfg),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
            # rebuild scheduler for remaining epochs
            remaining = cfg.num_epochs - epoch + 1
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining * len(train_loader), eta_min=1e-7)
            print(f"  ↑ Backbone top-3 blocks unfrozen at epoch {epoch}")

        # Label smoothing applied manually so BCEWithLogitsLoss stays standard
        def smoothed_criterion(logits, labels, eps=cfg.label_smoothing):
            labels_smooth = labels * (1 - eps) + 0.5 * eps
            return nn.functional.binary_cross_entropy_with_logits(logits, labels_smooth)

        train_m = train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                                   smoothed_criterion, device, cfg)
        val_m = validate(model, val_loader, smoothed_criterion, device)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{cfg.num_epochs}  "
            f"train loss={train_m['loss']:.4f} acc={train_m['acc']:.4f} f1={train_m['f1']:.4f}  |  "
            f"val   loss={val_m['loss']:.4f} acc={val_m['acc']:.4f} f1={val_m['f1']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        history.append({"epoch": epoch, "train": train_m, "val": val_m})

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            save_checkpoint(model, optimizer, epoch, val_m, cfg, tag="best")
            print(f"  ✓ New best val F1: {best_val_f1:.4f}")

        # periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, val_m, cfg, tag=f"epoch{epoch}")

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    return history


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--num_epochs", type=int)
    p.add_argument("--unfreeze_epoch", type=int)
    p.add_argument("--num_frames", type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()
    for k, v in vars(args).items():
        if v is not None:
            setattr(cfg, k, v)
    train(cfg)
