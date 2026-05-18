"""Shared utilities: checkpointing, metrics, logging."""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# ── Metrics ───────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(labels, probs, threshold: float = 0.5) -> dict:
    labels = np.array(labels)
    probs = np.array(probs)
    preds = (probs >= threshold).astype(int)
    return {
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="binary", zero_division=0)),
        "precision": float(precision_score(labels, preds, average="binary", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="binary", zero_division=0)),
    }


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    cfg,
    tag: str = "best",
    scheduler=None,
):
    path = os.path.join(cfg.checkpoint_dir, f"{tag}.pth")
    data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "cfg": vars(cfg),  # save as plain dict so weights_only=True works
    }
    if scheduler is not None:
        data["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(data, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"Loaded checkpoint from {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt.get("metrics", {})


# ── Training history plot ─────────────────────────────────────────────────────

def plot_training_history(history: list, out_dir: str = "./eval_outputs"):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]
    train_f1 = [h["train"]["f1"] for h in history]
    val_f1 = [h["val"]["f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, train_loss, label="train")
    ax1.plot(epochs, val_loss, label="val")
    ax1.set_xlabel("Epoch");  ax1.set_ylabel("Loss");  ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, train_f1, label="train")
    ax2.plot(epochs, val_f1, label="val")
    ax2.set_xlabel("Epoch");  ax2.set_ylabel("F1");  ax2.set_title("F1 Score")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training history plot → {path}")
