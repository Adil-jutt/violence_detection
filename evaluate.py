"""
Offline evaluation script.

Loads the best checkpoint and runs on the full validation split.
Reports: accuracy, F1, precision, recall, AUC-ROC, confusion matrix, and
per-class AP.  All plots saved under ./eval_outputs/.

Usage
─────
    conda run -n contrastive310 python evaluate.py
    conda run -n contrastive310 python evaluate.py --checkpoint checkpoints/best.pth
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from torch.cuda.amp import autocast  # type: ignore[import]
from tqdm import tqdm

from config import Config
from dataset import build_dataloaders
from model import ViolenceDetector
from utils import load_checkpoint


OUT_DIR = "./eval_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


@torch.no_grad()
def run_inference(model: ViolenceDetector, loader, device: torch.device):
    model.eval()
    all_labels, all_probs, all_paths = [], [], []

    for batch in tqdm(loader, desc="Evaluating"):
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label"]

        with autocast():
            logits = model(frames)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()

        all_probs.extend(probs)
        all_labels.extend(labels.numpy().tolist())
        all_paths.extend(batch["path"])

    return np.array(all_labels), np.array(all_probs), all_paths


def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Youden's J statistic threshold (maximises sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def plot_confusion_matrix(labels, preds, save_path: str):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["NonViolence", "Violence"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (val)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(labels, probs, auc: float, save_path: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(labels, probs, ax=ax, name=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title("ROC Curve (val)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_score_distribution(labels, probs, threshold: float, save_path: str):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(probs[labels == 0], bins=40, alpha=0.6, color="steelblue", label="NonViolence")
    ax.hist(probs[labels == 1], bins=40, alpha=0.6, color="tomato", label="Violence")
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.2f}")
    ax.set_xlabel("Violence probability")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution (val)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate(cfg: Config, checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, _ = build_dataloaders(cfg)

    model = ViolenceDetector(cfg).to(device)
    load_checkpoint(model, checkpoint_path, device)

    labels, probs, paths = run_inference(model, val_loader, device)

    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    thresh = find_optimal_threshold(labels, probs)
    preds = (probs >= thresh).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")

    print("\n" + "=" * 55)
    print(f"  Threshold (Youden J): {thresh:.4f}")
    print(f"  Accuracy           : {acc:.4f}")
    print(f"  F1 (binary)        : {f1:.4f}")
    print(f"  AUC-ROC            : {auc:.4f}")
    print(f"  Average Precision  : {ap:.4f}")
    print("=" * 55)
    print(classification_report(labels, preds, target_names=["NonViolence", "Violence"]))

    plot_confusion_matrix(labels, preds, os.path.join(OUT_DIR, "confusion_matrix.png"))
    plot_roc_curve(labels, probs, auc, os.path.join(OUT_DIR, "roc_curve.png"))
    plot_score_distribution(labels, probs, thresh, os.path.join(OUT_DIR, "score_dist.png"))

    # surface the worst false positives and false negatives
    fp_indices = np.where((preds == 1) & (labels == 0))[0]
    fn_indices = np.where((preds == 0) & (labels == 1))[0]

    print(f"\nFalse Positives ({len(fp_indices)}):")
    for i in sorted(fp_indices, key=lambda x: -probs[x])[:10]:
        print(f"  score={probs[i]:.3f}  {os.path.basename(paths[i])}")

    print(f"\nFalse Negatives ({len(fn_indices)}):")
    for i in sorted(fn_indices, key=lambda x: probs[x])[:10]:
        print(f"  score={probs[i]:.3f}  {os.path.basename(paths[i])}")

    print(f"\nPlots saved to {OUT_DIR}/")
    return {"acc": acc, "f1": f1, "auc": auc, "ap": ap, "threshold": thresh}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="./checkpoints/best.pth")
    args = p.parse_args()
    cfg = Config()
    evaluate(cfg, args.checkpoint)
