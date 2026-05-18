"""
VideoDataset with raw-frame pipeline.

Training uses raw frames only — no YOLO ROI cropping.
YOLO is used exclusively at inference time (inference.py).

Frame cache: pre-extracted uint8 .npy files (~9 GB for 1996 clips).
Built once on first run (~15 min); every subsequent epoch reads np.load()
(~2 ms/clip) instead of seeking video (~100 ms/clip).
"""

import os
import cv2
import hashlib
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from tqdm import tqdm

from config import Config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _video_meta(path: str) -> Tuple[float, int, int, int]:
    """Return (fps, total_frames, width, height) or (0,0,0,0) on failure."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, nf, w, h


def _sample_frame_indices(fps: float, total_frames: int, num_frames: int,
                           temporal_jitter: bool = False) -> List[int]:
    """
    Sample `num_frames` frame indices uniformly in time across the clip.
    With temporal_jitter, each index is perturbed by ±1 frame (training only).
    """
    if total_frames <= num_frames:
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices.append(indices[-1])
        return indices

    step = total_frames / num_frames
    indices = [int(i * step + step / 2) for i in range(num_frames)]

    if temporal_jitter:
        indices = [
            max(0, min(total_frames - 1, idx + random.randint(-1, 1)))
            for idx in indices
        ]
    return indices


def _read_frames(path: str, indices: List[int], size: int) -> Optional[np.ndarray]:
    """
    Read specific frame indices and resize to size×size.
    Returns (N, size, size, 3) float32 in [0, 1].
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    idx_set = sorted(set(indices))
    frame_map: Dict[int, np.ndarray] = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx_set[0])
    current = idx_set[0]

    for target in idx_set:
        if target < current:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            current = target
        while current < target:
            cap.grab()
            current += 1
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        frame_map[target] = frame
        current += 1

    cap.release()

    frames = []
    for idx in indices:
        if idx in frame_map:
            frames.append(frame_map[idx])
        else:
            candidates = [k for k in frame_map if k <= idx]
            if not candidates:
                return None
            frames.append(frame_map[max(candidates)])

    return np.stack(frames).astype(np.float32) / 255.0


# ── Frame cache ───────────────────────────────────────────────────────────────

def _cache_key(path: str) -> str:
    h = hashlib.md5(path.encode()).hexdigest()[:8]
    return Path(path).stem + "_" + h


def precompute_frame_cache(cfg: Config, video_paths: List[str]) -> str:
    """
    Pre-extract resized raw frames to disk as uint8 .npy files.
    Only processes clips not already cached — safe to call after adding new videos.
    """
    cache_dir = cfg.frame_cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    pending = [p for p in video_paths
               if not os.path.exists(os.path.join(cache_dir, _cache_key(p) + ".npy"))]

    if not pending:
        print(f"Frame cache up to date  ({len(video_paths)} clips)  →  {cache_dir}")
        return cache_dir

    gb = len(video_paths) * cfg.num_frames * cfg.img_size * cfg.img_size * 3 / 1e9
    print(f"Pre-extracting frames: {len(pending)} clips remaining  "
          f"(total cache ≈ {gb:.1f} GB)  →  {cache_dir}")

    for path in tqdm(pending, desc="Frame cache", unit="clip"):
        fps, total_frames, _, _ = _video_meta(path)
        if fps <= 0 or total_frames <= 0:
            continue
        indices = _sample_frame_indices(fps, total_frames, cfg.num_frames)
        frames = _read_frames(path, indices, cfg.img_size)
        if frames is not None:
            out = os.path.join(cache_dir, _cache_key(path) + ".npy")
            np.save(out, (frames * 255).astype(np.uint8))

    print(f"Frame cache saved  →  {cache_dir}")
    return cache_dir


# ── Temporally consistent augmentation ───────────────────────────────────────

_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_NORM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _TemporalAugment:
    """
    Temporally consistent spatial augmentation for a clip.
    All T frames share ONE set of randomly sampled parameters.
    """

    def __init__(self, cfg: Config):
        self.hflip_p = cfg.hflip_p
        self.cj_p = cfg.color_jitter_p
        self.brightness = 0.3
        self.contrast = 0.3
        self.saturation = 0.2
        self.hue = 0.05

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.hflip_p:
            tensor = torch.flip(tensor, dims=[-1])
        if random.random() < self.cj_p:
            tensor = self._consistent_jitter(tensor)
        return (tensor - _NORM_MEAN) / _NORM_STD

    def _consistent_jitter(self, tensor: torch.Tensor) -> torch.Tensor:
        b = random.uniform(max(0.0, 1 - self.brightness), 1 + self.brightness)
        c = random.uniform(max(0.0, 1 - self.contrast),   1 + self.contrast)
        s = random.uniform(max(0.0, 1 - self.saturation), 1 + self.saturation)
        h = random.uniform(-self.hue, self.hue)
        tensor = TF.adjust_brightness(tensor, b)
        tensor = TF.adjust_contrast(tensor, c)
        tensor = TF.adjust_saturation(tensor, s)
        tensor = TF.adjust_hue(tensor, h)
        return tensor


class _ValTransform:
    """Normalize only — no augmentation for validation."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - _NORM_MEAN) / _NORM_STD


# ── Dataset ───────────────────────────────────────────────────────────────────

class ViolenceDataset(Dataset):
    """
    Returns a dict:
        frames  : (T, C, H, W) float32 tensor, normalised
        label   : scalar long tensor  (0=NonViolence, 1=Violence)
        path    : video path string (for debugging)
    """

    def __init__(
        self,
        video_paths: List[str],
        labels: List[int],
        cfg: Config,
        is_train: bool = True,
        frame_cache_dir: Optional[str] = None,
    ):
        self.paths = video_paths
        self.labels = labels
        self.cfg = cfg
        self.is_train = is_train
        self.transform = _TemporalAugment(cfg) if is_train else _ValTransform()

        self._npy: Dict[str, str] = {}
        if frame_cache_dir is not None:
            for p in video_paths:
                npy = os.path.join(frame_cache_dir, _cache_key(p) + ".npy")
                if os.path.exists(npy):
                    self._npy[p] = npy
            cached = len(self._npy)
            print(f"  {'Train' if is_train else 'Val  '} dataset: "
                  f"{cached}/{len(video_paths)} clips served from frame cache")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        label = self.labels[idx]

        if path in self._npy:
            frames = np.load(self._npy[path]).astype(np.float32) / 255.0
        else:
            fps, total_frames, _, _ = _video_meta(path)
            jitter = self.is_train and self.cfg.temporal_jitter
            frame_indices = _sample_frame_indices(fps, total_frames, self.cfg.num_frames, jitter)
            frames = _read_frames(path, frame_indices, self.cfg.img_size)
            if frames is None:
                frames = np.zeros(
                    (self.cfg.num_frames, self.cfg.img_size, self.cfg.img_size, 3),
                    dtype=np.float32,
                )

        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
        return {
            "frames": self.transform(tensor),
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
        }


# ── Data loading ──────────────────────────────────────────────────────────────

def collect_video_paths(cfg: Config) -> Tuple[List[str], List[int]]:
    """Walk data_root/{Violence,NonViolence} and return (paths, labels)."""
    paths, labels = [], []
    for label_idx, cls_name in enumerate(cfg.classes):
        cls_dir = os.path.join(cfg.data_root, cls_name)
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.endswith((".mp4", ".avi", ".mov")):
                continue
            fpath = os.path.join(cls_dir, fname)
            fps, nf, _, _ = _video_meta(fpath)
            if fps <= 0 or nf <= 0:
                continue
            dur = nf / fps
            if dur < cfg.min_clip_duration or dur > cfg.max_clip_duration:
                continue
            paths.append(fpath)
            labels.append(label_idx)
    return paths, labels


def build_dataloaders(cfg: Config):
    """Returns (train_loader, val_loader, None)."""
    from torch.utils.data import DataLoader
    import sklearn.model_selection as ms

    paths, labels = collect_video_paths(cfg)
    print(f"Dataset: {len(paths)} clips  "
          f"({labels.count(1)} Violence / {labels.count(0)} NonViolence)")

    train_p, val_p, train_l, val_l = ms.train_test_split(
        paths, labels,
        test_size=1 - cfg.train_ratio,
        stratify=labels,
        random_state=cfg.seed,
    )

    frame_cache_dir = None
    if cfg.use_frame_cache:
        frame_cache_dir = precompute_frame_cache(cfg, paths)

    train_ds = ViolenceDataset(train_p, train_l, cfg, is_train=True,
                               frame_cache_dir=frame_cache_dir)
    val_ds   = ViolenceDataset(val_p,   val_l,   cfg, is_train=False,
                               frame_cache_dir=frame_cache_dir)

    _w = cfg.num_workers
    loader_kwargs = dict(
        num_workers=_w, pin_memory=True,
        persistent_workers=(_w > 0),
        prefetch_factor=(2 if _w > 0 else None),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        **loader_kwargs,
    )
    val_batch = cfg.batch_size * cfg.val_batch_multiplier
    val_loader = DataLoader(
        val_ds, batch_size=val_batch, shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, None
