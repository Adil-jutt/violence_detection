"""
VideoDataset with YOLO-based ROI extraction.

Pre-caching strategy: YOLO runs once offline (precompute_yolo_cache) and saves
union-person bboxes per video per sampled frame to a pickle. DataLoader reads
the cache rather than running inference every epoch — this makes training
~10× faster than online detection.

Sampling: time-proportional (not frame-index uniform) so 11fps and 37fps videos
are treated consistently. Each call samples `num_frames` evenly across the clip's
real duration, skipping both very short (<2 s) and long-tail outlier (>20 s) clips.
"""

import os
import cv2
import hashlib
import pickle
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
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


def _read_frames(
    path: str,
    indices: List[int],
    size: int,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    """
    Read specific frame indices, crop to ROI inline, resize to size×size.
    Returns (N, size, size, 3) float32 [0,1].

    Doing crop+resize here (not after stacking) keeps peak memory at
    (N, size, size, 3) instead of (N, native_H, native_W, 3) — ~40× less
    for 1080p sources.
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
        if roi is not None:
            fh, fw = frame.shape[:2]
            x1 = max(0, roi[0]);  y1 = max(0, roi[1])
            x2 = min(fw, roi[2]); y2 = min(fh, roi[3])
            if x2 > x1 and y2 > y1:
                frame = frame[y1:y2, x1:x2]
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


def _compute_union_roi(
    bboxes: List[Optional[Tuple[int, int, int, int]]],
    frame_w: int,
    frame_h: int,
    padding: float = 0.15,
) -> Tuple[int, int, int, int]:
    """
    Given a list of (x1,y1,x2,y2) person bboxes (None = no detection),
    return the padded union bbox clamped to frame bounds.
    Falls back to full-frame if no detections.
    """
    valid = [b for b in bboxes if b is not None]
    if not valid:
        return 0, 0, frame_w, frame_h

    x1 = min(b[0] for b in valid)
    y1 = min(b[1] for b in valid)
    x2 = max(b[2] for b in valid)
    y2 = max(b[3] for b in valid)

    pw = (x2 - x1) * padding
    ph = (y2 - y1) * padding
    x1 = max(0, int(x1 - pw))
    y1 = max(0, int(y1 - ph))
    x2 = min(frame_w, int(x2 + pw))
    y2 = min(frame_h, int(y2 + ph))
    return x1, y1, x2, y2


def _crop_and_resize(
    frames: np.ndarray,
    roi: Tuple[int, int, int, int],
    size: int,
) -> np.ndarray:
    """Crop ROI from all frames and resize to (size, size). Returns (N, size, size, 3)."""
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, frames.shape[2], frames.shape[1]

    cropped = frames[:, y1:y2, x1:x2, :]
    resized = np.stack([
        cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
        for f in cropped
    ])
    return resized


# ── Frame cache helpers ────────────────────────────────────────────────────────

def _cache_key(path: str) -> str:
    """Deterministic cache key: stem + 8-char md5 suffix."""
    h = hashlib.md5(path.encode()).hexdigest()[:8]
    return Path(path).stem + "_" + h


# ── YOLO ROI pre-computation ──────────────────────────────────────────────────

def precompute_yolo_cache(cfg: Config, video_paths: List[str]) -> Dict:
    """
    Run YOLOv8n person detection on sampled frames for every video and save
    the union ROI per video to `cfg.yolo_cache_path`.

    Cache schema:
        {video_path: {"roi": (x1,y1,x2,y2), "frame_w": int, "frame_h": int}}

    The union ROI is computed across all sampled frames' detections so a single
    stable crop is used per clip (avoids jitter during training).
    """
    from ultralytics import YOLO

    if os.path.exists(cfg.yolo_cache_path):
        print(f"Loading existing YOLO cache from {cfg.yolo_cache_path}")
        with open(cfg.yolo_cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Pre-computing YOLO ROI cache for {len(video_paths)} videos...")
    yolo = YOLO(cfg.yolo_weights)
    yolo.to(cfg.yolo_device)

    cache = {}
    for path in tqdm(video_paths, desc="YOLO cache"):
        fps, total_frames, w, h = _video_meta(path)
        if fps <= 0 or total_frames <= 0:
            cache[path] = {"roi": (0, 0, w or 1, h or 1), "frame_w": w, "frame_h": h}
            continue

        indices = _sample_frame_indices(fps, total_frames, cfg.num_frames)

        cap = cv2.VideoCapture(path)
        all_bboxes = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                all_bboxes.append(None)
                continue

            results = yolo(
                frame,
                conf=cfg.yolo_conf,
                iou=cfg.yolo_iou,
                classes=[0],   # person only
                verbose=False,
            )
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                all_bboxes.append(None)
                continue

            # pick union of all person boxes in this frame
            xyxy = boxes.xyxy.cpu().numpy()
            fx1, fy1 = xyxy[:, 0].min(), xyxy[:, 1].min()
            fx2, fy2 = xyxy[:, 2].max(), xyxy[:, 3].max()
            all_bboxes.append((int(fx1), int(fy1), int(fx2), int(fy2)))

        cap.release()
        roi = _compute_union_roi(all_bboxes, w, h, cfg.yolo_padding)
        cache[path] = {"roi": roi, "frame_w": w, "frame_h": h}

    with open(cfg.yolo_cache_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"YOLO cache saved → {cfg.yolo_cache_path}")
    return cache


# ── Frame pre-extraction cache ────────────────────────────────────────────────

def precompute_frame_cache(cfg: Config, video_paths: List[str], yolo_cache: Dict) -> str:
    """
    Pre-extract YOLO-cropped, resized frames to disk as uint8 .npy files.
    Eliminates video seeking every epoch — __getitem__ becomes a fast np.load().

    Storage: 32 × 224 × 224 × 3 uint8 ≈ 4.6 MB/clip → ~9 GB for 1996 clips.
    One-time cost of ~15 min; every subsequent epoch loads in <1 s per worker.
    """
    cache_dir = cfg.frame_cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    pending = [p for p in video_paths
               if not os.path.exists(os.path.join(cache_dir, _cache_key(p) + ".npy"))]

    if not pending:
        print(f"Frame cache complete  ({len(video_paths)} clips)  →  {cache_dir}")
        return cache_dir

    gb = len(video_paths) * cfg.num_frames * cfg.img_size * cfg.img_size * 3 / 1e9
    print(f"Pre-extracting frames: {len(pending)} clips remaining  "
          f"(total cache ≈ {gb:.1f} GB)  →  {cache_dir}")

    for path in tqdm(pending, desc="Frame cache", unit="clip"):
        fps, total_frames, w, h = _video_meta(path)
        if fps <= 0 or total_frames <= 0:
            continue
        indices = _sample_frame_indices(fps, total_frames, cfg.num_frames)
        entry = yolo_cache.get(path)
        roi = entry["roi"] if entry is not None else (0, 0, w, h)
        frames = _read_frames(path, indices, cfg.img_size, roi=roi)
        if frames is not None:
            out = os.path.join(cache_dir, _cache_key(path) + ".npy")
            np.save(out, (frames * 255).astype(np.uint8))

    print(f"Frame cache saved  →  {cache_dir}")
    return cache_dir


# ── Transforms ───────────────────────────────────────────────────────────────

def _build_transforms(is_train: bool, cfg: Config) -> T.Compose:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if is_train:
        return T.Compose([
            T.RandomHorizontalFlip(p=cfg.hflip_p),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.Normalize(mean=mean, std=std),
        ])
    return T.Compose([T.Normalize(mean=mean, std=std)])


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
        yolo_cache: Dict,
        cfg: Config,
        is_train: bool = True,
        frame_cache_dir: Optional[str] = None,
    ):
        self.paths = video_paths
        self.labels = labels
        self.cache = yolo_cache
        self.cfg = cfg
        self.is_train = is_train
        self.transform = _build_transforms(is_train, cfg)

        # Build path → .npy file map for fast loading
        self._npy: Dict[str, str] = {}
        if frame_cache_dir is not None:
            for p in video_paths:
                npy = os.path.join(frame_cache_dir, _cache_key(p) + ".npy")
                if os.path.exists(npy):
                    self._npy[p] = npy
        cached = len(self._npy)
        if frame_cache_dir is not None:
            print(f"  {'Train' if is_train else 'Val  '} dataset: "
                  f"{cached}/{len(video_paths)} clips served from frame cache")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        label = self.labels[idx]

        if path in self._npy:
            # Fast path: pre-extracted uint8 array  (T, H, W, C)
            frames = np.load(self._npy[path]).astype(np.float32) / 255.0
        else:
            # Slow path: seek video + YOLO crop inline
            fps, total_frames, w, h = _video_meta(path)
            cache_entry = self.cache.get(path)
            roi = cache_entry["roi"] if cache_entry is not None else (0, 0, w, h)
            jitter = self.is_train and self.cfg.temporal_jitter
            frame_indices = _sample_frame_indices(fps, total_frames, self.cfg.num_frames, jitter)
            frames = _read_frames(path, frame_indices, self.cfg.img_size, roi=roi)
            if frames is None:
                frames = np.zeros(
                    (self.cfg.num_frames, self.cfg.img_size, self.cfg.img_size, 3),
                    dtype=np.float32,
                )

        # (T, H, W, C) → (T, C, H, W)
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)

        # apply per-frame spatial transforms (hflip, color jitter, normalize)
        augmented = torch.stack([self.transform(tensor[t]) for t in range(tensor.shape[0])])

        return {
            "frames": augmented,
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
        }


# ── Data loading utilities ────────────────────────────────────────────────────

def collect_video_paths(cfg: Config) -> Tuple[List[str], List[int]]:
    """
    Walk data_root/{Violence,NonViolence} and return (paths, labels).
    Filters clips outside [min_clip_duration, max_clip_duration].
    """
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
    """
    Returns (train_loader, val_loader, yolo_cache).
    Builds the YOLO cache if it doesn't exist.
    """
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

    yolo_cache = precompute_yolo_cache(cfg, paths)

    frame_cache_dir = None
    if cfg.use_frame_cache:
        frame_cache_dir = precompute_frame_cache(cfg, paths, yolo_cache)

    train_ds = ViolenceDataset(train_p, train_l, yolo_cache, cfg,
                               is_train=True, frame_cache_dir=frame_cache_dir)
    val_ds   = ViolenceDataset(val_p,   val_l,   yolo_cache, cfg,
                               is_train=False, frame_cache_dir=frame_cache_dir)

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
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, yolo_cache
