from dataclasses import dataclass, field
import os


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    data_root: str = "/home/adil/Downloads/voilence_detection/Real Life Violence Dataset"
    checkpoint_dir: str = "./checkpoints"
    frame_cache_dir: str = "./frame_cache_raw"  # pre-extracted raw frames; ~9 GB for 1996 clips
    use_frame_cache: bool = True                # set False to skip and read videos live

    # ── Data / sampling ────────────────────────────────────────────────────
    classes: tuple = ("NonViolence", "Violence")   # 0 = NonViolence, 1 = Violence
    num_frames: int = 32           # frames sampled per clip
    img_size: int = 224            # spatial input to backbone
    train_ratio: float = 0.8
    max_clip_duration: float = 20.0  # filter outlier-long clips (375s outliers)
    min_clip_duration: float = 2.0

    # ── YOLO (person detection) ─────────────────────────────────────────────
    yolo_weights: str = "yolov8n.pt"
    yolo_conf: float = 0.25        # min confidence to accept a person detection
    yolo_iou: float = 0.60         # IoU threshold for NMS
    yolo_padding: float = 0.15     # fractional padding added around union ROI
    yolo_device: str = "cuda"

    # ── Backbone ───────────────────────────────────────────────────────────
    # EfficientNetV2-S pre-trained on ImageNet-21k then fine-tuned on IN-1k.
    # Richer semantic features than IN-1k only; fits 8 GB GPU at batch=4.
    backbone_name: str = "tf_efficientnetv2_s.in21k_ft_in1k"
    feature_dim: int = 1280        # output dim of EfficientNetV2-S pool layer
    unfreeze_epoch: int = 5        # epoch after which backbone blocks are unfrozen

    # ── Temporal Transformer ───────────────────────────────────────────────
    d_model: int = 256             # projection dim inside transformer
    nhead: int = 8
    num_tf_layers: int = 2
    tf_dropout: float = 0.1
    tf_ff_dim: int = 512           # feed-forward hidden dim inside each layer

    # ── Training ───────────────────────────────────────────────────────────
    seed: int = 42
    batch_size: int = 4            # (B, 32, 224, 224, 3) — safe on 8 GB with AMP
    num_epochs: int = 40
    lr: float = 3e-4
    backbone_lr_scale: float = 0.1  # backbone LR = lr * scale when unfrozen
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    warmup_epochs: int = 3
    gradient_clip: float = 1.0
    num_workers: int = 2

    # ── Augmentation ───────────────────────────────────────────────────────
    color_jitter_p: float = 0.5
    hflip_p: float = 0.5
    temporal_jitter: bool = True   # randomly sub-sample within the sampled window

    # ── Inference / streaming ──────────────────────────────────────────────
    # Training sampled 32 frames from the full clip (2–20 s). The ring buffer
    # must span a comparable duration so the model sees a similar temporal range.
    # 10 s at 30 fps = 300 frames; use ~10 s to sit in the middle of the
    # training distribution. RAM cost: ~300 × 640×360×3 ≈ 200 MB (uint8).
    buffer_seconds: float = 10.0   # ring buffer length in seconds
    fps_assumed: int = 30          # assumed stream FPS for buffer sizing
    stride_frames: int = 8         # run inference every N new frames (~0.27 s at 30 fps)
    ema_alpha: float = 0.45        # EMA smoothing weight — slightly more responsive than 0.35
    alert_threshold: float = 0.65
    alert_consecutive: int = 3     # consecutive windows above threshold → alert (was 6)
    yolo_infer_frames: int = 5     # YOLO runs on this many keyframes (not all 32) → 6× faster
    buffer_max_side: int = 640     # downscale frames before buffering; 0 = keep native res
    compile_model: bool = False    # torch.compile reduce-overhead; adds ~30s warmup, ~20% faster

    # ── Production training ────────────────────────────────────────────────────
    grad_accum_steps: int = 1      # effective batch = batch_size × grad_accum_steps
    early_stop_patience: int = 10  # epochs without val F1 improvement before halting
    val_batch_multiplier: int = 2  # val loader batch = batch_size × this (no backward pass)

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @property
    def buffer_maxlen(self) -> int:
        return int(self.buffer_seconds * self.fps_assumed)
