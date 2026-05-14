from dataclasses import dataclass, field
import os


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    data_root: str = "/home/adil/Downloads/voilence_detection/Real Life Violence Dataset"
    checkpoint_dir: str = "./checkpoints"
    yolo_cache_path: str = "./yolo_roi_cache.pkl"
    frame_cache_dir: str = "./frame_cache"   # pre-extracted frames; ~9 GB for 1996 clips
    use_frame_cache: bool = True             # set False to skip and read videos live

    # ── Data / sampling ────────────────────────────────────────────────────
    classes: tuple = ("NonViolence", "Violence")   # 0 = NonViolence, 1 = Violence
    num_frames: int = 32           # frames sampled per clip
    img_size: int = 224            # spatial input to backbone
    train_ratio: float = 0.8
    max_clip_duration: float = 20.0  # filter outlier-long clips (375s outliers)
    min_clip_duration: float = 2.0

    # ── YOLO (person detection) ─────────────────────────────────────────────
    yolo_weights: str = "yolov8n.pt"
    yolo_conf: float = 0.20        # min confidence to accept a person detection
    yolo_iou: float = 0.45
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
    num_workers: int = 4

    # ── Augmentation ───────────────────────────────────────────────────────
    color_jitter_p: float = 0.5
    hflip_p: float = 0.5
    temporal_jitter: bool = True   # randomly sub-sample within the sampled window

    # ── Inference / streaming ──────────────────────────────────────────────
    buffer_seconds: float = 5.0    # ring buffer length in seconds
    fps_assumed: int = 30          # assumed stream FPS for buffer sizing
    stride_frames: int = 8         # run inference every N new frames (~0.27 s at 30 fps)
    ema_alpha: float = 0.35        # EMA smoothing weight (higher = more responsive)
    alert_threshold: float = 0.65
    alert_consecutive: int = 3     # consecutive windows above threshold → alert
    yolo_infer_frames: int = 5     # YOLO runs on this many keyframes (not all 32) → 6× faster
    buffer_max_side: int = 640     # downscale frames before buffering; 0 = keep native res
    compile_model: bool = False    # torch.compile reduce-overhead; adds ~30s warmup, ~20% faster

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @property
    def buffer_maxlen(self) -> int:
        return int(self.buffer_seconds * self.fps_assumed)
