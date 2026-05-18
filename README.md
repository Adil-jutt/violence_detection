# Violence Detection

Real-time violence detection for CCTV streams using a two-stage pipeline: **YOLOv8** person detection followed by an **EfficientNetV2-S + Temporal Transformer** classifier.

## Architecture

```
Input frames
     │
     ▼
YOLOv8n ──► Union ROI crop (person bounding boxes)
     │
     ▼
EfficientNetV2-S  (SpatialEncoder, TimeDistributed)
     │  (B, T, 1280)
     ▼
Temporal Transformer  (2-layer, CLS token, learnable pos. enc.)
     │  (B, d_model=256)
     ▼
Classification head  →  Violence probability [0, 1]
```

**Why transformer over LSTM:** motion magnitude is not discriminative in this dataset (non-violent clips actually have more optical flow than violent ones). What matters is semantic context — who is near whom, contact patterns, posture sequences. Self-attention can identify which frame pairs are informative without forcing a fixed-order hidden state.

> All commands in one place: [`commands.txt`](commands.txt)

## Project Structure

```
.
├── config.py        # All hyperparameters and paths in one dataclass
├── dataset.py       # VideoDataset with YOLO ROI cache + frame cache
├── model.py         # SpatialEncoder + TemporalTransformer
├── train.py         # Training loop (warmup → progressive unfreeze → cosine LR)
├── evaluate.py      # Offline metrics: AUC, F1, confusion matrix, ROC curve
├── utils.py         # Checkpointing, metrics, training history plot
└── inference.py     # Production real-time stream detector (two-thread design)
```

### Not tracked by git (see `.gitignore`)

| Path | Size | Description |
|---|---|---|
| `frame_cache/` | ~9 GB | Pre-extracted YOLO-cropped uint8 `.npy` frames (one per clip). Built once by `dataset.py`, eliminates video seeking every epoch. |
| `yolo_roi_cache.pkl` | ~5 MB | YOLO bounding-box results cached per video path. Avoids re-running detection on the same clip. |
| `checkpoints/` | ~300 MB | `best.pth` and periodic `epoch{N}.pth` saves. |
| `eval_outputs/` | ~1 MB | PNG plots from `evaluate.py` (ROC, confusion matrix, score distribution). |
| `*.pt` | ~6 MB | YOLO weights auto-downloaded by ultralytics (`yolov8n.pt`). |
| `runs/` | variable | YOLO training artefacts (if fine-tuning YOLO). |
| `*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.mpeg` | — | Video files (dataset + test clips). |
| `*.mp3 *.wav *.ogg *.flac *.aac` | — | Audio files. |

## Requirements

```
torch >= 2.0
torchvision
timm
ultralytics        # YOLOv8
opencv-python
scikit-learn
tqdm
matplotlib
```

Install into a conda environment:

```bash
conda create -n violence python=3.10 -y
conda activate violence
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm ultralytics opencv-python scikit-learn tqdm matplotlib
```

## Dataset

Expected layout under `config.py → data_root`:

```
Real Life Violence Dataset/
├── Violence/
│   ├── V_001.mp4
│   └── ...
└── NonViolence/
    ├── NV_001.mp4
    └── ...
```

The dataset used during development is the [Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset) (~1996 clips).

## Training

```bash
# Default run (reads config.py defaults)
python train.py

# Override hyperparameters
python train.py --lr 1e-4 --batch_size 8 --num_epochs 50

# Gradient accumulation (effective batch = batch_size × steps)
python train.py --grad_accum_steps 4

# Resume after a crash or early stop
python train.py --resume checkpoints/best.pth

# Override early stopping patience
python train.py --early_stop_patience 15
```

### Training strategy

| Phase | Epochs | What trains |
|---|---|---|
| Warmup | 0 → `warmup_epochs` (3) | Temporal head only; LR ramps 0 → `cfg.lr` |
| Frozen backbone | 3 → `unfreeze_epoch` (5) | Head at full LR |
| Progressive unfreeze | 5 → end | Top-3 backbone blocks + head; backbone LR = `lr × 0.1`; cosine annealing |

The frame pre-extraction cache (`frame_cache/`) is built automatically on the first run and takes ~15 minutes for 1996 clips. Subsequent epochs read `np.load()` (~2 ms/clip) instead of seeking video (~100 ms/clip).

Key config values (`config.py`):

| Parameter | Default | Description |
|---|---|---|
| `num_frames` | 32 | Frames sampled per clip window |
| `img_size` | 224 | Spatial input size to EfficientNetV2-S |
| `batch_size` | 4 | Safe on 8 GB VRAM with AMP |
| `backbone_name` | `tf_efficientnetv2_s.in21k_ft_in1k` | ImageNet-21k pretrained |
| `label_smoothing` | 0.1 | Reduces overconfidence on small dataset |
| `ema_alpha` | 0.35 | EMA weight for smoothing inference scores |
| `alert_threshold` | 0.65 | Violence probability threshold |

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best.pth
```

Outputs:
- Accuracy, F1, AUC-ROC, Average Precision (console)
- `eval_outputs/confusion_matrix.png`
- `eval_outputs/roc_curve.png`
- `eval_outputs/score_dist.png`
- Top-10 false positives and false negatives (console)

Threshold is selected automatically via Youden's J statistic.

## Inference

```bash
# Video file
python inference.py --source path/to/video.mp4

# Webcam (index 0)
python inference.py --source 0

# RTSP stream
python inference.py --source rtsp://camera-ip/stream

# Headless — no display window, write annotated output
python inference.py --source video.mp4 --no-display --output annotated.mp4

# Show YOLO person detection boxes on overlay
python inference.py --source video.mp4 --show-yolo

# Enable torch.compile (~30s warmup, ~20% faster throughput after)
python inference.py --source video.mp4 --compile

# Custom checkpoint
python inference.py --source video.mp4 --checkpoint checkpoints/epoch30.pth
```

### Production design (two-thread)

```
Main thread                    Inference thread
────────────                   ────────────────
cap.read()                     wait(trigger Event)
  │                              │
process_frame(frame)  ──set──►  snapshot buffer
  │  (never blocks)             YOLO on 5 keyframes
  │                             model forward (fp16)
return last result ◄──update── update smoothed prob + alert
```

The main thread runs at full camera FPS. Inference latency does not affect display.

### Inference optimizations

| Optimization | Speedup | Detail |
|---|---|---|
| YOLO on 5 keyframes (not 32) | ~6× fewer YOLO calls | Union ROI still valid; people don't teleport in 5 s |
| Dedicated inference thread | Display never blocks | `process_frame()` returns last result instantly |
| fp16 on CUDA | ~30% faster, 50% less GPU memory | `model.half()`; `logit.float()` before sigmoid |
| Buffer downscaled to 640 px | ~6× less RAM | 640 is YOLO's native resolution |
| `torch.compile(reduce-overhead)` | ~20% throughput | Optional; 30s warmup; enable with `--compile` |

### Overlay legend

| Label | Condition |
|---|---|
| `NORMAL` | score < 0.45 |
| `SUSPICIOUS` | 0.45 ≤ score < `alert_threshold` |
| `VIOLENCE DETECTED` | score ≥ `alert_threshold` for `alert_consecutive` (3) consecutive windows; red border |

## Key config for inference (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `buffer_seconds` | 5.0 | Ring buffer duration |
| `stride_frames` | 8 | Trigger inference every N frames (~0.27 s at 30 fps) |
| `yolo_infer_frames` | 5 | YOLO keyframe count per window |
| `buffer_max_side` | 640 | Downscale long edge before buffering (0 = native) |
| `alert_threshold` | 0.65 | Score above which window counts as violent |
| `alert_consecutive` | 3 | Windows above threshold needed to fire alert |
| `compile_model` | False | Enable `torch.compile` |

## License

For research and educational purposes only. Do not deploy without appropriate legal and ethical review.
