"""
Real-time violence detection on a CCTV stream (file or RTSP URL).

Buffer strategy
───────────────
Ring buffer (deque maxlen = buffer_seconds × fps_assumed, default 150 frames).

Every `stride_frames` new frames the buffer triggers inference:
  1. Uniformly sample `num_frames` (32) from the buffer  → covers last ~5 s.
  2. Run YOLOv8n on each sampled frame → union person ROI.
  3. Crop + resize → EfficientNetV2-S + TemporalTransformer → raw logit.
  4. Apply EMA: smoothed = α × new + (1-α) × prev_smoothed.
  5. Alert if smoothed ≥ threshold for ≥ alert_consecutive windows.

Why this buffer design:
  • 5-s window matches the dataset clip duration (p10=4 s, p90=5.6 s).
  • 8-frame stride (~0.27 s at 30 fps) gives near-instant response.
  • EMA suppresses single-frame noise without adding noticeable lag.
  • Re-sampling YOLO per window means a person entering mid-window is included.

Usage
─────
    # on a video file
    conda run -n contrastive310 python inference.py --source path/to/video.mp4

    # on RTSP stream
    conda run -n contrastive310 python inference.py --source rtsp://...

    # headless (no display)
    conda run -n contrastive310 python inference.py --source ... --no-display
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from torch.cuda.amp import autocast  # type: ignore[import]
from ultralytics import YOLO

from config import Config
from model import ViolenceDetector
from utils import load_checkpoint


# ── Normalization (must match training) ───────────────────────────────────────
_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _frames_to_tensor(frames: list, size: int) -> torch.Tensor:
    """Convert list of (H,W,3) uint8 BGR frames → (1,T,C,H,W) float32 tensor."""
    arr = []
    for f in frames:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1)
        t = _NORMALIZE(t)
        arr.append(t)
    return torch.stack(arr).unsqueeze(0)  # (1, T, C, H, W)


def _union_roi_with_padding(boxes_xyxy: np.ndarray, frame_w: int, frame_h: int,
                             padding: float) -> tuple:
    if len(boxes_xyxy) == 0:
        return 0, 0, frame_w, frame_h
    x1 = boxes_xyxy[:, 0].min()
    y1 = boxes_xyxy[:, 1].min()
    x2 = boxes_xyxy[:, 2].max()
    y2 = boxes_xyxy[:, 3].max()
    pw, ph = (x2 - x1) * padding, (y2 - y1) * padding
    x1 = max(0, int(x1 - pw));  y1 = max(0, int(y1 - ph))
    x2 = min(frame_w, int(x2 + pw));  y2 = min(frame_h, int(y2 + ph))
    return x1, y1, x2, y2


def _sample_uniform(buf: deque, n: int) -> list:
    """Uniformly sample `n` frames from a deque."""
    total = len(buf)
    if total <= n:
        return list(buf)
    indices = [int(i * total / n + total / n / 2) for i in range(n)]
    buf_list = list(buf)
    return [buf_list[min(i, total - 1)] for i in indices]


# ── Overlay helpers ────────────────────────────────────────────────────────────

def _draw_overlay(frame: np.ndarray, prob: float, alert: bool,
                  fps: float, threshold: float) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_w = int(w * 0.35)
    bar_h = 18
    pad = 10

    label = "VIOLENCE DETECTED" if alert else ("SUSPICIOUS" if prob > 0.45 else "NORMAL")
    color = (0, 0, 220) if alert else ((0, 140, 255) if prob > 0.45 else (30, 180, 30))

    # background bar
    cv2.rectangle(frame, (pad, pad), (pad + bar_w + 4, pad + bar_h + 4), (30, 30, 30), -1)
    # fill bar
    fill = int(bar_w * prob)
    cv2.rectangle(frame, (pad + 2, pad + 2), (pad + 2 + fill, pad + 2 + bar_h), color, -1)

    cv2.putText(frame, f"{label}  {prob:.2f}",
                (pad + 4, pad + bar_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(frame, f"FPS:{fps:.1f}",
                (w - 90, pad + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if alert:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 4)

    return frame


# ── Main inference class ───────────────────────────────────────────────────────

class StreamDetector:
    """
    Stateful detector for a continuous video stream.

    Call `process_frame(frame_bgr)` for each incoming frame.
    Returns (prob, alert) after every prediction window;
    returns (None, False) during buffer fill and between strides.
    """

    def __init__(self, model: ViolenceDetector, cfg: Config, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device

        self.yolo = YOLO(cfg.yolo_weights)
        self.yolo.to(cfg.yolo_device)

        self.buffer: deque = deque(maxlen=cfg.buffer_maxlen)
        self.frame_count = 0
        self.smoothed_prob = 0.0
        self.consecutive_above = 0
        self.last_prob: float = 0.0

    def _detect_and_crop(self, frames: list) -> list:
        """Run YOLO on each frame, crop to union ROI, return list of cropped frames."""
        all_boxes = []
        first_frame = frames[0]
        fh, fw = first_frame.shape[:2]

        for f in frames:
            results = self.yolo(f, conf=self.cfg.yolo_conf, iou=self.cfg.yolo_iou,
                                classes=[0], verbose=False)
            boxes = results[0].boxes
            if boxes is not None and len(boxes):
                all_boxes.append(boxes.xyxy.cpu().numpy())

        if all_boxes:
            combined = np.concatenate(all_boxes, axis=0)
        else:
            combined = np.empty((0, 4))

        x1, y1, x2, y2 = _union_roi_with_padding(combined, fw, fh, self.cfg.yolo_padding)
        return [f[y1:y2, x1:x2] for f in frames]

    @torch.no_grad()
    def _infer(self, sampled: list) -> float:
        cropped = self._detect_and_crop(sampled)
        tensor = _frames_to_tensor(cropped, self.cfg.img_size).to(self.device)
        with autocast():
            logit = self.model(tensor)
        return float(torch.sigmoid(logit).item())

    def process_frame(self, frame_bgr: np.ndarray):
        """
        Returns (smoothed_prob, alert) when inference runs, else (None, False).
        """
        self.buffer.append(frame_bgr.copy())
        self.frame_count += 1

        if len(self.buffer) < self.cfg.num_frames:
            return None, False

        if self.frame_count % self.cfg.stride_frames != 0:
            return None, False

        sampled = _sample_uniform(self.buffer, self.cfg.num_frames)
        prob = self._infer(sampled)

        alpha = self.cfg.ema_alpha
        self.smoothed_prob = alpha * prob + (1 - alpha) * self.smoothed_prob
        self.last_prob = self.smoothed_prob

        if self.smoothed_prob >= self.cfg.alert_threshold:
            self.consecutive_above += 1
        else:
            self.consecutive_above = 0

        alert = self.consecutive_above >= self.cfg.alert_consecutive
        return self.smoothed_prob, alert


# ── Entry point ────────────────────────────────────────────────────────────────

def run(source, checkpoint: str, cfg: Config, display: bool = True,
        output_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViolenceDetector(cfg).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    detector = StreamDetector(model, cfg, device)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    stream_fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps_assumed
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, stream_fps, (frame_w, frame_h))

    print(f"Stream: {source}  |  {stream_fps:.1f} fps  |  {frame_w}×{frame_h}")
    print("Press 'q' to quit.")

    t_prev = time.time()
    display_fps = 0.0
    last_prob = 0.0
    last_alert = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prob, alert = detector.process_frame(frame)
        if prob is not None:
            last_prob = prob
            last_alert = alert
            if alert:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] *** VIOLENCE ALERT ***  score={prob:.3f}")

        # FPS counter
        now = time.time()
        display_fps = 0.9 * display_fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        vis = _draw_overlay(frame.copy(), last_prob, last_alert, display_fps,
                            cfg.alert_threshold)

        if writer:
            writer.write(vis)
        if display:
            cv2.imshow("Violence Detector", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Stream ended.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, required=True,
                   help="Video file path or RTSP URL (e.g. rtsp://...)")
    p.add_argument("--checkpoint", type=str, default="./checkpoints/best.pth")
    p.add_argument("--output", type=str, default=None, help="Save annotated video here")
    p.add_argument("--no-display", action="store_true")
    args = p.parse_args()

    cfg = Config()

    # Try to convert source to int for webcam index
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    run(
        source=source,
        checkpoint=args.checkpoint,
        cfg=cfg,
        display=not args.no_display,
        output_path=args.output,
    )
