"""
Production-optimized real-time violence detection on a CCTV stream (file or RTSP).

Why the original was slow
─────────────────────────
• YOLO ran on all 32 sampled frames per window → ~320 ms blocked per window
• Main thread did inference → display froze every 8 frames
• Full-resolution frames filled the ring buffer (~900 MB for 1080p × 150 frames)

Optimizations applied
─────────────────────
1. YOLO on cfg.yolo_infer_frames (default 5) keyframes → 6× fewer YOLO calls,
   union ROI still valid because people don't teleport within 5 seconds.
2. Dedicated inference thread — capture + display run at full camera FPS;
   process_frame() returns the last known result instantly without blocking.
3. fp16 model on CUDA → ~30% faster forward pass, half GPU memory.
4. Buffer frames downscaled to 640 px (YOLO's native resolution) → 6× less RAM.
5. Optional torch.compile(mode="reduce-overhead") for ~20% extra GPU throughput.

Usage
─────
    python inference.py --source path/to/video.mp4
    python inference.py --source rtsp://...
    python inference.py --source 0                          # webcam
    python inference.py --source video.mp4 --no-display --output out.mp4
    python inference.py --source video.mp4 --compile       # enable torch.compile
    python inference.py --source video.mp4 --show-yolo     # draw YOLO boxes + union ROI
"""

import argparse
import threading
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


_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# ── Frame helpers ─────────────────────────────────────────────────────────────

def _downscale(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longest edge ≤ max_side; no-op if already small enough."""
    if max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    scale = max_side / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_LINEAR)


def _sample_uniform(buf: list, n: int) -> list:
    total = len(buf)
    if total <= n:
        return buf[:]
    indices = [int(i * total / n + total / n / 2) for i in range(n)]
    return [buf[min(i, total - 1)] for i in indices]


def _union_roi(boxes_xyxy: np.ndarray, fw: int, fh: int, padding: float) -> tuple:
    if len(boxes_xyxy) == 0:
        return 0, 0, fw, fh
    x1, y1 = boxes_xyxy[:, 0].min(), boxes_xyxy[:, 1].min()
    x2, y2 = boxes_xyxy[:, 2].max(), boxes_xyxy[:, 3].max()
    pw, ph = (x2 - x1) * padding, (y2 - y1) * padding
    return (max(0, int(x1 - pw)), max(0, int(y1 - ph)),
            min(fw, int(x2 + pw)), min(fh, int(y2 + ph)))


def _frames_to_tensor(frames: list, size: int, fp16: bool) -> torch.Tensor:
    """BGR frame list → (1, T, C, H, W) tensor, normalised, optionally fp16."""
    arr = []
    for f in frames:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
        t = _NORMALIZE(torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1))
        arr.append(t)
    tensor = torch.stack(arr).unsqueeze(0)        # (1, T, C, H, W)
    return tensor.half() if fp16 else tensor


# ── Overlay ────────────────────────────────────────────────────────────────────

def _draw_overlay(frame: np.ndarray, prob: float, alert: bool,
                  fps: float, threshold: float) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_w, bar_h, pad = int(w * 0.35), 18, 10

    label = ("VIOLENCE DETECTED" if alert
             else ("SUSPICIOUS" if prob > 0.45 else "NORMAL"))
    color = ((0, 0, 220) if alert
             else ((0, 140, 255) if prob > 0.45 else (30, 180, 30)))

    cv2.rectangle(frame, (pad, pad), (pad + bar_w + 4, pad + bar_h + 4), (30, 30, 30), -1)
    fill = int(bar_w * prob)
    cv2.rectangle(frame, (pad + 2, pad + 2), (pad + 2 + fill, pad + 2 + bar_h), color, -1)
    cv2.putText(frame, f"{label}  {prob:.2f}",
                (pad + 4, pad + bar_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(frame, f"FPS:{fps:.1f}",
                (w - 90, pad + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if alert:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 4)
    return frame


# ── YOLO overlay ──────────────────────────────────────────────────────────────

def _draw_yolo_overlay(frame: np.ndarray, boxes: np.ndarray, confs: np.ndarray,
                       roi, small_wh: tuple) -> np.ndarray:
    """
    Draw person bounding boxes (green) and union ROI (cyan) on frame.
    Boxes are in small-buffer coordinate space; scaled to frame resolution here.
    """
    fh, fw = frame.shape[:2]
    sw, sh = small_wh
    sx = fw / sw if sw > 0 else 1.0
    sy = fh / sh if sh > 0 else 1.0

    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"person {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), (0, 220, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    if roi is not None:
        rx1, ry1, rx2, ry2 = (int(roi[0] * sx), int(roi[1] * sy),
                               int(roi[2] * sx), int(roi[3] * sy))
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 220, 0), 2)
        cv2.putText(frame, "ROI", (rx1 + 4, ry1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1)

    return frame


# ── Production detector ────────────────────────────────────────────────────────

class StreamDetector:
    """
    Two-thread design:
      Main thread  — feeds frames via process_frame(); returns last result immediately.
      Infer thread — waits on a trigger Event, snapshots buffer, runs YOLO + model.

    process_frame() never blocks on inference latency, so capture/display FPS is
    limited only by the camera, not the model.
    """

    def __init__(self, model: ViolenceDetector, cfg: Config, device: torch.device,
                 show_yolo: bool = False):
        self.cfg = cfg
        self.device = device
        self.fp16 = (device.type == "cuda")
        self.show_yolo = show_yolo

        self.model = model.half() if self.fp16 else model
        self.model.eval()

        self.yolo = YOLO(cfg.yolo_weights)
        self.yolo.to(cfg.yolo_device)

        # Ring buffer of downscaled BGR frames (thread-safe via _buf_lock)
        self._buffer: deque = deque(maxlen=cfg.buffer_maxlen)
        self._buf_lock = threading.Lock()

        self.frame_count = 0
        self.smoothed_prob = 0.0
        self.consecutive_above = 0
        self._result = (0.0, False)          # (prob, alert) — latest completed window
        self._result_lock = threading.Lock()

        # YOLO overlay state — written by infer thread, read by main thread
        self._last_boxes = np.empty((0, 4), dtype=np.float32)   # xyxy in small-frame coords
        self._last_confs = np.empty(0, dtype=np.float32)
        self._last_roi: tuple = None                              # (x1,y1,x2,y2)
        self._last_small_wh: tuple = (1, 1)                      # (w, h) of buffer frame

        self._trigger = threading.Event()    # set by main thread, cleared by infer thread
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._infer_loop, daemon=True,
                                        name="InferenceThread")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._trigger.set()          # unblock the waiting thread so it can exit
        self._thread.join(timeout=5)

    # ── YOLO on keyframes only ─────────────────────────────────────────────────

    def _detect_and_crop(self, frames: list) -> list:
        """
        Run YOLO on yolo_infer_frames evenly-spaced keyframes.
        Compute union ROI, then crop all frames to it.

        Reduces YOLO calls from 32 → 5 (6×) with negligible ROI quality loss
        because body position doesn't change significantly within 5 seconds.
        """
        fh, fw = frames[0].shape[:2]
        n = min(self.cfg.yolo_infer_frames, len(frames))
        step = max(1, len(frames) // n)
        keyframes = [frames[i * step] for i in range(n)]

        all_boxes, all_confs = [], []
        last_boxes = np.empty((0, 4), dtype=np.float32)
        last_confs = np.empty(0, dtype=np.float32)
        for f in keyframes:
            res = self.yolo(f, conf=self.cfg.yolo_conf, iou=self.cfg.yolo_iou,
                            classes=[0], verbose=False)
            boxes = res[0].boxes
            if boxes is not None and len(boxes):
                b = boxes.xyxy.cpu().numpy()
                c = boxes.conf.cpu().numpy()
                all_boxes.append(b)
                all_confs.append(c)
                last_boxes, last_confs = b, c  # keep only the most recent keyframe

        combined = np.concatenate(all_boxes) if all_boxes else np.empty((0, 4))
        roi = _union_roi(combined, fw, fh, self.cfg.yolo_padding)

        if self.show_yolo:
            with self._result_lock:
                self._last_boxes = last_boxes
                self._last_confs = last_confs
                self._last_roi = roi
                self._last_small_wh = (fw, fh)

        x1, y1, x2, y2 = roi
        return [f[y1:y2, x1:x2] for f in frames]

    # ── Model forward pass ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_model(self, sampled: list) -> float:
        cropped = self._detect_and_crop(sampled)
        tensor = _frames_to_tensor(cropped, self.cfg.img_size, self.fp16).to(self.device)
        with autocast():
            logit = self.model(tensor)
        return float(torch.sigmoid(logit.float()).item())

    # ── Background inference loop ──────────────────────────────────────────────

    def _infer_loop(self):
        while not self._stop.is_set():
            fired = self._trigger.wait(timeout=0.05)
            if not fired or self._stop.is_set():
                continue
            self._trigger.clear()

            with self._buf_lock:
                snapshot = list(self._buffer)

            if len(snapshot) < self.cfg.num_frames:
                continue

            try:
                sampled = _sample_uniform(snapshot, self.cfg.num_frames)
                prob = self._run_model(sampled)
            except Exception as e:
                print(f"[InferenceThread] {e}")
                continue

            alpha = self.cfg.ema_alpha
            with self._result_lock:
                self.smoothed_prob = alpha * prob + (1 - alpha) * self.smoothed_prob
                if self.smoothed_prob >= self.cfg.alert_threshold:
                    self.consecutive_above += 1
                else:
                    self.consecutive_above = 0
                alert = self.consecutive_above >= self.cfg.alert_consecutive
                self._result = (self.smoothed_prob, alert)

            if alert:
                ts = time.strftime("%H:%M:%S")
                print(f"\r[{ts}] *** VIOLENCE ALERT ***  score={self.smoothed_prob:.3f}",
                      flush=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_yolo_overlay(self):
        """Return (boxes_xyxy, confs, roi, (small_w, small_h)) from last YOLO run."""
        with self._result_lock:
            return (self._last_boxes.copy(), self._last_confs.copy(),
                    self._last_roi, self._last_small_wh)

    def process_frame(self, frame_bgr: np.ndarray):
        """
        Buffer the frame and trigger inference if stride reached.
        Returns (prob, alert) from the most recently completed window — never blocks.
        """
        small = _downscale(frame_bgr, self.cfg.buffer_max_side)

        with self._buf_lock:
            self._buffer.append(small)

        self.frame_count += 1
        if (len(self._buffer) >= self.cfg.num_frames
                and self.frame_count % self.cfg.stride_frames == 0):
            self._trigger.set()

        with self._result_lock:
            return self._result


# ── Entry point ────────────────────────────────────────────────────────────────

def run(source, checkpoint: str, cfg: Config, display: bool = True,
        output_path: str = None, show_yolo: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViolenceDetector(cfg).to(device)
    load_checkpoint(model, checkpoint, device)

    if cfg.compile_model and hasattr(torch, "compile"):
        print("Compiling model (torch.compile reduce-overhead) — first inference ~30 s…")
        model = torch.compile(model, mode="reduce-overhead")

    detector = StreamDetector(model, cfg, device, show_yolo=show_yolo)
    detector.start()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        detector.stop()
        raise RuntimeError(f"Cannot open source: {source}")

    stream_fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps_assumed
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, stream_fps, (frame_w, frame_h))

    mode_str = "fp16" if device.type == "cuda" else "fp32"
    print(f"Stream : {source}  |  {stream_fps:.1f} fps  |  {frame_w}×{frame_h}")
    print(f"Device : {device}  |  {mode_str}  |  "
          f"YOLO on {cfg.yolo_infer_frames} keyframes / {cfg.num_frames} sampled")
    print("Press 'q' to quit.")

    t_prev = time.time()
    display_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            prob, alert = detector.process_frame(frame)

            now = time.time()
            display_fps = 0.9 * display_fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            vis = _draw_overlay(frame.copy(), prob, alert, display_fps, cfg.alert_threshold)
            if show_yolo:
                boxes, confs, roi, small_wh = detector.get_yolo_overlay()
                vis = _draw_yolo_overlay(vis, boxes, confs, roi, small_wh)

            if writer:
                writer.write(vis)
            if display:
                cv2.imshow("Violence Detector", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        detector.stop()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Stream ended.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, required=True,
                   help="Video file, RTSP URL, or webcam index")
    p.add_argument("--checkpoint", type=str, default="./checkpoints/best.pth")
    p.add_argument("--output", type=str, default=None, help="Save annotated video here")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile (reduce-overhead); ~30s warmup, then faster")
    p.add_argument("--show-yolo", action="store_true",
                   help="Overlay YOLO person boxes (green) and union ROI (cyan) on display")
    args = p.parse_args()

    cfg = Config()
    if args.compile:
        cfg.compile_model = True

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    run(source=source, checkpoint=args.checkpoint,
        cfg=cfg, display=not args.no_display, output_path=args.output,
        show_yolo=args.show_yolo)
