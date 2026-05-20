"""
gRPC violence detection server.

All heavy work lives here: YOLO person detection, model inference, ring buffer,
EMA smoothing, and alert logic — identical to inference.py's StreamDetector.
The client (runner.py) is a thin display layer that sends JPEG frames and renders
the results it gets back.

Usage:
    python grpc_service/server.py --checkpoint checkpoints/best.pth
    python grpc_service/server.py --checkpoint checkpoints/best.pth --port 50051 --compile
"""

import argparse
import logging
import sys
import threading
from collections import deque
from concurrent import futures
import os

import cv2
import grpc
import numpy as np
import torch
import torchvision.transforms as T
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from torch.cuda.amp import autocast
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from model import ViolenceDetector
from utils import load_checkpoint

import violence_pb2
import violence_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# ── Frame helpers (same as inference.py) ──────────────────────────────────────

def _downscale(frame: np.ndarray, max_side: int) -> np.ndarray:
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
    arr = []
    for f in frames:
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
        t = _NORMALIZE(torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1))
        arr.append(t)
    tensor = torch.stack(arr).unsqueeze(0)
    return tensor.half() if fp16 else tensor


# ── Per-connection stream detector ────────────────────────────────────────────

class StreamDetector:
    """
    Mirrors inference.py's StreamDetector exactly.
    YOLO overlay data is always stored (not gated on show_yolo) so the client
    can decide whether to render it.
    """

    def __init__(self, model, yolo: YOLO, cfg: Config, device: torch.device):
        self.cfg    = cfg
        self.device = device
        self.fp16   = (device.type == "cuda")

        self.model = model
        self.yolo  = yolo

        self._buffer: deque = deque(maxlen=cfg.buffer_maxlen)
        self._buf_lock = threading.Lock()

        self.frame_count       = 0
        self.smoothed_prob     = 0.0
        self.consecutive_above = 0
        self._result           = (0.0, False)
        self._result_lock      = threading.Lock()

        self._last_boxes    = np.empty((0, 4), dtype=np.float32)
        self._last_confs    = np.empty(0,      dtype=np.float32)
        self._last_roi      = None
        self._last_small_wh = (1, 1)

        self._trigger = threading.Event()
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._infer_loop, daemon=True,
                                         name="InferThread")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._trigger.set()
        self._thread.join(timeout=5)

    def _detect_and_crop(self, frames: list) -> list:
        fh, fw = frames[0].shape[:2]
        n    = min(self.cfg.yolo_infer_frames, len(frames))
        step = max(1, len(frames) // n)
        keyframes = [frames[i * step] for i in range(n)]

        all_boxes, all_confs = [], []
        last_boxes = np.empty((0, 4), dtype=np.float32)
        last_confs = np.empty(0,      dtype=np.float32)
        for f in keyframes:
            res = self.yolo(f, conf=self.cfg.yolo_conf, iou=self.cfg.yolo_iou,
                            classes=[0], verbose=False)
            boxes = res[0].boxes
            if boxes is not None and len(boxes):
                b = boxes.xyxy.cpu().numpy()
                c = boxes.conf.cpu().numpy()
                all_boxes.append(b)
                all_confs.append(c)
                last_boxes, last_confs = b, c

        combined = np.concatenate(all_boxes) if all_boxes else np.empty((0, 4))
        roi = _union_roi(combined, fw, fh, self.cfg.yolo_padding)

        with self._result_lock:
            self._last_boxes    = last_boxes
            self._last_confs    = last_confs
            self._last_roi      = roi
            self._last_small_wh = (fw, fh)

        x1, y1, x2, y2 = roi
        return [f[y1:y2, x1:x2] for f in frames]

    @torch.no_grad()
    def _run_model(self, sampled: list) -> float:
        cropped = self._detect_and_crop(sampled)
        tensor  = _frames_to_tensor(cropped, self.cfg.img_size, self.fp16).to(self.device)
        with autocast():
            logit = self.model(tensor)
        return float(torch.sigmoid(logit.float()).item())

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
                prob    = self._run_model(sampled)
            except Exception as e:
                log.error("InferThread: %s", e)
                continue

            alpha = self.cfg.ema_alpha
            with self._result_lock:
                self.smoothed_prob = alpha * prob + (1 - alpha) * self.smoothed_prob
                if self.smoothed_prob >= self.cfg.alert_threshold:
                    self.consecutive_above += 1
                else:
                    self.consecutive_above = 0
                alert        = self.consecutive_above >= self.cfg.alert_consecutive
                self._result = (self.smoothed_prob, alert)

            if alert:
                log.warning("ALERT  score=%.4f", self.smoothed_prob)

    def process_frame(self, frame_bgr: np.ndarray) -> tuple:
        small = _downscale(frame_bgr, self.cfg.buffer_max_side)

        with self._buf_lock:
            self._buffer.append(small)

        self.frame_count += 1
        if (len(self._buffer) >= self.cfg.num_frames
                and self.frame_count % self.cfg.stride_frames == 0):
            self._trigger.set()

        with self._result_lock:
            return self._result

    def get_yolo_overlay(self) -> tuple:
        with self._result_lock:
            return (self._last_boxes.copy(), self._last_confs.copy(),
                    self._last_roi, self._last_small_wh)


# ── gRPC servicer ─────────────────────────────────────────────────────────────

class ViolenceDetectorServicer(violence_pb2_grpc.ViolenceDetectorServicer):

    def __init__(self, model, yolo: YOLO, cfg: Config, device: torch.device):
        self._model  = model
        self._yolo   = yolo
        self._cfg    = cfg
        self._device = device

    def DetectStream(self, request_iterator, context):
        detector = StreamDetector(self._model, self._yolo, self._cfg, self._device)
        detector.start()
        log.info("New stream connection from %s", context.peer())
        try:
            for req in request_iterator:
                if req.width > 0 and req.height > 0:
                    frame = np.frombuffer(req.frame, dtype=np.uint8).reshape(
                        req.height, req.width, 3)
                else:
                    buf   = np.frombuffer(req.frame, dtype=np.uint8)
                    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                prob, alert = detector.process_frame(frame)
                boxes, confs, roi, small_wh = detector.get_yolo_overlay()

                label = ("VIOLENCE DETECTED" if alert
                         else ("SUSPICIOUS" if prob > 0.45 else "NORMAL"))

                resp = violence_pb2.DetectResponse(
                    frame_idx   = req.frame_idx,
                    probability = prob,
                    alert       = alert,
                    label       = label,
                    small_w     = small_wh[0],
                    small_h     = small_wh[1],
                    has_roi     = roi is not None,
                )
                if roi is not None:
                    resp.roi_x1 = float(roi[0])
                    resp.roi_y1 = float(roi[1])
                    resp.roi_x2 = float(roi[2])
                    resp.roi_y2 = float(roi[3])
                if len(boxes):
                    resp.boxes_flat.extend(boxes.flatten().tolist())
                    resp.confs.extend(confs.tolist())

                yield resp
        finally:
            detector.stop()
            log.info("Stream connection closed: %s", context.peer())


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="./checkpoints/best.pth")
    p.add_argument("--port",       type=int, default=50051)
    p.add_argument("--workers",    type=int, default=4,
                   help="gRPC thread-pool workers")
    p.add_argument("--compile",    action="store_true",
                   help="torch.compile reduce-overhead (~30s warmup, ~20% faster)")
    args = p.parse_args()

    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Loading model from %s on %s", args.checkpoint, device)
    model = ViolenceDetector(cfg).to(device)
    load_checkpoint(model, args.checkpoint, device)
    fp16 = device.type == "cuda"
    model = model.half() if fp16 else model
    model.eval()

    if args.compile and hasattr(torch, "compile"):
        log.info("Compiling model (reduce-overhead)…")
        model = torch.compile(model, mode="reduce-overhead")

    log.info("Loading YOLO %s", cfg.yolo_weights)
    yolo = YOLO(cfg.yolo_weights)
    yolo.to(cfg.yolo_device)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.workers))
    violence_pb2_grpc.add_ViolenceDetectorServicer_to_server(
        ViolenceDetectorServicer(model, yolo, cfg, device), server
    )
    addr = f"[::]:{args.port}"
    server.add_insecure_port(addr)
    server.start()
    log.info("Server listening on %s", addr)

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        log.info("Shutting down.")
        server.stop(grace=2)


if __name__ == "__main__":
    main()
