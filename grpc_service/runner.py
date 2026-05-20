"""
gRPC violence detection runner (client).

Sends frames to the gRPC server paced at the video's native FPS and renders
results with the exact same overlay as inference.py. After the stream ends,
waits for all in-flight server results then plots the probability timeline.

Usage:
    python grpc_service/runner.py --source path/to/video.mp4
    python grpc_service/runner.py --source 0                        # webcam
    python grpc_service/runner.py --source rtsp://cam/stream
    python grpc_service/runner.py --source video.mp4 --no-display --output out.mp4
    python grpc_service/runner.py --source video.mp4 --show-yolo
    python grpc_service/runner.py --source video.mp4 --plot-out scores.png
    python grpc_service/runner.py --server localhost:50051 --source video.mp4
"""

import argparse
import os
import queue
import sys
import threading
import time

import cv2
import grpc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import violence_pb2
import violence_pb2_grpc

QUEUE_MAX = 256  # large enough to absorb bursts without dropping frames


# ── Overlay (identical to inference.py) ───────────────────────────────────────

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


def _draw_yolo_overlay(frame: np.ndarray, boxes: np.ndarray, confs: np.ndarray,
                       roi, small_wh: tuple) -> np.ndarray:
    fh, fw = frame.shape[:2]
    sw, sh = small_wh
    sx = fw / sw if sw > 0 else 1.0
    sy = fh / sh if sh > 0 else 1.0

    for box, conf in zip(boxes, confs):
        x1, y1 = int(box[0] * sx), int(box[1] * sy)
        x2, y2 = int(box[2] * sx), int(box[3] * sy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"person {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), (0, 220, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    if roi is not None:
        rx1, ry1 = int(roi[0] * sx), int(roi[1] * sy)
        rx2, ry2 = int(roi[2] * sx), int(roi[3] * sy)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 220, 0), 2)
        cv2.putText(frame, "ROI", (rx1 + 4, ry1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1)

    return frame


# ── gRPC detector ─────────────────────────────────────────────────────────────

class GrpcDetector:
    """
    Non-blocking gRPC client — same contract as inference.py's StreamDetector.
    process_frame() queues a frame and returns the latest known result immediately.
    Results are updated asynchronously as the server replies.
    """

    def __init__(self, server_addr: str):
        self._send_q    = queue.Queue(maxsize=QUEUE_MAX)
        self._result    = (0.0, False)
        self._boxes     = np.empty((0, 4), dtype=np.float32)
        self._confs     = np.empty(0,      dtype=np.float32)
        self._roi       = None
        self._small_wh  = (1, 1)
        self._lock      = threading.Lock()
        self._stop      = threading.Event()
        self._frame_idx = 0
        self._history   = []  # list of (frame_idx, prob, alert)

        self._channel = grpc.insecure_channel(
            server_addr,
            options=[
                ("grpc.max_send_message_length",    -1),
                ("grpc.max_receive_message_length", -1),
            ],
        )
        self._stub   = violence_pb2_grpc.ViolenceDetectorStub(self._channel)
        self._thread = threading.Thread(target=self._stream_loop, daemon=True,
                                        name="GrpcStreamThread")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self._channel.close()

    def drain_and_stop(self, idle_timeout: float = 3.0, hard_limit: float = 30.0):
        """
        After capture ends: wait for the send queue to empty (frames still being
        transmitted), then wait until results stop arriving for idle_timeout seconds.
        This ensures we collect all inference windows before plotting.
        """
        deadline = time.time() + hard_limit

        # Wait for all queued frames to be consumed by _frame_gen and sent
        while not self._send_q.empty() and time.time() < deadline:
            time.sleep(0.05)

        # Wait until the server stops sending new results
        with self._lock:
            prev_len = len(self._history)
        idle_since = time.time()
        while time.time() < deadline:
            time.sleep(0.2)
            with self._lock:
                cur_len = len(self._history)
            if cur_len != prev_len:
                prev_len = cur_len
                idle_since = time.time()
            elif time.time() - idle_since >= idle_timeout:
                break

        self.stop()

    # ── generator consumed by gRPC framework ──────────────────────────────────

    def _frame_gen(self):
        while not self._stop.is_set():
            try:
                req = self._send_q.get(timeout=0.1)
                yield req
            except queue.Empty:
                continue

    # ── background thread: bidirectional stream loop ──────────────────────────

    def _stream_loop(self):
        try:
            for resp in self._stub.DetectStream(self._frame_gen()):
                boxes_flat = np.array(resp.boxes_flat, dtype=np.float32)
                boxes = (boxes_flat.reshape(-1, 4) if len(boxes_flat)
                         else np.empty((0, 4), dtype=np.float32))
                confs = np.array(resp.confs, dtype=np.float32)
                roi   = ((resp.roi_x1, resp.roi_y1, resp.roi_x2, resp.roi_y2)
                         if resp.has_roi else None)

                with self._lock:
                    self._result    = (resp.probability, resp.alert)
                    self._boxes     = boxes
                    self._confs     = confs
                    self._roi       = roi
                    self._small_wh  = (resp.small_w, resp.small_h)
                    self._history.append((resp.frame_idx, resp.probability, resp.alert))

                if resp.alert:
                    ts = time.strftime("%H:%M:%S")
                    print(f"\r[{ts}] *** VIOLENCE ALERT ***  score={resp.probability:.3f}",
                          flush=True)

                if self._stop.is_set():
                    break
        except grpc.RpcError as e:
            if not self._stop.is_set():
                print(f"[gRPC] {e.code()}: {e.details()}")
                print("Is the server running?  "
                      "python grpc_service/server.py --checkpoint checkpoints/best.pth")

    # ── public API ─────────────────────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray, max_side: int = 640) -> tuple:
        """Downscale, enqueue as raw BGR bytes. Returns latest result without blocking."""
        h, w = frame_bgr.shape[:2]
        if max_side > 0 and max(h, w) > max_side:
            scale     = max_side / max(h, w)
            frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_LINEAR)

        h, w = frame_bgr.shape[:2]
        req = violence_pb2.FrameRequest(
            frame=frame_bgr.tobytes(),
            frame_idx=self._frame_idx,
            width=w,
            height=h,
        )
        self._frame_idx += 1

        try:
            self._send_q.put_nowait(req)
        except queue.Full:
            pass  # drop frame rather than stall display

        with self._lock:
            return self._result

    def get_yolo_overlay(self) -> tuple:
        with self._lock:
            return (self._boxes.copy(), self._confs.copy(),
                    self._roi, self._small_wh)

    def get_history(self) -> list:
        with self._lock:
            return list(self._history)


# ── Result plot ───────────────────────────────────────────────────────────────

def _plot_results(history: list, threshold: float, source_name: str,
                  out_path) -> None:
    if not history:
        print("No inference results to plot.")
        return

    idxs   = [h[0] for h in history]
    probs  = [h[1] for h in history]
    alerts = [h[2] for h in history]

    if out_path is None:
        matplotlib.use("TkAgg")

    fig, ax = plt.subplots(figsize=(12, 4))

    # Shade alert regions
    in_alert  = False
    start_idx = 0
    first_alert_span = True
    for idx, alert in zip(idxs, alerts):
        if alert and not in_alert:
            start_idx = idx
            in_alert  = True
        elif not alert and in_alert:
            label = "alert" if first_alert_span else "_"
            ax.axvspan(start_idx, idx, color="red", alpha=0.15, label=label)
            in_alert = False
            first_alert_span = False
    if in_alert:
        ax.axvspan(start_idx, idxs[-1], color="red", alpha=0.15,
                   label="alert" if first_alert_span else "_")

    ax.plot(idxs, probs, linewidth=1.5, color="#2196F3", label="violence prob (EMA)")
    ax.axhline(threshold, color="red",    linestyle="--", linewidth=1.2,
               label=f"alert threshold ({threshold})")
    ax.axhline(0.45,      color="orange", linestyle="--", linewidth=1.0,
               label="suspicious (0.45)")

    ax.set_ylim(0, 1)
    ax.set_xlabel("Frame index (server-side)")
    ax.set_ylabel("Violence probability")
    ax.set_title(f"Violence Score — {os.path.basename(str(source_name))}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Plot saved → {out_path}")
    else:
        plt.show()

    plt.close(fig)


# ── Main runner ───────────────────────────────────────────────────────────────

def run(source, server: str, display: bool, output_path,
        show_yolo: bool, threshold: float, plot_out) -> None:

    detector = GrpcDetector(server)
    detector.start()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        detector.stop()
        raise RuntimeError(f"Cannot open source: {source}")

    stream_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Target inter-frame interval — paces file sources to their native FPS so the
    # gRPC round-trip has time to complete before the video is fully read.
    # For live sources (webcam/RTSP) cap.read() naturally throttles, so sleep = 0.
    frame_interval = 1.0 / stream_fps

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, stream_fps, (frame_w, frame_h))

    print(f"Stream : {source}  |  {stream_fps:.1f} fps  |  {frame_w}×{frame_h}")
    print(f"Server : {server}")
    if display:
        print("Press 'q' to quit.")

    t_prev      = time.time()
    t_frame     = time.time()
    display_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            prob, alert = detector.process_frame(frame)

            now         = time.time()
            display_fps = 0.9 * display_fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev      = now

            vis = _draw_overlay(frame.copy(), prob, alert, display_fps, threshold)
            if show_yolo:
                boxes, confs, roi, small_wh = detector.get_yolo_overlay()
                vis = _draw_yolo_overlay(vis, boxes, confs, roi, small_wh)

            if writer:
                writer.write(vis)
            if display:
                cv2.imshow("Violence Detector [gRPC]", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Pace to native FPS so the server has time to process frames.
            # For live sources, cap.read() already spent the time; sleep = 0.
            elapsed = time.time() - t_frame
            wait    = frame_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            t_frame = time.time()

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    # Video done — wait for the server to process remaining buffered frames
    # and send back all results before we plot.
    print("Capture done. Waiting for server to finish remaining windows…")
    detector.drain_and_stop()
    print("Stream ended.")

    history = detector.get_history()
    _plot_results(history, threshold, source, plot_out)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="gRPC violence detection runner")
    p.add_argument("--source",     required=True,
                   help="Video file path, RTSP URL, or webcam index")
    p.add_argument("--server",     default="localhost:50051")
    p.add_argument("--output",     default=None,  help="Save annotated video here")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--show-yolo",  action="store_true",
                   help="Draw YOLO person boxes (green) and union ROI (cyan)")
    p.add_argument("--threshold",  type=float, default=0.65,
                   help="Alert threshold shown on the plot (default: 0.65)")
    p.add_argument("--plot-out",   default=None,
                   help="Save probability plot to this path. "
                        "Omit to display interactively after the stream ends.")
    args = p.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    run(
        source      = source,
        server      = args.server,
        display     = not args.no_display,
        output_path = args.output,
        show_yolo   = args.show_yolo,
        threshold   = args.threshold,
        plot_out    = args.plot_out,
    )
