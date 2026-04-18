#!/usr/bin/env python3
"""Demo visual Lazarillo — stream MJPEG con detección YOLO en background thread."""
import os
import sys
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse

sys.path.insert(0, os.path.dirname(__file__))
from core.detector import build_detector, PRIORITY_CLASSES, Detection

RKNN_MODEL = os.path.join(os.path.dirname(__file__), "models/rknn/yolov8n_std.rknn")
ONNX_MODEL = os.path.join(os.path.dirname(__file__), "models/rknn/yolov8n.onnx")

# Solo clases relevantes para navegación — reduce falsos positivos irrelevantes
DISPLAY_CLASSES = frozenset(PRIORITY_CLASSES.keys())

CONFIDENCE_THRESHOLD = 0.50  # 0.50 filtra person/sheep confusion


def _load_detector():
    if os.path.isfile(RKNN_MODEL) and os.path.getsize(RKNN_MODEL) > 100_000:
        print(f"[Lazarillo] Backend: RKNNDetector ({RKNN_MODEL})")
        return build_detector(dev_mode=False, model_path=RKNN_MODEL,
                              confidence_threshold=CONFIDENCE_THRESHOLD)
    if os.path.isfile(ONNX_MODEL) and os.path.getsize(ONNX_MODEL) > 100_000:
        print(f"[Lazarillo] Backend: ONNXDetector ({ONNX_MODEL})")
        return build_detector(dev_mode=False, model_path=ONNX_MODEL,
                              confidence_threshold=CONFIDENCE_THRESHOLD)
    print("[Lazarillo] Backend: MockDetector")
    return build_detector(dev_mode=True)


class CameraDetector:
    """Captura webcam y corre detección en background. El stream lee sin bloquear."""

    def __init__(self) -> None:
        self._detector = _load_detector()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._detections: list[Detection] = []
        self._det_fps: float = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️  No se pudo abrir webcam")
            return

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            t0 = time.time()
            try:
                dets = self._detector.detect(frame)
            except Exception as e:
                print(f"[detect error] {e}")
                dets = []
            elapsed = time.time() - t0

            # Solo clases prioritarias para reducir ruido visual
            dets = [d for d in dets if d.class_id in DISPLAY_CLASSES]

            with self._lock:
                self._frame = frame
                self._detections = dets
                self._det_fps = 1.0 / elapsed if elapsed > 0 else 0

        cap.release()

    def get_annotated_jpeg(self) -> bytes | None:
        with self._lock:
            if self._frame is None:
                return None
            frame = self._frame.copy()
            dets = self._detections
            fps = self._det_fps

        for d in dets:
            x1, y1, x2, y2 = d.bbox
            color = (0, 220, 0) if d.class_id in PRIORITY_CLASSES else (220, 220, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{d.class_name} {d.confidence:.0%} {d.direction}"
            cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        status = f"Lazarillo | {len(dets)} obj | {fps:.1f} FPS det"
        cv2.putText(frame, status, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return jpeg.tobytes()

    def stop(self) -> None:
        self._running = False


cam = CameraDetector()
app = FastAPI()


def _gen_stream():
    while True:
        jpeg = cam.get_annotated_jpeg()
        if jpeg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1 / 30)  # ~30 FPS stream, detección no bloquea


@app.get("/", response_class=HTMLResponse)
def index():
    return """<html><body style="background:#111;margin:0">
<img src='/video' style='width:100%;max-width:800px;display:block;margin:auto'>
</body></html>"""


@app.get("/video")
def video():
    return StreamingResponse(_gen_stream(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
