#!/usr/bin/env python3
"""Servidor MJPEG mock para desarrollo — simula la Pi Zero W Camera."""
import io
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

PORT = 8080
BOUNDARY = b"--frame"

# Frame compartido entre threads
_latest_frame: bytes = b""
_frame_lock = threading.Lock()


def _generate_frames() -> None:
    """Genera frames sintéticos con objetos simulados."""
    global _latest_frame
    cap = None

    # Intentar abrir video de prueba si existe
    try:
        cap = cv2.VideoCapture("/fixtures/sample.mp4")
        if not cap.isOpened():
            cap = None
    except Exception:
        cap = None

    frame_count = 0
    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = cv2.resize(frame, (640, 480))
        else:
            # Frame sintético: fondo degradado + texto + rectángulo simulando bbox
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            t = frame_count / 30.0
            frame[:, :, 0] = int(abs(float(np.sin(t * 0.3))) * 60)
            frame[:, :, 1] = int(abs(float(np.sin(t * 0.2))) * 40)
            frame[:, :, 2] = int(abs(float(np.sin(t * 0.1))) * 80 + 30)
            frame = np.ascontiguousarray(frame)  # requerido por OpenCV 4.13 + numpy 2.x

            # Simular persona moviéndose
            x = int(200 + float(np.sin(t * 0.5)) * 150)
            cv2.rectangle(frame, (x, 100), (x + 80, 300), (0, 255, 0), 2)
            cv2.putText(frame, "person 0.92", (x, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Simular auto
            ax = int(400 + float(np.cos(t * 0.3)) * 100)
            cv2.rectangle(frame, (ax, 300), (ax + 120, 420), (0, 0, 255), 2)
            cv2.putText(frame, "car 0.87", (ax, 295),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Info del frame
            cv2.putText(frame, f"LAZARILLO MOCK STREAM | frame {frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "640x480 @ 10fps",
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with _frame_lock:
            _latest_frame = jpeg.tobytes()

        frame_count += 1
        time.sleep(0.1)  # 10 fps


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass  # silenciar logs HTTP

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return

        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        last_frame = None
        try:
            while True:
                with _frame_lock:
                    frame = _latest_frame
                if frame and frame is not last_frame:
                    last_frame = frame
                    self.wfile.write(
                        BOUNDARY + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
                        frame + b"\r\n"
                    )
                else:
                    time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    t = threading.Thread(target=_generate_frames, daemon=True)
    t.start()

    # Esperar primer frame
    while not _latest_frame:
        time.sleep(0.1)

    server = HTTPServer(("0.0.0.0", PORT), MJPEGHandler)
    print(f"[mock-stream] http://0.0.0.0:{PORT}/stream.mjpg", flush=True)
    server.serve_forever()
