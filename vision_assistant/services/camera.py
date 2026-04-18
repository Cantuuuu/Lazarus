"""
services/camera.py — Captura continua de cámara con ring buffer thread-safe.

CameraService corre en su propio thread daemon y mantiene un deque circular
con los últimos N segundos de frames. Los triggers consumen frames vía
get_latest_frame() y get_buffer_snapshot() sin bloquear la captura.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
from loguru import logger


class CameraService:
    """Captura continua de cámara con ring buffer thread-safe.

    Args:
        config: Sección "camera" del config.yaml.

    Example:
        cam = CameraService(config["camera"])
        cam.start()
        frame = cam.get_latest_frame()
        cam.stop()
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._device_index: int = config.get("device_index", 0)
        self._width: int = config.get("width", 1280)
        self._height: int = config.get("height", 720)
        self._fps: int = config.get("fps", 30)
        self._buffer_seconds: int = config.get("buffer_seconds", 10)

        # Ring buffer: maxlen = fps × buffer_seconds
        maxlen = self._fps * self._buffer_seconds
        self._buffer: deque[np.ndarray] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Abre la cámara e inicia el thread de captura."""
        if self._running:
            logger.warning("CameraService ya está corriendo, ignorando start()")
            return

        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara con índice {self._device_index}"
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraService",
            daemon=True,
        )
        self._running = True
        self._thread.start()
        logger.info(
            f"CameraService iniciado: device={self._device_index} "
            f"{self._width}x{self._height}@{self._fps}fps "
            f"buffer={self._buffer_seconds}s ({self._buffer.maxlen} frames)"
        )

    def stop(self) -> None:
        """Señala al thread que pare y espera a que termine."""
        if not self._running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._running = False
        logger.info("CameraService detenido")

    # ------------------------------------------------------------------
    # API pública para los triggers
    # ------------------------------------------------------------------

    def get_latest_frame(self) -> np.ndarray | None:
        """Retorna el frame más reciente del buffer, o None si está vacío.

        Thread-safe. No bloquea al capturador.
        """
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1].copy()

    def get_buffer_snapshot(self) -> list[np.ndarray]:
        """Retorna una copia de todos los frames en el ring buffer.

        Útil para que el dispatcher guarde los últimos N segundos de video.
        Thread-safe.
        """
        with self._lock:
            return [f.copy() for f in self._buffer]

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def buffer_size(self) -> int:
        """Número de frames actualmente en el buffer."""
        with self._lock:
            return len(self._buffer)

    # ------------------------------------------------------------------
    # Loop interno
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Loop de captura que corre en el thread daemon."""
        frame_interval = 1.0 / self._fps
        consecutive_failures = 0
        max_failures = 30  # ~1 segundo a 30 fps antes de abortar

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            ret, frame = self._cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                logger.warning(
                    f"Fallo al leer frame ({consecutive_failures}/{max_failures})"
                )
                if consecutive_failures >= max_failures:
                    logger.error(
                        "Demasiados fallos consecutivos de cámara, deteniendo captura"
                    )
                    self._running = False
                    break
                time.sleep(frame_interval)
                continue

            consecutive_failures = 0

            with self._lock:
                self._buffer.append(frame)

            # Mantener el ritmo de captura aproximado
            elapsed = time.monotonic() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.debug("_capture_loop finalizado")
