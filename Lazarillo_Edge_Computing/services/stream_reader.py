"""Lector async de stream MJPEG desde la Pi Zero W."""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)

_BOUNDARY_CANDIDATES = [b"--frame", b"--mjpegstream", b"--myboundary"]


class StreamReader:
    """Lee frames JPEG de un stream MJPEG HTTP.

    Reconecta automáticamente con backoff exponencial si la conexión falla.
    """

    def __init__(
        self,
        url: str,
        *,
        max_retries: int = 10,
        timeout: float = 10.0,
    ) -> None:
        self._url = url
        self._max_retries = max_retries
        self._timeout = timeout
        self._latest_frame: np.ndarray | None = None
        self._lock = asyncio.Lock()
        self._running = False

    async def start(self) -> None:
        """Inicia el loop de captura en background."""
        self._running = True
        asyncio.create_task(self._capture_loop())

    async def stop(self) -> None:
        self._running = False

    async def get_frame(self) -> np.ndarray | None:
        """Retorna el último frame disponible (puede ser None si no hay aún)."""
        async with self._lock:
            return self._latest_frame

    async def frames(self) -> AsyncIterator[np.ndarray]:
        """Itera sobre frames nuevos."""
        last = None
        while self._running:
            frame = await self.get_frame()
            if frame is not None and frame is not last:
                last = frame
                yield frame
            else:
                await asyncio.sleep(0.05)

    async def _capture_loop(self) -> None:
        retries = 0
        while self._running and retries < self._max_retries:
            try:
                await self._read_stream()
                retries = 0  # reset en conexión exitosa
            except Exception as e:
                wait = min(2 ** retries, 30)
                logger.warning(f"Stream desconectado: {e}. Reintentando en {wait}s...")
                retries += 1
                await asyncio.sleep(wait)

        if retries >= self._max_retries:
            logger.error(f"Stream no disponible tras {self._max_retries} intentos: {self._url}")

    async def _read_stream(self) -> None:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream("GET", self._url) as response:
                response.raise_for_status()
                logger.info(f"Conectado al stream: {self._url}")
                buf = b""
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    buf += chunk
                    frames, buf = _extract_frames(buf)
                    for jpeg in frames:
                        frame = _decode_jpeg(jpeg)
                        if frame is not None:
                            async with self._lock:
                                self._latest_frame = frame


def _extract_frames(buf: bytes) -> tuple[list[bytes], bytes]:
    """Extrae frames JPEG completos del buffer MJPEG."""
    frames = []
    while True:
        start = buf.find(b"\xff\xd8")
        end = buf.find(b"\xff\xd9", start + 2)
        if start == -1 or end == -1:
            break
        frames.append(buf[start:end + 2])
        buf = buf[end + 2:]
    return frames, buf


def _decode_jpeg(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame if frame is not None else None
