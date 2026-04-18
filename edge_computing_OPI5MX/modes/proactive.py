"""Modo proactivo: frame → YOLO → AlertEngine → TTS → audio."""
from __future__ import annotations

import asyncio
import logging

import numpy as np

from core.alert_engine import AlertEngine
from core.audio_manager import AudioManager
from core.detector import DetectorBackend, Detection
from core.traffic_light import classify_traffic_light
from services.elevenlabs_client import ElevenLabsClient
from services.stream_reader import StreamReader

logger = logging.getLogger(__name__)

PRIORITY_ALERT = 3
PRIORITY_NORMAL = 5
PRIORITY_LOW = 8

_ALERT_CLASSES = frozenset({"traffic_light", "stairs", "stop_sign"})
_NORMAL_CLASSES = frozenset({"person", "car", "bicycle"})


def _detection_priority(det: Detection) -> int:
    if det.class_name in _ALERT_CLASSES:
        return PRIORITY_ALERT
    if det.class_name in _NORMAL_CLASSES:
        return PRIORITY_NORMAL
    return PRIORITY_LOW


class ProactiveMode:
    def __init__(
        self,
        stream: StreamReader,
        detector: DetectorBackend,
        alert_engine: AlertEngine,
        tts: ElevenLabsClient,
        audio: AudioManager,
        *,
        fps_limit: float = 5.0,
    ) -> None:
        self._stream = stream
        self._detector = detector
        self._alert_engine = alert_engine
        self._tts = tts
        self._audio = audio
        self.fps_limit = fps_limit
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Inicia el loop proactivo como tarea en background."""
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Detiene el loop limpiamente."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _loop(self) -> None:
        frame_interval = 1.0 / self.fps_limit
        while self._running:
            try:
                frame = await self._stream.get_frame()
                if frame is not None:
                    await self._process_frame(frame)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Error en loop proactivo: %s", e)
            await asyncio.sleep(frame_interval)

    async def _process_frame(self, frame: np.ndarray) -> None:
        """Detecta objetos y dispara alertas para detecciones relevantes."""
        try:
            detections = self._detector.detect(frame)
        except Exception as e:
            logger.error("Error en detección: %s", e)
            return

        for det in detections:
            if not self._alert_engine.should_alert(det):
                continue
            self._alert_engine.mark_alerted(det)
            message = _build_message(frame, det, self._alert_engine)
            priority = _detection_priority(det)
            audio_bytes = await self._tts.generate(message)
            await self._audio.speak(message, audio_bytes, priority=priority)


def _build_message(frame: np.ndarray, det: Detection, alert_engine: AlertEngine) -> str:
    if det.class_name == "traffic_light":
        color = classify_traffic_light(frame, det)
        direction = alert_engine.direction_message(det)
        base = direction.replace("semáforo", "").strip()
        return f"semáforo {color} {base}".strip()
    return alert_engine.direction_message(det)
