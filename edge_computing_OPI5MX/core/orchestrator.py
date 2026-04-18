"""Orquestador central: coordina modos proactivo/reactivo y pausa entre ellos."""
from __future__ import annotations

import asyncio
import logging
from enum import Enum

from core.audio_manager import AudioManager
from core.alert_engine import AlertEngine
from core.detector import DetectorBackend
from modes.proactive import ProactiveMode
from modes.reactive import ReactiveMode
from services.elevenlabs_client import ElevenLabsClient
from services.gemini_client import GeminiClient
from services.stream_reader import StreamReader

logger = logging.getLogger(__name__)


class Mode(str, Enum):
    PROACTIVE = "proactive"
    REACTIVE = "reactive"
    PAUSED = "paused"


class Orchestrator:
    """Coordina los modos y garantiza que el modo reactivo silencia el proactivo."""

    def __init__(
        self,
        stream: StreamReader,
        detector: DetectorBackend,
        alert_engine: AlertEngine,
        tts: ElevenLabsClient,
        audio: AudioManager,
        gemini: GeminiClient,
        *,
        fps_limit: float = 5.0,
    ) -> None:
        self._proactive = ProactiveMode(
            stream, detector, alert_engine, tts, audio, fps_limit=fps_limit
        )
        self._reactive = ReactiveMode(stream, gemini, tts, audio)
        self._mode = Mode.PAUSED
        self._lock = asyncio.Lock()

    @property
    def mode(self) -> Mode:
        return self._mode

    async def start(self) -> None:
        """Arranca en modo proactivo."""
        async with self._lock:
            await self._set_mode(Mode.PROACTIVE)

    async def stop(self) -> None:
        """Para todo."""
        async with self._lock:
            await self._set_mode(Mode.PAUSED)

    async def set_mode(self, mode: Mode) -> None:
        async with self._lock:
            await self._set_mode(mode)

    async def handle_question(self, question: str) -> str:
        """Pausa el proactivo, responde la pregunta, reanuda el proactivo."""
        async with self._lock:
            was_proactive = self._mode == Mode.PROACTIVE
            if was_proactive:
                await self._stop_proactive()
                self._mode = Mode.REACTIVE

        answer = await self._reactive.handle_question(question)

        async with self._lock:
            if was_proactive and self._mode == Mode.REACTIVE:
                await self._start_proactive()
                self._mode = Mode.PROACTIVE

        return answer

    async def trigger_button(self) -> None:
        """Acción del botón físico: alterna entre proactivo y pausado."""
        async with self._lock:
            if self._mode == Mode.PAUSED:
                await self._set_mode(Mode.PROACTIVE)
            else:
                await self._set_mode(Mode.PAUSED)

    async def _set_mode(self, mode: Mode) -> None:
        if mode == self._mode:
            return
        logger.info("Modo: %s → %s", self._mode, mode)
        if self._mode == Mode.PROACTIVE:
            await self._stop_proactive()
        if mode == Mode.PROACTIVE:
            await self._start_proactive()
        self._mode = mode

    async def _start_proactive(self) -> None:
        await self._proactive.start()

    async def _stop_proactive(self) -> None:
        await self._proactive.stop()
