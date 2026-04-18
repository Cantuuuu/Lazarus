"""Captura de voz con VAD simple basado en energía RMS."""
from __future__ import annotations

import asyncio
import logging
from collections import deque

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_CHUNK_FRAMES = 512
_SILENCE_THRESHOLD = 300
_MIN_SPEECH_CHUNKS = 5
_SILENCE_CHUNKS = 20


class VoiceCapture:
    """Escucha el micrófono y produce segmentos de voz completos."""

    def __init__(
        self,
        *,
        device: str = "default",
        sample_rate: int = _SAMPLE_RATE,
        silence_threshold: int = _SILENCE_THRESHOLD,
        min_speech_chunks: int = _MIN_SPEECH_CHUNKS,
        silence_chunks: int = _SILENCE_CHUNKS,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._silence_threshold = silence_threshold
        self._min_speech_chunks = min_speech_chunks
        self._silence_chunks = silence_chunks
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    async def listen_once(self) -> bytes | None:
        """Graba un único enunciado y retorna los bytes PCM (int16).

        Retorna None si no se detecta voz o si se produce error.
        """
        try:
            import sounddevice as sd  # type: ignore
            import numpy as np
        except ImportError:
            logger.warning("sounddevice no disponible — VoiceCapture no funciona")
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._record_sync, sd, np)

    def _record_sync(self, sd, np) -> bytes | None:  # type: ignore[return]
        chunks: list[bytes] = []
        speech_chunks = 0
        silence_count = 0
        in_speech = False

        def callback(indata, frames, time, status):  # noqa: ANN001
            nonlocal speech_chunks, silence_count, in_speech
            rms = int(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
            is_voice = rms > self._silence_threshold
            if is_voice:
                silence_count = 0
                speech_chunks += 1
                in_speech = True
                chunks.append(indata.tobytes())
            elif in_speech:
                silence_count += 1
                chunks.append(indata.tobytes())

        try:
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
                blocksize=_CHUNK_FRAMES,
                device=self._device if self._device != "default" else None,
                callback=callback,
            ):
                import time as _time
                deadline = _time.monotonic() + 10.0
                while _time.monotonic() < deadline:
                    _time.sleep(0.05)
                    if in_speech and silence_count >= self._silence_chunks:
                        break
        except Exception as exc:
            logger.error("Error capturando audio: %s", exc)
            return None

        if speech_chunks < self._min_speech_chunks:
            return None
        return b"".join(chunks)
