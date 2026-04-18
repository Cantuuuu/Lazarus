"""Speech-to-text usando Whisper (local, CPU/NPU)."""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000


class WhisperSTT:
    """Transcribe audio PCM int16 a texto usando whisper."""

    def __init__(self, model_size: str = "tiny") -> None:
        self._model_size = model_size
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import whisper  # type: ignore
            self._model = whisper.load_model(self._model_size)
            logger.info("Whisper modelo '%s' cargado", self._model_size)
        except ImportError as e:
            raise RuntimeError(
                "openai-whisper no disponible — instala con: pip install openai-whisper"
            ) from e

    def transcribe(self, pcm_bytes: bytes) -> str:
        """Transcribe PCM int16 (16kHz mono) a texto.

        Retorna cadena vacía si el audio está vacío o hay error.
        """
        if not pcm_bytes:
            return ""
        try:
            self._load()
        except RuntimeError as e:
            logger.error("No se pudo cargar Whisper: %s", e)
            return ""

        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            result = self._model.transcribe(audio, language="es", fp16=False)
            text = result.get("text", "").strip()
            logger.debug("Transcripción: %r", text)
            return text
        except Exception as exc:
            logger.error("Error en transcripción: %s", exc)
            return ""
