"""Modo reactivo: pregunta → Gemini → respuesta de voz."""
from __future__ import annotations

import logging

import numpy as np

from core.audio_manager import AudioManager
from services.elevenlabs_client import ElevenLabsClient
from services.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

_NO_IMAGE_MESSAGE = "No tengo imagen en este momento."
_REACTIVE_PRIORITY = 1  # Máxima prioridad en la cola de audio


class ReactiveMode:
    """Responde preguntas del usuario capturando un frame y consultando Gemini."""

    def __init__(
        self,
        stream,
        gemini: GeminiClient,
        tts: ElevenLabsClient,
        audio: AudioManager,
    ) -> None:
        self._stream = stream
        self._gemini = gemini
        self._tts = tts
        self._audio = audio

    async def handle_question(self, question: str) -> str:
        """Obtiene el frame actual, pregunta a Gemini y habla la respuesta.

        Retorna el texto de la respuesta. Nunca lanza excepción.
        """
        frame = await self._stream.get_frame()

        if frame is None:
            logger.warning("No hay frame disponible para responder la pregunta: %r", question)
            return await self._speak_and_return(_NO_IMAGE_MESSAGE)

        answer = await self._gemini.describe_scene(frame, question)

        try:
            audio_bytes = await self._tts.generate(answer)
        except Exception as exc:
            logger.error("Error generando TTS para respuesta reactiva: %s", exc)
            audio_bytes = b""

        await self._audio.speak(answer, audio_bytes, priority=_REACTIVE_PRIORITY)
        return answer

    async def _speak_and_return(self, message: str) -> str:
        """Genera TTS para el mensaje, lo encola y retorna el texto."""
        try:
            audio_bytes = await self._tts.generate(message)
        except Exception as exc:
            logger.error("Error generando TTS para mensaje de error: %s", exc)
            audio_bytes = b""

        await self._audio.speak(message, audio_bytes, priority=_REACTIVE_PRIORITY)
        return message
