"""Cliente async Gemini para descripción de escenas."""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections import deque

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)

_FALLBACK = "No pude analizar la escena en este momento."
_MODEL = "gemini-1.5-flash"
_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
_DEV_MOCK_DESCRIPTION = "Modo desarrollo activo. Frente a ti hay una escena simulada con una persona caminando y un automóvil estacionado a la derecha."


class GeminiClient:
    """Cliente async para Gemini Vision con rate limiting y modo mock."""

    def __init__(
        self,
        api_key: str,
        *,
        dev_mode: bool = True,
        rpm_limit: int = 10,
    ) -> None:
        self._api_key = api_key
        self._dev_mode = dev_mode
        self._rpm_limit = rpm_limit
        self._timestamps: deque[float] = deque()

    async def describe_scene(
        self,
        frame: np.ndarray,
        question: str = "¿Qué hay frente a mí?",
    ) -> str:
        """Retorna descripción en español de la escena. Nunca lanza excepción."""
        if self._dev_mode:
            logger.debug("dev_mode activo — retornando descripción mock")
            return _DEV_MOCK_DESCRIPTION

        try:
            await self._rate_limit()
            return await self._call_api(frame, question)
        except Exception as exc:
            logger.error("Error al describir escena con Gemini: %s", exc)
            return _FALLBACK

    async def _rate_limit(self) -> None:
        """Aplica rpm_limit durmiendo si es necesario."""
        now = time.monotonic()

        # Eliminar timestamps más viejos que 60 segundos
        while self._timestamps and now - self._timestamps[0] > 60:
            self._timestamps.popleft()

        if len(self._timestamps) >= self._rpm_limit:
            oldest = self._timestamps[0]
            sleep_duration = 60 - (now - oldest)
            if sleep_duration > 0:
                logger.debug(
                    "Rate limit alcanzado (%d rpm). Esperando %.2fs.",
                    self._rpm_limit,
                    sleep_duration,
                )
                await asyncio.sleep(sleep_duration)
            # Limpiar de nuevo después del sleep
            now = time.monotonic()
            while self._timestamps and now - self._timestamps[0] > 60:
                self._timestamps.popleft()

        self._timestamps.append(time.monotonic())

    async def _call_api(self, frame: np.ndarray, question: str) -> str:
        """Llama a la API de Gemini y retorna el texto de la respuesta."""
        b64_frame = _encode_frame_as_jpeg_b64(frame)
        url = f"{_API_BASE}/models/{_MODEL}:generateContent?key={self._api_key}"
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": question},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_frame,
                            }
                        },
                    ]
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=body)

        if response.status_code != 200:
            logger.error("Gemini API error %d: %s", response.status_code, response.text)
            return _FALLBACK

        try:
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Respuesta Gemini inesperada: %s — %s", exc, response.text[:200])
            return _FALLBACK


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _encode_frame_as_jpeg_b64(frame: np.ndarray) -> str:
    """Codifica un frame numpy como JPEG en base64."""
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("No se pudo codificar el frame como JPEG")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")
