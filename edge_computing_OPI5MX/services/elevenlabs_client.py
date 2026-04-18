"""Cliente async ElevenLabs TTS con cache LRU y modo mock."""
from __future__ import annotations

import logging
from collections import OrderedDict

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_ID = "eleven_turbo_v2"
_DEFAULT_STABILITY = 0.5
_DEFAULT_SIMILARITY_BOOST = 0.75

_DEFAULT_VOICE_SETTINGS: dict[str, float] = {
    "stability": _DEFAULT_STABILITY,
    "similarity_boost": _DEFAULT_SIMILARITY_BOOST,
}


# ---------------------------------------------------------------------------
# Cliente
# ---------------------------------------------------------------------------


class ElevenLabsClient:
    """Cliente async para ElevenLabs TTS con cache LRU y modo mock para desarrollo."""

    BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        *,
        dev_mode: bool = True,
        max_cache_size: int = 50,
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._dev_mode = dev_mode
        self._max_cache_size = max_cache_size
        self._cache: OrderedDict[str, bytes] = OrderedDict()

    async def generate(self, text: str) -> bytes:
        """Retorna bytes MP3 del audio generado. En dev_mode retorna bytes vacíos."""
        if self._dev_mode:
            logger.debug("dev_mode activo — omitiendo llamada TTS: %r", text)
            return b""

        cached = self._cache_get(text)
        if cached is not None:
            logger.debug("Cache hit para texto: %r", text)
            return cached

        audio = await self._fetch_audio(text)
        self._cache_put(text, audio)
        return audio

    async def _fetch_audio(self, text: str) -> bytes:
        """Realiza la llamada HTTP a ElevenLabs y retorna los bytes de audio."""
        url = f"{self.BASE_URL}/text-to-speech/{self._voice_id}"
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        body = {
            "text": text,
            "model_id": _DEFAULT_MODEL_ID,
            "voice_settings": dict(_DEFAULT_VOICE_SETTINGS),
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=body, headers=headers)

        if response.status_code != 200:
            raise RuntimeError(
                f"ElevenLabs API error {response.status_code}: {response.text}"
            )

        return response.content

    # ------------------------------------------------------------------
    # Cache LRU (OrderedDict: último en orden = más reciente)
    # ------------------------------------------------------------------

    def _cache_get(self, key: str) -> bytes | None:
        """Retorna el valor cacheado o None. Mueve la entrada al final (MRU)."""
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def _cache_put(self, key: str, value: bytes) -> None:
        """Inserta o actualiza una entrada. Evicta la más antigua si se supera el límite."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return

        self._cache[key] = value
        if len(self._cache) > self._max_cache_size:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug("Cache LRU: eviccionando entrada %r", evicted_key)
