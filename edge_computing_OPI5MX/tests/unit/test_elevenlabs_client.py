"""Tests para ElevenLabsClient — TDD fase RED antes de implementar services/elevenlabs_client.py."""
from __future__ import annotations

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from services.elevenlabs_client import ElevenLabsClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_client(
    *,
    dev_mode: bool = True,
    api_key: str = "test-key",
    voice_id: str = "test-voice",
    max_cache_size: int = 50,
) -> ElevenLabsClient:
    """Crea un ElevenLabsClient con parámetros controlados."""
    return ElevenLabsClient(
        api_key=api_key,
        voice_id=voice_id,
        dev_mode=dev_mode,
        max_cache_size=max_cache_size,
    )


# ---------------------------------------------------------------------------
# Modo dev: devuelve bytes vacíos sin llamar la API
# ---------------------------------------------------------------------------

class TestDevMode:
    async def test_returns_empty_bytes_in_dev_mode(self) -> None:
        client = make_client(dev_mode=True)
        result = await client.generate("persona al frente")
        assert result == b""

    async def test_dev_mode_does_not_call_api(self) -> None:
        client = make_client(dev_mode=True)
        with patch("httpx.AsyncClient.post") as mock_post:
            await client.generate("auto a la izquierda")
            mock_post.assert_not_called()

    async def test_generate_returns_bytes_type(self) -> None:
        client = make_client(dev_mode=True)
        result = await client.generate("semáforo al frente")
        assert isinstance(result, bytes)

    async def test_generate_is_awaitable(self) -> None:
        """generate() debe ser async."""
        client = make_client(dev_mode=True)
        import inspect
        coro = client.generate("test")
        assert inspect.isawaitable(coro)
        await coro


# ---------------------------------------------------------------------------
# Cache: mismo texto retorna bytes cacheados sin segunda llamada a la API
# ---------------------------------------------------------------------------

class TestCache:
    async def test_same_text_uses_cache_no_second_api_call(self) -> None:
        fake_audio = b"fake-mp3-data"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_audio

        client = make_client(dev_mode=False)
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_http

            first = await client.generate("persona al frente")
            second = await client.generate("persona al frente")

        assert first == fake_audio
        assert second == fake_audio
        # La API solo debe haberse llamado una vez
        assert mock_http.post.call_count == 1

    async def test_different_texts_each_call_api(self) -> None:
        fake_audio_1 = b"mp3-persona"
        fake_audio_2 = b"mp3-auto"

        def make_response(content: bytes) -> MagicMock:
            r = MagicMock()
            r.status_code = 200
            r.content = content
            return r

        client = make_client(dev_mode=False)
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(
                side_effect=[make_response(fake_audio_1), make_response(fake_audio_2)]
            )
            mock_client_cls.return_value = mock_http

            r1 = await client.generate("persona al frente")
            r2 = await client.generate("auto a la izquierda")

        assert r1 == fake_audio_1
        assert r2 == fake_audio_2
        assert mock_http.post.call_count == 2

    async def test_cache_returns_correct_bytes_per_text(self) -> None:
        """Cada clave de cache retorna sus propios bytes."""
        fake_a = b"bytes-a"
        fake_b = b"bytes-b"

        def make_response(content: bytes) -> MagicMock:
            r = MagicMock()
            r.status_code = 200
            r.content = content
            return r

        client = make_client(dev_mode=False)
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(
                side_effect=[make_response(fake_a), make_response(fake_b)]
            )
            mock_client_cls.return_value = mock_http

            await client.generate("texto-a")
            await client.generate("texto-b")

            # Segunda pasada — deben venir de cache
            cached_a = await client.generate("texto-a")
            cached_b = await client.generate("texto-b")

        assert cached_a == fake_a
        assert cached_b == fake_b


# ---------------------------------------------------------------------------
# LRU cache: evicción cuando se supera max_cache_size
# ---------------------------------------------------------------------------

class TestLRUCacheEviction:
    async def test_lru_evicts_oldest_entry_when_full(self) -> None:
        """Al llenar la cache con N+1 entradas, la primera debe ser eviccionada."""
        max_size = 3
        client = make_client(dev_mode=False, max_cache_size=max_size)

        # Rellenamos la cache directamente via _cache_put
        client._cache_put("key-0", b"v0")
        client._cache_put("key-1", b"v1")
        client._cache_put("key-2", b"v2")

        # Insertar una cuarta entrada provoca evicción de key-0
        client._cache_put("key-3", b"v3")

        assert client._cache_get("key-0") is None, "key-0 debería haber sido eviccionada"
        assert client._cache_get("key-1") == b"v1"
        assert client._cache_get("key-2") == b"v2"
        assert client._cache_get("key-3") == b"v3"

    async def test_lru_hit_moves_entry_to_recent(self) -> None:
        """Un cache hit mueve la entrada al final (más reciente), postergando su evicción."""
        max_size = 3
        client = make_client(dev_mode=False, max_cache_size=max_size)

        client._cache_put("key-0", b"v0")
        client._cache_put("key-1", b"v1")
        client._cache_put("key-2", b"v2")

        # Acceder a key-0 lo mueve al frente — ahora key-1 es el más antiguo
        client._cache_get("key-0")

        # Insertar nueva entrada debe evictar key-1, no key-0
        client._cache_put("key-3", b"v3")

        assert client._cache_get("key-0") == b"v0", "key-0 fue accedida, no debe eviccionarse"
        assert client._cache_get("key-1") is None, "key-1 debería haber sido eviccionada"
        assert client._cache_get("key-3") == b"v3"

    async def test_cache_size_does_not_exceed_max(self) -> None:
        max_size = 5
        client = make_client(dev_mode=False, max_cache_size=max_size)

        for i in range(10):
            client._cache_put(f"key-{i}", f"v{i}".encode())

        assert len(client._cache) <= max_size


# ---------------------------------------------------------------------------
# API real: endpoint, headers y body correctos (con mock httpx)
# ---------------------------------------------------------------------------

class TestRealAPICall:
    async def test_calls_correct_endpoint(self) -> None:
        fake_audio = b"mp3-data"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_audio

        voice_id = "test-voice-123"
        client = make_client(dev_mode=False, voice_id=voice_id)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_http

            await client.generate("persona al frente")

        call_args = mock_http.post.call_args
        url = call_args[0][0] if call_args[0] else call_args[1].get("url", call_args.args[0] if call_args.args else "")
        assert f"/text-to-speech/{voice_id}" in url

    async def test_sends_correct_api_key_header(self) -> None:
        fake_audio = b"mp3-data"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_audio

        api_key = "my-secret-key"
        client = make_client(dev_mode=False, api_key=api_key)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_http

            await client.generate("auto a la izquierda")

        call_kwargs = mock_http.post.call_args[1]
        headers = call_kwargs.get("headers", {})
        assert headers.get("xi-api-key") == api_key

    async def test_sends_correct_content_type_header(self) -> None:
        fake_audio = b"mp3-data"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_audio

        client = make_client(dev_mode=False)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_http

            await client.generate("semáforo al frente")

        call_kwargs = mock_http.post.call_args[1]
        headers = call_kwargs.get("headers", {})
        assert headers.get("Content-Type") == "application/json"

    async def test_sends_correct_body(self) -> None:
        fake_audio = b"mp3-data"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_audio

        client = make_client(dev_mode=False)
        text = "persona al frente"

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_http

            await client.generate(text)

        call_kwargs = mock_http.post.call_args[1]
        body = call_kwargs.get("json", {})
        assert body.get("text") == text
        assert body.get("model_id") == "eleven_turbo_v2"
        assert "voice_settings" in body
        assert body["voice_settings"]["stability"] == 0.5
        assert body["voice_settings"]["similarity_boost"] == 0.75

    async def test_raises_runtime_error_on_non_200_response(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        client = make_client(dev_mode=False)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_http

            with pytest.raises(RuntimeError, match="401"):
                await client.generate("persona al frente")
