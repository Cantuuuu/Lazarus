"""Tests para GeminiClient — TDD fase RED antes de implementar services/gemini_client.py."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from services.gemini_client import GeminiClient, _FALLBACK, _MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_client(*, dev_mode: bool = True, rpm_limit: int = 10) -> GeminiClient:
    return GeminiClient(api_key="test-key", dev_mode=dev_mode, rpm_limit=rpm_limit)


def _make_api_response(text: str = "Hay una persona frente a ti.") -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": text}]
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# Dev mode: retorna descripción mock sin llamar la API
# ---------------------------------------------------------------------------

class TestDevMode:
    async def test_returns_non_empty_string_in_dev_mode(self) -> None:
        client = _make_client(dev_mode=True)
        result = await client.describe_scene(_make_frame())
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_dev_mode_does_not_call_api(self) -> None:
        client = _make_client(dev_mode=True)
        with patch("httpx.AsyncClient") as mock_cls:
            await client.describe_scene(_make_frame())
            mock_cls.assert_not_called()

    async def test_dev_mode_returns_mock_description(self) -> None:
        """Dev mode debe retornar un string descriptivo predecible (no el fallback)."""
        client = _make_client(dev_mode=True)
        result = await client.describe_scene(_make_frame(), "¿Qué hay aquí?")
        # No debe ser el string de fallback de error — es una respuesta mock válida
        assert result != _FALLBACK


# ---------------------------------------------------------------------------
# describe_scene: retorna string en español
# ---------------------------------------------------------------------------

class TestDescribeScene:
    async def test_returns_string(self) -> None:
        client = _make_client(dev_mode=False)
        response_json = _make_api_response("Frente a ti hay una persona caminando.")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=response_json)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_http

            result = await client.describe_scene(_make_frame())

        assert isinstance(result, str)
        assert len(result) > 0

    async def test_returns_text_from_api_response(self) -> None:
        expected = "Hay un automóvil rojo estacionado frente a ti."
        client = _make_client(dev_mode=False)
        response_json = _make_api_response(expected)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=response_json)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_http

            result = await client.describe_scene(_make_frame())

        assert result == expected

    async def test_accepts_custom_question(self) -> None:
        question = "¿Hay escaleras cerca?"
        client = _make_client(dev_mode=False)
        response_json = _make_api_response("No veo escaleras en este momento.")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=response_json)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_http

            result = await client.describe_scene(_make_frame(), question)

        # La pregunta debe haberse incluido en el cuerpo de la solicitud
        call_kwargs = mock_http.post.call_args[1]
        body = call_kwargs.get("json", {})
        parts = body["contents"][0]["parts"]
        questions_in_parts = [p["text"] for p in parts if "text" in p]
        assert any(question in q for q in questions_in_parts)


# ---------------------------------------------------------------------------
# Rate limiting: máximo rpm_limit por minuto
# ---------------------------------------------------------------------------

class TestRateLimit:
    async def test_rate_limiter_allows_calls_under_limit(self) -> None:
        client = _make_client(dev_mode=False, rpm_limit=10)
        response_json = _make_api_response("Escena descrita.")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=response_json)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_http

            # 3 llamadas bajo el límite de 10 deben completarse sin delay artificial
            for _ in range(3):
                result = await client.describe_scene(_make_frame())
                assert isinstance(result, str)

    async def test_rate_limiter_tracks_timestamps(self) -> None:
        """El rate limiter debe registrar timestamps de cada llamada."""
        client = _make_client(dev_mode=False, rpm_limit=10)
        response_json = _make_api_response("Descripción.")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=response_json)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_http

            await client.describe_scene(_make_frame())

        assert len(client._timestamps) == 1

    async def test_rate_limiter_sleeps_when_limit_reached(self) -> None:
        """Cuando se alcanzan rpm_limit llamadas, _rate_limit debe invocar asyncio.sleep."""
        client = _make_client(dev_mode=False, rpm_limit=2)
        now = time.monotonic()
        # Simular que ya se hicieron 2 llamadas hace 5 segundos (dentro de la ventana de 60s)
        client._timestamps = deque([now - 5, now - 3])

        sleep_called = False

        async def fake_sleep(duration: float) -> None:
            nonlocal sleep_called
            sleep_called = True

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await client._rate_limit()

        assert sleep_called, "_rate_limit debe llamar asyncio.sleep cuando se alcanza el límite"

    async def test_rate_limiter_removes_old_timestamps(self) -> None:
        """Timestamps más antiguos que 60s deben eliminarse del deque."""
        client = _make_client(dev_mode=False, rpm_limit=10)
        now = time.monotonic()
        # Agregar timestamps viejos (>60s) y recientes
        client._timestamps = deque([now - 120, now - 90, now - 5])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await client._rate_limit()

        # Solo el timestamp reciente debe quedar (más el que agrega _rate_limit en sí)
        remaining = [t for t in client._timestamps if now - t <= 60]
        assert len(remaining) >= 1
        assert all(now - t <= 61 for t in client._timestamps)


# ---------------------------------------------------------------------------
# API call: modelo correcto y estructura de la petición
# ---------------------------------------------------------------------------

class TestAPICall:
    async def test_uses_correct_model(self) -> None:
        client = _make_client(dev_mode=False)
        response_json = _make_api_response("Descripción de escena.")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=response_json)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_http

            await client.describe_scene(_make_frame())

        called_url: str = mock_http.post.call_args[0][0]
        assert _MODEL in called_url, f"URL debe contener el modelo '{_MODEL}', fue: {called_url}"

    async def test_sends_api_key_as_query_param(self) -> None:
        api_key = "my-gemini-key"
        client = GeminiClient(api_key=api_key, dev_mode=False)
        response_json = _make_api_response("Descripción.")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=response_json)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_http

            await client.describe_scene(_make_frame())

        called_url: str = mock_http.post.call_args[0][0]
        assert api_key in called_url

    async def test_sends_frame_as_base64_jpeg(self) -> None:
        import base64
        client = _make_client(dev_mode=False)
        response_json = _make_api_response("Descripción.")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=response_json)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_http

            await client.describe_scene(_make_frame())

        body = mock_http.post.call_args[1]["json"]
        parts = body["contents"][0]["parts"]
        image_parts = [p for p in parts if "inline_data" in p]
        assert len(image_parts) == 1
        inline = image_parts[0]["inline_data"]
        assert inline["mime_type"] == "image/jpeg"
        # Debe ser base64 válido
        decoded = base64.b64decode(inline["data"])
        assert len(decoded) > 0

    async def test_request_body_structure(self) -> None:
        client = _make_client(dev_mode=False)
        response_json = _make_api_response("Descripción.")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value=response_json)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_http

            await client.describe_scene(_make_frame())

        body = mock_http.post.call_args[1]["json"]
        assert "contents" in body
        assert len(body["contents"]) == 1
        assert "parts" in body["contents"][0]


# ---------------------------------------------------------------------------
# Error handling: nunca lanza excepción, retorna fallback
# ---------------------------------------------------------------------------

class TestErrorHandling:
    async def test_returns_fallback_on_http_error(self) -> None:
        client = _make_client(dev_mode=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_resp.json = MagicMock(return_value={})

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_http

            result = await client.describe_scene(_make_frame())

        assert result == _FALLBACK

    async def test_returns_fallback_on_network_exception(self) -> None:
        import httpx as _httpx
        client = _make_client(dev_mode=False)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(side_effect=_httpx.RequestError("timeout"))
            mock_cls.return_value = mock_http

            result = await client.describe_scene(_make_frame())

        assert result == _FALLBACK

    async def test_does_not_raise_on_malformed_response(self) -> None:
        """Una respuesta JSON mal formada no debe propagar excepción."""
        client = _make_client(dev_mode=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json = MagicMock(return_value={"unexpected": "structure"})

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_http

            result = await client.describe_scene(_make_frame())

        assert isinstance(result, str)
        assert result == _FALLBACK

    async def test_describe_scene_never_raises(self) -> None:
        """describe_scene NO debe propagar ninguna excepción bajo ninguna condición."""
        client = _make_client(dev_mode=False)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(side_effect=Exception("Fallo catastrófico"))
            mock_cls.return_value = mock_http

            # No debe lanzar
            result = await client.describe_scene(_make_frame())

        assert result == _FALLBACK
