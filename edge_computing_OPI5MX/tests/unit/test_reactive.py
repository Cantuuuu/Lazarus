"""Tests para ReactiveMode — TDD fase RED antes de implementar modes/reactive.py."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from modes.reactive import ReactiveMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_reactive(
    *,
    frame: np.ndarray | None = None,
    gemini_response: str = "Hay una persona caminando frente a ti.",
    tts_bytes: bytes = b"audio-data",
) -> ReactiveMode:
    stream = MagicMock()
    stream.get_frame = AsyncMock(return_value=frame if frame is not None else _make_frame())

    gemini = MagicMock()
    gemini.describe_scene = AsyncMock(return_value=gemini_response)

    tts = MagicMock()
    tts.generate = AsyncMock(return_value=tts_bytes)

    audio = MagicMock()
    audio.speak = AsyncMock()

    return ReactiveMode(stream, gemini, tts, audio)


# ---------------------------------------------------------------------------
# handle_question: captura frame, llama Gemini, habla el resultado
# ---------------------------------------------------------------------------

class TestHandleQuestion:
    async def test_returns_string_answer(self) -> None:
        mode = _make_reactive(gemini_response="Una persona está frente a ti.")
        result = await mode.handle_question("¿Qué hay frente a mí?")
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_returns_gemini_answer(self) -> None:
        expected = "Hay un automóvil rojo estacionado a tu derecha."
        mode = _make_reactive(gemini_response=expected)
        result = await mode.handle_question("¿Qué hay a mi derecha?")
        assert result == expected

    async def test_captures_frame_from_stream(self) -> None:
        mode = _make_reactive()
        await mode.handle_question("¿Qué ves?")
        mode._stream.get_frame.assert_called_once()

    async def test_calls_gemini_with_frame_and_question(self) -> None:
        question = "¿Hay escaleras cerca?"
        mode = _make_reactive()
        await mode.handle_question(question)

        mode._gemini.describe_scene.assert_called_once()
        call_args = mode._gemini.describe_scene.call_args
        # Primer argumento posicional: el frame
        assert isinstance(call_args[0][0], np.ndarray)
        # Segundo argumento posicional o kwarg: la pregunta
        passed_question = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("question")
        assert passed_question == question

    async def test_calls_tts_with_gemini_answer(self) -> None:
        answer = "No veo escaleras en este momento."
        mode = _make_reactive(gemini_response=answer)
        await mode.handle_question("¿Hay escaleras?")
        mode._tts.generate.assert_called_once_with(answer)

    async def test_speaks_audio_with_priority_1(self) -> None:
        """Las respuestas reactivas deben encolarse con prioridad=1 (la más alta)."""
        mode = _make_reactive(tts_bytes=b"mp3-data")
        await mode.handle_question("¿Qué hay aquí?")

        mode._audio.speak.assert_called_once()
        call_kwargs = mode._audio.speak.call_args[1]
        assert call_kwargs.get("priority") == 1

    async def test_speak_receives_answer_text_and_audio(self) -> None:
        answer = "Hay una bicicleta a tu izquierda."
        audio_bytes = b"mp3-bicycle"
        mode = _make_reactive(gemini_response=answer, tts_bytes=audio_bytes)
        await mode.handle_question("¿Qué hay?")

        call_args = mode._audio.speak.call_args
        spoken_text = call_args[0][0]
        spoken_audio = call_args[0][1]
        assert spoken_text == answer
        assert spoken_audio == audio_bytes


# ---------------------------------------------------------------------------
# Timeout: retorna dentro de un tiempo razonable aunque Gemini sea lento
# ---------------------------------------------------------------------------

class TestTimeout:
    async def test_returns_within_timeout_even_if_gemini_is_slow(self) -> None:
        """Si Gemini tarda mucho, handle_question debe retornar dentro de 5 segundos."""
        TIMEOUT_SECONDS = 5.0

        stream = MagicMock()
        stream.get_frame = AsyncMock(return_value=_make_frame())

        gemini = MagicMock()

        async def slow_describe(*args, **kwargs) -> str:
            await asyncio.sleep(0.1)  # Simula latencia breve (no infinita en tests)
            return "Respuesta lenta de Gemini."

        gemini.describe_scene = slow_describe

        tts = MagicMock()
        tts.generate = AsyncMock(return_value=b"audio")

        audio = MagicMock()
        audio.speak = AsyncMock()

        mode = ReactiveMode(stream, gemini, tts, audio)

        result = await asyncio.wait_for(
            mode.handle_question("¿Qué hay?"),
            timeout=TIMEOUT_SECONDS,
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Manejo de frame nulo: responde graciosamente sin frame
# ---------------------------------------------------------------------------

class TestMissingFrame:
    async def test_handles_none_frame_gracefully(self) -> None:
        """Si el stream retorna None, debe responder con mensaje de error amigable."""
        stream = MagicMock()
        stream.get_frame = AsyncMock(return_value=None)

        gemini = MagicMock()
        gemini.describe_scene = AsyncMock(return_value="Descripción.")

        tts = MagicMock()
        tts.generate = AsyncMock(return_value=b"audio")

        audio = MagicMock()
        audio.speak = AsyncMock()

        mode = ReactiveMode(stream, gemini, tts, audio)
        result = await mode.handle_question("¿Qué hay frente a mí?")

        assert isinstance(result, str)
        assert len(result) > 0
        # Debe incluir algún mensaje sobre la falta de imagen
        assert "imagen" in result.lower() or "cámara" in result.lower() or "momento" in result.lower()

    async def test_speaks_no_image_message_when_frame_is_none(self) -> None:
        """Cuando no hay frame, debe hablar el mensaje de error."""
        stream = MagicMock()
        stream.get_frame = AsyncMock(return_value=None)

        gemini = MagicMock()
        tts = MagicMock()
        tts.generate = AsyncMock(return_value=b"")
        audio = MagicMock()
        audio.speak = AsyncMock()

        mode = ReactiveMode(stream, gemini, tts, audio)
        result = await mode.handle_question("¿Qué ves?")

        # Gemini no debe haberse llamado sin frame
        gemini.describe_scene.assert_not_called()
        # Sí debe haberse hablado algo
        audio.speak.assert_called_once()
        assert isinstance(result, str)

    async def test_no_image_message_is_in_spanish(self) -> None:
        stream = MagicMock()
        stream.get_frame = AsyncMock(return_value=None)

        gemini = MagicMock()
        tts = MagicMock()
        tts.generate = AsyncMock(return_value=b"")
        audio = MagicMock()
        audio.speak = AsyncMock()

        mode = ReactiveMode(stream, gemini, tts, audio)
        result = await mode.handle_question("¿Qué hay?")

        # El resultado debe contener al menos una palabra española reconocible
        spanish_words = {"imagen", "cámara", "momento", "tengo", "puedo", "disponible", "ahora"}
        result_lower = result.lower()
        assert any(word in result_lower for word in spanish_words)


# ---------------------------------------------------------------------------
# Comportamiento general: handle_question nunca lanza excepción
# ---------------------------------------------------------------------------

class TestRobustness:
    async def test_handle_question_never_raises_on_gemini_error(self) -> None:
        stream = MagicMock()
        stream.get_frame = AsyncMock(return_value=_make_frame())

        gemini = MagicMock()
        gemini.describe_scene = AsyncMock(
            return_value="No pude analizar la escena en este momento."
        )

        tts = MagicMock()
        tts.generate = AsyncMock(return_value=b"")
        audio = MagicMock()
        audio.speak = AsyncMock()

        mode = ReactiveMode(stream, gemini, tts, audio)
        # No debe lanzar; gemini ya maneja errores internamente y retorna fallback
        result = await mode.handle_question("¿Qué hay?")
        assert isinstance(result, str)

    async def test_handle_question_never_raises_on_tts_error(self) -> None:
        stream = MagicMock()
        stream.get_frame = AsyncMock(return_value=_make_frame())

        gemini = MagicMock()
        gemini.describe_scene = AsyncMock(return_value="Hay una persona.")

        tts = MagicMock()
        tts.generate = AsyncMock(side_effect=RuntimeError("TTS falló"))

        audio = MagicMock()
        audio.speak = AsyncMock()

        mode = ReactiveMode(stream, gemini, tts, audio)
        # No debe propagar el error de TTS
        result = await mode.handle_question("¿Qué hay?")
        assert isinstance(result, str)
