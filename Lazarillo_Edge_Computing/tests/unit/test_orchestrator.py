"""Unit tests for core/orchestrator.py."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.orchestrator import Mode, Orchestrator


def _make_orchestrator(**kwargs):
    stream = MagicMock()
    detector = MagicMock()
    alert_engine = MagicMock()
    tts = MagicMock()
    audio = MagicMock()
    gemini = MagicMock()
    return Orchestrator(stream, detector, alert_engine, tts, audio, gemini, **kwargs)


class TestMode:
    def test_enum_values(self) -> None:
        assert Mode.PROACTIVE == "proactive"
        assert Mode.REACTIVE == "reactive"
        assert Mode.PAUSED == "paused"


class TestOrchestratorInit:
    def test_initial_mode_is_paused(self) -> None:
        orch = _make_orchestrator()
        assert orch.mode == Mode.PAUSED

    def test_fps_limit_forwarded(self) -> None:
        orch = _make_orchestrator(fps_limit=3.0)
        assert orch._proactive.fps_limit == 3.0


class TestOrchestratorStart:
    @pytest.mark.asyncio
    async def test_start_sets_proactive_mode(self) -> None:
        orch = _make_orchestrator()
        with patch.object(orch._proactive, "start", new_callable=AsyncMock):
            await orch.start()
        assert orch.mode == Mode.PROACTIVE

    @pytest.mark.asyncio
    async def test_start_calls_proactive_start(self) -> None:
        orch = _make_orchestrator()
        mock_start = AsyncMock()
        with patch.object(orch._proactive, "start", mock_start):
            await orch.start()
        mock_start.assert_called_once()


class TestOrchestratorStop:
    @pytest.mark.asyncio
    async def test_stop_sets_paused_mode(self) -> None:
        orch = _make_orchestrator()
        with patch.object(orch._proactive, "start", new_callable=AsyncMock):
            await orch.start()
        with patch.object(orch._proactive, "stop", new_callable=AsyncMock):
            await orch.stop()
        assert orch.mode == Mode.PAUSED

    @pytest.mark.asyncio
    async def test_stop_from_paused_stays_paused(self) -> None:
        orch = _make_orchestrator()
        await orch.stop()
        assert orch.mode == Mode.PAUSED


class TestOrchestratorSetMode:
    @pytest.mark.asyncio
    async def test_set_mode_to_same_is_noop(self) -> None:
        orch = _make_orchestrator()
        mock_start = AsyncMock()
        with patch.object(orch._proactive, "start", mock_start):
            await orch.set_mode(Mode.PAUSED)
        mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_mode_to_proactive(self) -> None:
        orch = _make_orchestrator()
        mock_start = AsyncMock()
        with patch.object(orch._proactive, "start", mock_start):
            await orch.set_mode(Mode.PROACTIVE)
        assert orch.mode == Mode.PROACTIVE
        mock_start.assert_called_once()


class TestHandleQuestion:
    @pytest.mark.asyncio
    async def test_pauses_proactive_during_question(self) -> None:
        orch = _make_orchestrator()
        start_mock = AsyncMock()
        stop_mock = AsyncMock()
        answer_mock = AsyncMock(return_value="hay una persona")
        with patch.object(orch._proactive, "start", start_mock):
            await orch.start()
        with (
            patch.object(orch._proactive, "stop", stop_mock),
            patch.object(orch._reactive, "handle_question", answer_mock),
            patch.object(orch._proactive, "start", start_mock),
        ):
            result = await orch.handle_question("¿qué ves?")
        assert result == "hay una persona"

    @pytest.mark.asyncio
    async def test_resumes_proactive_after_question(self) -> None:
        orch = _make_orchestrator()
        start_calls = []

        async def mock_start():
            start_calls.append(1)

        with patch.object(orch._proactive, "start", mock_start):
            await orch.start()

        stop_mock = AsyncMock()
        answer_mock = AsyncMock(return_value="una calle")
        with (
            patch.object(orch._proactive, "stop", stop_mock),
            patch.object(orch._reactive, "handle_question", answer_mock),
            patch.object(orch._proactive, "start", mock_start),
        ):
            await orch.handle_question("¿qué hay?")

        assert orch.mode == Mode.PROACTIVE
        assert len(start_calls) == 2

    @pytest.mark.asyncio
    async def test_question_from_paused_does_not_restart_proactive(self) -> None:
        orch = _make_orchestrator()
        start_mock = AsyncMock()
        answer_mock = AsyncMock(return_value="nada")
        with (
            patch.object(orch._proactive, "start", start_mock),
            patch.object(orch._reactive, "handle_question", answer_mock),
        ):
            await orch.handle_question("¿qué ves?")
        start_mock.assert_not_called()
        assert orch.mode == Mode.PAUSED


class TestTriggerButton:
    @pytest.mark.asyncio
    async def test_button_from_paused_starts_proactive(self) -> None:
        orch = _make_orchestrator()
        start_mock = AsyncMock()
        with patch.object(orch._proactive, "start", start_mock):
            await orch.trigger_button()
        assert orch.mode == Mode.PROACTIVE
        start_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_button_from_proactive_pauses(self) -> None:
        orch = _make_orchestrator()
        start_mock = AsyncMock()
        stop_mock = AsyncMock()
        with patch.object(orch._proactive, "start", start_mock):
            await orch.start()
        with patch.object(orch._proactive, "stop", stop_mock):
            await orch.trigger_button()
        assert orch.mode == Mode.PAUSED
        stop_mock.assert_called_once()
