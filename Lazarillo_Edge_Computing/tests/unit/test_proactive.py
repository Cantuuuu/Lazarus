"""Unit tests for modes/proactive.py — all mocked, no real hardware."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.detector import Detection
from modes.proactive import ProactiveMode, _detection_priority, PRIORITY_ALERT, PRIORITY_NORMAL, PRIORITY_LOW


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_detection(class_name: str = "person", confidence: float = 0.9) -> Detection:
    return Detection(0, class_name, confidence, (200, 100, 400, 300))


def _make_proactive(
    detections: list[Detection] | None = None,
    should_alert: bool = True,
) -> ProactiveMode:
    stream = MagicMock()
    stream.get_frame = AsyncMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))

    detector = MagicMock()
    detector.detect = MagicMock(return_value=detections or [_make_detection()])

    alert_engine = MagicMock()
    alert_engine.should_alert = MagicMock(return_value=should_alert)
    alert_engine.mark_alerted = MagicMock()
    alert_engine.direction_message = MagicMock(return_value="persona al frente")

    tts = MagicMock()
    tts.generate = AsyncMock(return_value=b"audio")

    audio = MagicMock()
    audio.speak = AsyncMock()

    return ProactiveMode(stream, detector, alert_engine, tts, audio, fps_limit=100.0)


# ---------------------------------------------------------------------------
# is_running state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_running_false_before_start() -> None:
    mode = _make_proactive()
    assert mode.is_running is False


@pytest.mark.asyncio
async def test_is_running_true_after_start() -> None:
    mode = _make_proactive()
    await mode.start()
    assert mode.is_running is True
    await mode.stop()


@pytest.mark.asyncio
async def test_is_running_false_after_stop() -> None:
    mode = _make_proactive()
    await mode.start()
    await mode.stop()
    assert mode.is_running is False


# ---------------------------------------------------------------------------
# start / stop lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_creates_background_task() -> None:
    mode = _make_proactive()
    await mode.start()
    assert mode._task is not None
    assert not mode._task.done()
    await mode.stop()


@pytest.mark.asyncio
async def test_stop_cancels_task_cleanly() -> None:
    mode = _make_proactive()
    await mode.start()
    await mode.stop()
    assert mode._task is None or mode._task.done()


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_high_confidence_detection_triggers_alert_and_speech() -> None:
    mode = _make_proactive(should_alert=True)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    await mode._process_frame(frame)

    mode._alert_engine.mark_alerted.assert_called_once()
    mode._tts.generate.assert_called_once()
    mode._audio.speak.assert_called_once()


@pytest.mark.asyncio
async def test_low_confidence_below_threshold_no_alert() -> None:
    low_conf = _make_detection("person", 0.1)
    mode = _make_proactive(detections=[low_conf], should_alert=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    await mode._process_frame(frame)

    mode._alert_engine.mark_alerted.assert_not_called()
    mode._tts.generate.assert_not_called()
    mode._audio.speak.assert_not_called()


@pytest.mark.asyncio
async def test_cooldown_respected_no_repeated_alert() -> None:
    mode = _make_proactive(should_alert=False)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    await mode._process_frame(frame)

    mode._alert_engine.mark_alerted.assert_not_called()
    mode._audio.speak.assert_not_called()


# ---------------------------------------------------------------------------
# Traffic light classifier wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_traffic_light_detection_calls_classifier() -> None:
    tl_detection = Detection(9, "traffic_light", 0.95, (100, 50, 150, 130))

    stream = MagicMock()
    stream.get_frame = AsyncMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))
    detector = MagicMock()
    detector.detect = MagicMock(return_value=[tl_detection])
    alert_engine = MagicMock()
    alert_engine.should_alert = MagicMock(return_value=True)
    alert_engine.mark_alerted = MagicMock()
    alert_engine.direction_message = MagicMock(return_value="semáforo al frente")
    tts = MagicMock()
    tts.generate = AsyncMock(return_value=b"audio")
    audio = MagicMock()
    audio.speak = AsyncMock()

    mode = ProactiveMode(stream, detector, alert_engine, tts, audio)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch("modes.proactive.classify_traffic_light", return_value="rojo") as mock_clf:
        await mode._process_frame(frame)
        mock_clf.assert_called_once_with(frame, tl_detection)

    tts.generate.assert_called_once()
    spoken_text: str = tts.generate.call_args[0][0]
    assert "rojo" in spoken_text


# ---------------------------------------------------------------------------
# Exception handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_handles_detector_exception_without_crashing() -> None:
    stream = MagicMock()
    stream.get_frame = AsyncMock(return_value=np.zeros((480, 640, 3), dtype=np.uint8))
    detector = MagicMock()
    detector.detect = MagicMock(side_effect=RuntimeError("NPU fallo"))
    alert_engine = MagicMock()
    tts = MagicMock()
    tts.generate = AsyncMock(return_value=b"")
    audio = MagicMock()
    audio.speak = AsyncMock()

    mode = ProactiveMode(stream, detector, alert_engine, tts, audio)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Should not raise
    await mode._process_frame(frame)
    alert_engine.mark_alerted.assert_not_called()


# ---------------------------------------------------------------------------
# _detection_priority helper
# ---------------------------------------------------------------------------


def test_priority_alert_for_traffic_light() -> None:
    det = Detection(9, "traffic_light", 0.9, (0, 0, 50, 100))
    assert _detection_priority(det) == PRIORITY_ALERT


def test_priority_alert_for_stop_sign() -> None:
    det = Detection(11, "stop_sign", 0.9, (0, 0, 50, 100))
    assert _detection_priority(det) == PRIORITY_ALERT


def test_priority_alert_for_stairs() -> None:
    det = Detection(99, "stairs", 0.9, (0, 0, 50, 100))
    assert _detection_priority(det) == PRIORITY_ALERT


def test_priority_normal_for_person() -> None:
    det = Detection(0, "person", 0.9, (0, 0, 50, 100))
    assert _detection_priority(det) == PRIORITY_NORMAL


def test_priority_normal_for_car() -> None:
    det = Detection(2, "car", 0.9, (0, 0, 50, 100))
    assert _detection_priority(det) == PRIORITY_NORMAL


def test_priority_low_for_other() -> None:
    det = Detection(50, "banana", 0.9, (0, 0, 50, 100))
    assert _detection_priority(det) == PRIORITY_LOW
