"""
tests/test_manual_trigger.py — Tests de ManualTrigger sin hardware real.

Se mockean pynput y CameraService para que los tests corran en CI
sin teclado ni cámara física.
"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from core.events import Event, EventType
from triggers.manual_trigger import ManualTrigger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(hotkey: str = "<ctrl>+<shift>+g") -> dict:
    return {"hotkey": hotkey}


def _make_camera(frame: np.ndarray | None = None) -> MagicMock:
    cam = MagicMock()
    cam.get_latest_frame.return_value = (
        frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    )
    cam.get_buffer_snapshot.return_value = []
    return cam


def _make_mock_listener():
    """Mock de keyboard.GlobalHotKeys que guarda el dict de hotkeys registradas."""
    listener = MagicMock()
    listener.daemon = False
    return listener


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestManualTriggerInit:
    def test_hotkey_default(self):
        cam = _make_camera()
        trigger = ManualTrigger(queue.Queue(), {}, cam)
        assert trigger._hotkey_str == "<ctrl>+<shift>+g"

    def test_hotkey_from_config(self):
        cam = _make_camera()
        trigger = ManualTrigger(queue.Queue(), {"hotkey": "<alt>+<shift>+c"}, cam)
        assert trigger._hotkey_str == "<alt>+<shift>+c"

    def test_not_running_before_start(self):
        cam = _make_camera()
        trigger = ManualTrigger(queue.Queue(), _make_config(), cam)
        assert not trigger._running.is_set()


class TestManualTriggerStartStop:
    def test_start_marca_running(self):
        cam = _make_camera()
        mock_listener = _make_mock_listener()

        with patch("triggers.manual_trigger.keyboard.GlobalHotKeys", return_value=mock_listener):
            trigger = ManualTrigger(queue.Queue(), _make_config(), cam)
            trigger.start()
            assert trigger._running.is_set()
            trigger.stop()

    def test_start_llama_listener_start(self):
        cam = _make_camera()
        mock_listener = _make_mock_listener()

        with patch("triggers.manual_trigger.keyboard.GlobalHotKeys", return_value=mock_listener):
            trigger = ManualTrigger(queue.Queue(), _make_config(), cam)
            trigger.start()
            mock_listener.start.assert_called_once()
            trigger.stop()

    def test_stop_marca_not_running(self):
        cam = _make_camera()
        mock_listener = _make_mock_listener()

        with patch("triggers.manual_trigger.keyboard.GlobalHotKeys", return_value=mock_listener):
            trigger = ManualTrigger(queue.Queue(), _make_config(), cam)
            trigger.start()
            trigger.stop()
            assert not trigger._running.is_set()

    def test_stop_sin_start_no_lanza(self):
        cam = _make_camera()
        trigger = ManualTrigger(queue.Queue(), _make_config(), cam)
        trigger.stop()  # no debe lanzar excepción


class TestManualTriggerOnHotkey:
    def test_on_hotkey_emite_evento_manual(self):
        q = queue.Queue()
        cam = _make_camera()
        mock_listener = _make_mock_listener()

        with patch("triggers.manual_trigger.keyboard.GlobalHotKeys", return_value=mock_listener):
            trigger = ManualTrigger(q, _make_config(), cam)
            trigger.start()
            trigger._on_hotkey()  # simulamos la pulsación
            trigger.stop()

        assert not q.empty()
        event: Event = q.get_nowait()
        assert event.event_type == EventType.MANUAL

    def test_on_hotkey_incluye_frame(self):
        q = queue.Queue()
        frame = np.full((480, 640, 3), 42, dtype=np.uint8)
        cam = _make_camera(frame=frame)
        mock_listener = _make_mock_listener()

        with patch("triggers.manual_trigger.keyboard.GlobalHotKeys", return_value=mock_listener):
            trigger = ManualTrigger(q, _make_config(), cam)
            trigger.start()
            trigger._on_hotkey()
            trigger.stop()

        event: Event = q.get_nowait()
        assert event.frame is not None
        assert event.frame.shape == (480, 640, 3)

    def test_on_hotkey_sin_camara_emite_igual(self):
        """Si la cámara no tiene frames, el evento se emite de todas formas."""
        q = queue.Queue()
        cam = _make_camera()
        cam.get_latest_frame.return_value = None  # sin frame disponible
        cam.get_buffer_snapshot.return_value = []
        mock_listener = _make_mock_listener()

        with patch("triggers.manual_trigger.keyboard.GlobalHotKeys", return_value=mock_listener):
            trigger = ManualTrigger(q, _make_config(), cam)
            trigger.start()
            trigger._on_hotkey()
            trigger.stop()

        event: Event = q.get_nowait()
        assert event.event_type == EventType.MANUAL
        assert event.frame is None

    def test_on_hotkey_metadata_vacia(self):
        q = queue.Queue()
        cam = _make_camera()
        mock_listener = _make_mock_listener()

        with patch("triggers.manual_trigger.keyboard.GlobalHotKeys", return_value=mock_listener):
            trigger = ManualTrigger(q, _make_config(), cam)
            trigger.start()
            trigger._on_hotkey()
            trigger.stop()

        event: Event = q.get_nowait()
        assert event.metadata == {}

    def test_multiples_pulsaciones_emiten_multiples_eventos(self):
        q = queue.Queue()
        cam = _make_camera()
        mock_listener = _make_mock_listener()

        with patch("triggers.manual_trigger.keyboard.GlobalHotKeys", return_value=mock_listener):
            trigger = ManualTrigger(q, _make_config(), cam)
            trigger.start()
            trigger._on_hotkey()
            trigger._on_hotkey()
            trigger._on_hotkey()
            trigger.stop()

        assert q.qsize() == 3
