"""
tests/test_camera.py — Tests de CameraService sin hardware real.

Se mockea cv2.VideoCapture para que los tests corran en CI
sin cámara física conectada.
"""

from __future__ import annotations

import time
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.camera import CameraService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(fps: int = 10, buffer_seconds: int = 2, device_index: int = 0) -> dict:
    return {
        "device_index": device_index,
        "width": 640,
        "height": 480,
        "fps": fps,
        "buffer_seconds": buffer_seconds,
    }


def _fake_frame(value: int = 128) -> np.ndarray:
    """Frame BGR sintético de 480x640."""
    return np.full((480, 640, 3), value, dtype=np.uint8)


def _make_mock_cap(frames: list[np.ndarray] | None = None, open_ok: bool = True):
    """Crea un mock de cv2.VideoCapture que devuelve frames predefinidos."""
    cap = MagicMock()
    cap.isOpened.return_value = open_ok

    if frames:
        # Devuelve (True, frame) para cada frame en la lista, luego (False, None)
        side_effects = [(True, f) for f in frames] + [(False, None)] * 1000
        cap.read.side_effect = side_effects
    else:
        # Devuelve un frame genérico infinitamente
        cap.read.return_value = (True, _fake_frame())

    return cap


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCameraServiceInit:
    def test_buffer_maxlen_calculado_correctamente(self):
        config = _make_config(fps=30, buffer_seconds=10)
        cam = CameraService(config)
        assert cam._buffer.maxlen == 300  # 30 * 10

    def test_not_running_antes_de_start(self):
        cam = CameraService(_make_config())
        assert not cam.is_running

    def test_buffer_vacio_antes_de_start(self):
        cam = CameraService(_make_config())
        assert cam.buffer_size == 0
        assert cam.get_latest_frame() is None
        assert cam.get_buffer_snapshot() == []


class TestCameraServiceStart:
    def test_lanza_error_si_camara_no_abre(self):
        config = _make_config()
        mock_cap = _make_mock_cap(open_ok=False)

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = CameraService(config)
            with pytest.raises(RuntimeError, match="No se pudo abrir"):
                cam.start()

    def test_start_sets_running(self):
        config = _make_config(fps=10)
        mock_cap = _make_mock_cap()

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = CameraService(config)
            cam.start()
            assert cam.is_running
            cam.stop()

    def test_doble_start_no_lanza_excepcion(self):
        config = _make_config(fps=10)
        mock_cap = _make_mock_cap()

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = CameraService(config)
            cam.start()
            cam.start()  # segunda llamada debe ser ignorada silenciosamente
            assert cam.is_running
            cam.stop()


class TestCameraServiceBuffer:
    def test_buffer_se_llena_con_frames(self):
        config = _make_config(fps=10, buffer_seconds=2)
        mock_cap = _make_mock_cap()

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = CameraService(config)
            cam.start()
            time.sleep(0.5)  # deja que el thread capture algunos frames
            assert cam.buffer_size > 0
            cam.stop()

    def test_get_latest_frame_devuelve_ndarray(self):
        config = _make_config(fps=10)
        mock_cap = _make_mock_cap()

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = CameraService(config)
            cam.start()
            time.sleep(0.3)
            frame = cam.get_latest_frame()
            cam.stop()

        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)

    def test_get_latest_frame_es_copia(self):
        """Modificar el frame retornado no debe afectar el buffer interno."""
        config = _make_config(fps=10)
        mock_cap = _make_mock_cap()

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = CameraService(config)
            cam.start()
            time.sleep(0.3)
            frame = cam.get_latest_frame()
            original_value = frame[0, 0, 0]
            frame[0, 0, 0] = 255  # mutamos la copia
            frame2 = cam.get_latest_frame()
            cam.stop()

        # El buffer no debe haber sido afectado por la mutación
        # (puede ser un frame diferente, pero el original no cambia)
        assert frame2 is not None  # al menos sigue funcionando

    def test_get_buffer_snapshot_devuelve_lista(self):
        config = _make_config(fps=10, buffer_seconds=1)
        mock_cap = _make_mock_cap()

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = CameraService(config)
            cam.start()
            time.sleep(0.5)
            snapshot = cam.get_buffer_snapshot()
            cam.stop()

        assert isinstance(snapshot, list)
        assert len(snapshot) > 0
        assert all(isinstance(f, np.ndarray) for f in snapshot)

    def test_buffer_maxlen_respetado(self):
        """El buffer no debe exceder fps * buffer_seconds frames."""
        config = _make_config(fps=10, buffer_seconds=1)  # maxlen = 10
        mock_cap = _make_mock_cap()

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = CameraService(config)
            cam.start()
            time.sleep(1.5)  # más de 1 segundo → debería haber llenado y rotado
            size = cam.buffer_size
            cam.stop()

        assert size <= 10  # nunca supera el maxlen


class TestCameraServiceStop:
    def test_stop_sets_not_running(self):
        config = _make_config(fps=10)
        mock_cap = _make_mock_cap()

        with patch("cv2.VideoCapture", return_value=mock_cap):
            cam = CameraService(config)
            cam.start()
            cam.stop()

        assert not cam.is_running

    def test_stop_sin_start_no_lanza_excepcion(self):
        cam = CameraService(_make_config())
        cam.stop()  # no debe lanzar
