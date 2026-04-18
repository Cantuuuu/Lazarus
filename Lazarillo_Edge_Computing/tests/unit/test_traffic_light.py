"""Unit tests for core/traffic_light.py — HSV color classifier."""
from __future__ import annotations

import numpy as np
import pytest

from core.detector import Detection
from core.traffic_light import classify_traffic_light


def _solid_bgr_frame(b: int, g: int, r: int, h: int = 100, w: int = 50) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (b, g, r)
    return frame


def _detection_full_frame(h: int = 100, w: int = 50) -> Detection:
    return Detection(9, "traffic_light", 0.95, (0, 0, w, h))


class TestClassifyTrafficLight:
    def test_red_roi_returns_rojo(self) -> None:
        frame = _solid_bgr_frame(0, 0, 200)
        det = _detection_full_frame()
        assert classify_traffic_light(frame, det) == "rojo"

    def test_green_roi_returns_verde(self) -> None:
        # Pure green in BGR = (0, 200, 0) → HSV hue ≈ 60
        frame = _solid_bgr_frame(0, 200, 0)
        det = _detection_full_frame()
        assert classify_traffic_light(frame, det) == "verde"

    def test_black_roi_returns_desconocido(self) -> None:
        frame = _solid_bgr_frame(0, 0, 0)
        det = _detection_full_frame()
        assert classify_traffic_light(frame, det) == "desconocido"

    def test_empty_roi_returns_desconocido(self) -> None:
        frame = np.zeros((100, 50, 3), dtype=np.uint8)
        # bbox with zero area
        det = Detection(9, "traffic_light", 0.9, (10, 10, 10, 10))
        assert classify_traffic_light(frame, det) == "desconocido"

    def test_equal_red_green_pixels_returns_verde(self) -> None:
        """Green wins on ties (green_px >= red_px when equal)."""
        frame = np.zeros((100, 50, 3), dtype=np.uint8)
        # Top half: red (hue 0), bottom half: green (hue 60)
        frame[:50, :] = (0, 0, 200)   # BGR red
        frame[50:, :] = (0, 200, 0)   # BGR green
        det = _detection_full_frame()
        result = classify_traffic_light(frame, det)
        # Equal pixels — function must return "verde"
        assert result == "verde"

    def test_mostly_red_returns_rojo(self) -> None:
        frame = np.zeros((100, 50, 3), dtype=np.uint8)
        frame[:80, :] = (0, 0, 200)   # mostly red
        frame[80:, :] = (0, 200, 0)   # small green section
        det = _detection_full_frame()
        assert classify_traffic_light(frame, det) == "rojo"

    def test_mostly_green_returns_verde(self) -> None:
        frame = np.zeros((100, 50, 3), dtype=np.uint8)
        frame[:80, :] = (0, 200, 0)   # mostly green
        frame[80:, :] = (0, 0, 200)   # small red section
        det = _detection_full_frame()
        assert classify_traffic_light(frame, det) == "verde"
