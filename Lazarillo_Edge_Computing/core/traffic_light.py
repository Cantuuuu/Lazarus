"""Clasificador de semáforos por color HSV."""
from __future__ import annotations

import cv2
import numpy as np

from core.detector import Detection

_RED_LOWER_1 = (0, 100, 100)
_RED_UPPER_1 = (10, 255, 255)
_RED_LOWER_2 = (160, 100, 100)
_RED_UPPER_2 = (180, 255, 255)
_GREEN_LOWER = (40, 100, 100)
_GREEN_UPPER = (90, 255, 255)


def classify_traffic_light(frame: np.ndarray, detection: Detection) -> str:
    """Retorna 'rojo', 'verde', o 'desconocido' según el color dominante en el ROI."""
    x1, y1, x2, y2 = detection.bbox
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "desconocido"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, _RED_LOWER_1, _RED_UPPER_1) | cv2.inRange(
        hsv, _RED_LOWER_2, _RED_UPPER_2
    )
    green_mask = cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)

    red_px = int(cv2.countNonZero(red_mask))
    green_px = int(cv2.countNonZero(green_mask))

    if red_px == 0 and green_px == 0:
        return "desconocido"

    return "rojo" if red_px > green_px else "verde"
