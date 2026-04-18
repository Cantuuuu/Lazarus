"""Motor de alertas — decide cuándo generar un aviso de voz por detección."""
from __future__ import annotations

import time
from core.detector import Detection

ALERT_COOLDOWNS: dict[str, float] = {
    "car": 5.0,
    "person": 8.0,
    "traffic_light": 3.0,
    "dog": 10.0,
    "stairs": 3.0,
    "bicycle": 5.0,
    "bus": 5.0,
    "truck": 5.0,
    "motorcycle": 5.0,
    "stop_sign": 5.0,
}

DEFAULT_COOLDOWN = 5.0

DIRECTION_LABELS: dict[str, str] = {
    "izquierda": "a la izquierda",
    "frente": "al frente",
    "derecha": "a la derecha",
}

CLASS_LABELS_ES: dict[str, str] = {
    "person": "persona",
    "car": "auto",
    "bicycle": "bicicleta",
    "motorcycle": "moto",
    "bus": "camión",
    "truck": "camión",
    "traffic_light": "semáforo",
    "stop_sign": "señal de alto",
    "dog": "perro",
    "cat": "gato",
}


class AlertEngine:
    """Gestiona cooldowns por clase y genera mensajes de alerta en español."""

    def __init__(self) -> None:
        self._last_alerted: dict[str, float] = {}

    def should_alert(self, detection: Detection) -> bool:
        name = detection.class_name
        cooldown = ALERT_COOLDOWNS.get(name, DEFAULT_COOLDOWN)
        last = self._last_alerted.get(name)
        if last is None:
            return True
        return (time.monotonic() - last) >= cooldown

    def mark_alerted(self, detection: Detection) -> None:
        self._last_alerted = {**self._last_alerted, detection.class_name: time.monotonic()}

    def direction_message(self, detection: Detection) -> str:
        label = CLASS_LABELS_ES.get(detection.class_name, detection.class_name)
        direction = DIRECTION_LABELS.get(detection.direction, detection.direction)
        return f"{label} {direction}"
