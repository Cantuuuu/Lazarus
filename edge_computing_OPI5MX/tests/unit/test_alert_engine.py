"""Tests para AlertEngine — TDD fase RED antes de implementar core/alert_engine.py."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from core.detector import Detection
from core.alert_engine import AlertEngine, DEFAULT_COOLDOWN, ALERT_COOLDOWNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_detection(class_name: str, cx: int = 320) -> Detection:
    """Crea una Detection mínima con center_x controlado."""
    half = 40
    return Detection(
        class_id=0,
        class_name=class_name,
        confidence=0.9,
        bbox=(cx - half, 100, cx + half, 300),
    )


PERSON_FRONT = make_detection("person", cx=320)   # frente (213-427)
CAR_LEFT = make_detection("car", cx=100)           # izquierda (<213)
CAR_RIGHT = make_detection("car", cx=500)          # derecha (≥427)
UNKNOWN_FRONT = make_detection("bench", cx=320)    # clase no prioritaria


# ---------------------------------------------------------------------------
# should_alert: primer aviso
# ---------------------------------------------------------------------------

class TestShouldAlertFirstDetection:
    def test_returns_true_on_first_detection(self) -> None:
        engine = AlertEngine()
        assert engine.should_alert(PERSON_FRONT) is True

    def test_returns_true_for_different_class_on_first_detection(self) -> None:
        engine = AlertEngine()
        assert engine.should_alert(CAR_LEFT) is True

    def test_returns_true_for_unknown_class(self) -> None:
        engine = AlertEngine()
        assert engine.should_alert(UNKNOWN_FRONT) is True


# ---------------------------------------------------------------------------
# should_alert: cooldown activo
# ---------------------------------------------------------------------------

class TestShouldAlertCooldown:
    def test_returns_false_within_cooldown(self) -> None:
        engine = AlertEngine()
        engine.mark_alerted(PERSON_FRONT)

        with patch("core.alert_engine.time.monotonic", return_value=1.0):
            assert engine.should_alert(PERSON_FRONT) is False

    def test_returns_true_after_cooldown_expires(self) -> None:
        engine = AlertEngine()
        cooldown = ALERT_COOLDOWNS.get("person", DEFAULT_COOLDOWN)

        with patch("core.alert_engine.time.monotonic", return_value=0.0):
            engine.mark_alerted(PERSON_FRONT)

        with patch("core.alert_engine.time.monotonic", return_value=cooldown + 0.1):
            assert engine.should_alert(PERSON_FRONT) is True

    def test_returns_false_exactly_at_cooldown_boundary(self) -> None:
        engine = AlertEngine()
        cooldown = ALERT_COOLDOWNS.get("person", DEFAULT_COOLDOWN)

        with patch("core.alert_engine.time.monotonic", return_value=0.0):
            engine.mark_alerted(PERSON_FRONT)

        with patch("core.alert_engine.time.monotonic", return_value=cooldown - 0.01):
            assert engine.should_alert(PERSON_FRONT) is False


# ---------------------------------------------------------------------------
# should_alert: cooldowns independientes por clase
# ---------------------------------------------------------------------------

class TestCooldownPerClass:
    def test_car_and_person_have_independent_cooldowns(self) -> None:
        engine = AlertEngine()
        engine.mark_alerted(PERSON_FRONT)

        # car nunca fue alertado — debe permitir alerta
        assert engine.should_alert(CAR_LEFT) is True

    def test_alerting_one_class_does_not_block_another(self) -> None:
        engine = AlertEngine()
        engine.mark_alerted(CAR_LEFT)
        engine.mark_alerted(CAR_RIGHT)

        # persona no ha sido alertada
        assert engine.should_alert(PERSON_FRONT) is True

    def test_same_class_name_shares_cooldown_regardless_of_position(self) -> None:
        """Dos detecciones del mismo class_name comparten el mismo cooldown."""
        engine = AlertEngine()

        with patch("core.alert_engine.time.monotonic", return_value=0.0):
            engine.mark_alerted(CAR_LEFT)

        with patch("core.alert_engine.time.monotonic", return_value=1.0):
            assert engine.should_alert(CAR_RIGHT) is False


# ---------------------------------------------------------------------------
# should_alert: clases sin prioridad usan cooldown por defecto
# ---------------------------------------------------------------------------

class TestDefaultCooldown:
    def test_unknown_class_uses_default_cooldown(self) -> None:
        engine = AlertEngine()

        with patch("core.alert_engine.time.monotonic", return_value=0.0):
            engine.mark_alerted(UNKNOWN_FRONT)

        with patch("core.alert_engine.time.monotonic", return_value=DEFAULT_COOLDOWN - 0.5):
            assert engine.should_alert(UNKNOWN_FRONT) is False

    def test_unknown_class_allowed_after_default_cooldown(self) -> None:
        engine = AlertEngine()

        with patch("core.alert_engine.time.monotonic", return_value=0.0):
            engine.mark_alerted(UNKNOWN_FRONT)

        with patch("core.alert_engine.time.monotonic", return_value=DEFAULT_COOLDOWN + 0.1):
            assert engine.should_alert(UNKNOWN_FRONT) is True


# ---------------------------------------------------------------------------
# direction_message: mensajes en español
# ---------------------------------------------------------------------------

class TestDirectionMessage:
    def test_person_at_front(self) -> None:
        engine = AlertEngine()
        msg = engine.direction_message(PERSON_FRONT)
        assert msg == "persona al frente"

    def test_car_at_left(self) -> None:
        engine = AlertEngine()
        msg = engine.direction_message(CAR_LEFT)
        assert msg == "auto a la izquierda"

    def test_car_at_right(self) -> None:
        engine = AlertEngine()
        msg = engine.direction_message(CAR_RIGHT)
        assert msg == "auto a la derecha"

    def test_unknown_class_uses_class_name_as_fallback(self) -> None:
        engine = AlertEngine()
        msg = engine.direction_message(UNKNOWN_FRONT)
        assert "bench" in msg or "frente" in msg

    def test_traffic_light_translation(self) -> None:
        engine = AlertEngine()
        light = make_detection("traffic_light", cx=320)
        msg = engine.direction_message(light)
        assert msg == "semáforo al frente"

    def test_bicycle_translation(self) -> None:
        engine = AlertEngine()
        bike = make_detection("bicycle", cx=100)
        msg = engine.direction_message(bike)
        assert msg == "bicicleta a la izquierda"


# ---------------------------------------------------------------------------
# mark_alerted: idempotencia y actualización de timestamp
# ---------------------------------------------------------------------------

class TestMarkAlerted:
    def test_mark_alerted_updates_timestamp(self) -> None:
        engine = AlertEngine()

        with patch("core.alert_engine.time.monotonic", return_value=0.0):
            engine.mark_alerted(PERSON_FRONT)

        cooldown = ALERT_COOLDOWNS.get("person", DEFAULT_COOLDOWN)

        # Después del cooldown, volvemos a alertar para resetear el reloj
        with patch("core.alert_engine.time.monotonic", return_value=cooldown + 1.0):
            assert engine.should_alert(PERSON_FRONT) is True
            engine.mark_alerted(PERSON_FRONT)

        # Ahora el cooldown corre desde cooldown+1.0 — debe bloquearse de nuevo
        with patch("core.alert_engine.time.monotonic", return_value=cooldown + 2.0):
            assert engine.should_alert(PERSON_FRONT) is False
