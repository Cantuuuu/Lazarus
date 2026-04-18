"""
tests/test_yolo_trigger.py — Tests de YoloTrigger con las 6 mejoras.

Se mockean YoloOnnx y CameraService para aislar la logica.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.events import Event, EventType
from triggers.yolo_trigger import YoloTrigger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    whitelist=None,
    cooldown=5.0,
    stability_frames=3,
    stability_window=5,
    conf=0.6,
    analysis_fps=10,
    **overrides,
) -> dict:
    cfg = {
        "model_path": "models/yolov8n.onnx",
        "confidence_threshold": conf,
        "analysis_fps": analysis_fps,
        "cooldown_seconds": cooldown,
        "stability_frames": stability_frames,
        "stability_window": stability_window,
        "class_whitelist": whitelist or ["person", "car", "bicycle"],
        # Mejora 2: prioridades
        "priority_cooldowns": {"urgent": 2, "important": 5, "informative": 15},
        "class_priority": {
            "car": "urgent",
            "bicycle": "urgent",
            "person": "important",
        },
        # Mejora 3: scene change
        "scene_change_enabled": False,
        "scene_change_threshold": 0.45,
        "scene_change_cooldown": 10,
        # Mejora 4: count change
        "count_change_enabled": False,
        "count_change_delta": 2,
        "count_change_cooldown": 10,
        # Mejora 5: summary
        "summary_enabled": False,
        "summary_interval_seconds": 30,
        # Mejora 6: context frames
        "context_frames_count": 5,
    }
    cfg.update(overrides)
    return cfg


def _make_camera(frame=None, buffer_size=10) -> MagicMock:
    cam = MagicMock()
    f = frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    cam.get_latest_frame.return_value = f
    cam.get_buffer_snapshot.return_value = [
        np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(buffer_size)
    ]
    return cam


def _det(class_name: str, conf: float = 0.9, bbox=None) -> dict:
    return {
        "class_name": class_name,
        "confidence": conf,
        "bbox": bbox or [100, 100, 200, 200],
    }


def _make_trigger(config=None, camera=None, q=None) -> YoloTrigger:
    config = config or _make_config()
    camera = camera or _make_camera()
    q = q or queue.Queue()
    trigger = YoloTrigger(q, config, camera)
    trigger._model = MagicMock()
    return trigger


# ---------------------------------------------------------------------------
# Mejora 1: Spatial Metadata
# ---------------------------------------------------------------------------

class TestSpatialMetadata:
    def test_zone_left(self):
        dets = [_det("person", bbox=[0, 0, 100, 200])]  # center_x = 50
        YoloTrigger._enrich_spatial(dets, 480, 640)
        assert dets[0]["zone"] == "left"

    def test_zone_center(self):
        dets = [_det("person", bbox=[250, 0, 400, 200])]  # center_x = 325
        YoloTrigger._enrich_spatial(dets, 480, 640)
        assert dets[0]["zone"] == "center"

    def test_zone_right(self):
        dets = [_det("person", bbox=[500, 0, 640, 200])]  # center_x = 570
        YoloTrigger._enrich_spatial(dets, 480, 640)
        assert dets[0]["zone"] == "right"

    def test_proximity_near(self):
        # area = 400*400 = 160000, frame = 480*640 = 307200, ratio = 0.52
        dets = [_det("person", bbox=[0, 0, 400, 400])]
        YoloTrigger._enrich_spatial(dets, 480, 640)
        assert dets[0]["proximity"] == "near"

    def test_proximity_medium(self):
        # area = 100*150 = 15000, ratio = 15000/307200 = 0.049
        dets = [_det("person", bbox=[100, 100, 200, 250])]
        YoloTrigger._enrich_spatial(dets, 480, 640)
        assert dets[0]["proximity"] == "medium"

    def test_proximity_far(self):
        # area = 30*30 = 900, ratio = 900/307200 = 0.003
        dets = [_det("person", bbox=[100, 100, 130, 130])]
        YoloTrigger._enrich_spatial(dets, 480, 640)
        assert dets[0]["proximity"] == "far"

    def test_preserves_existing_keys(self):
        dets = [_det("person", bbox=[0, 0, 100, 200])]
        YoloTrigger._enrich_spatial(dets, 480, 640)
        assert "class_name" in dets[0]
        assert "confidence" in dets[0]
        assert "bbox" in dets[0]

    def test_multiple_detections(self):
        dets = [
            _det("person", bbox=[0, 0, 50, 50]),        # left, far
            _det("car", bbox=[300, 100, 400, 400]),      # center, medium/near
            _det("bicycle", bbox=[550, 0, 640, 480]),    # right, near
        ]
        YoloTrigger._enrich_spatial(dets, 480, 640)
        assert dets[0]["zone"] == "left"
        assert dets[1]["zone"] == "center"
        assert dets[2]["zone"] == "right"


# ---------------------------------------------------------------------------
# Mejora 2: Priority System
# ---------------------------------------------------------------------------

class TestPrioritySystem:
    def test_urgent_class_gets_urgent_priority(self):
        trigger = _make_trigger()
        assert trigger._get_priority_for_class("car") == "urgent"

    def test_important_class_gets_important_priority(self):
        trigger = _make_trigger()
        assert trigger._get_priority_for_class("person") == "important"

    def test_unknown_class_defaults_to_informative(self):
        trigger = _make_trigger()
        assert trigger._get_priority_for_class("bench") == "informative"

    def test_urgent_cooldown_is_2s(self):
        trigger = _make_trigger()
        assert trigger._get_cooldown_for_class("car") == 2

    def test_important_cooldown_is_5s(self):
        trigger = _make_trigger()
        assert trigger._get_cooldown_for_class("person") == 5

    def test_informative_cooldown_is_15s(self):
        trigger = _make_trigger()
        assert trigger._get_cooldown_for_class("bench") == 15

    def test_priority_in_heuristics_uses_class_cooldown(self):
        """Car (urgent, 2s) should fire after 2s even though default cooldown is 5s."""
        trigger = _make_trigger(_make_config(stability_frames=1))
        trigger._stability_window.append({"car"})
        trigger._last_classes = set()
        trigger._cooldown_map["car"] = time.time() - 3.0  # 3s ago > 2s urgent cooldown

        result = trigger._evaluate_heuristics({"car"})
        assert "car" in result

    def test_priority_blocks_within_cooldown(self):
        """Car (urgent, 2s) should NOT fire within 2s."""
        trigger = _make_trigger(_make_config(stability_frames=1))
        trigger._stability_window.append({"car"})
        trigger._last_classes = set()
        trigger._cooldown_map["car"] = time.time() - 1.0  # 1s ago < 2s urgent cooldown

        result = trigger._evaluate_heuristics({"car"})
        assert "car" not in result


# ---------------------------------------------------------------------------
# Mejora 3: Scene Change
# ---------------------------------------------------------------------------

class TestSceneChange:
    def test_first_frame_no_event(self):
        trigger = _make_trigger()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = trigger._check_scene_change(frame, [])
        assert result is False
        assert trigger._last_histogram is not None

    def test_identical_frames_no_event(self):
        trigger = _make_trigger()
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        trigger._check_scene_change(frame, [])  # set baseline
        result = trigger._check_scene_change(frame, [])
        assert result is False

    def test_very_different_frames_fires_event(self):
        q = queue.Queue()
        config = _make_config(scene_change_enabled=True, scene_change_cooldown=0)
        trigger = _make_trigger(config=config, q=q)

        # Frames con colores HSV muy distintos para forzar alta distancia Bhattacharyya
        # Rojo puro (H=0) vs verde puro (H=60 en escala OpenCV)
        red_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        red_frame[:, :, 2] = 255  # BGR: R=255

        green_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        green_frame[:, :, 1] = 255  # BGR: G=255

        trigger._check_scene_change(red_frame, [])  # baseline
        result = trigger._check_scene_change(green_frame, [])
        assert result is True
        assert not q.empty()

        event = q.get_nowait()
        assert event.metadata["trigger_class"] == "scene_change"

    def test_scene_change_respects_cooldown(self):
        q = queue.Queue()
        config = _make_config(scene_change_enabled=True, scene_change_cooldown=60)
        trigger = _make_trigger(config=config, q=q)

        red_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        red_frame[:, :, 2] = 255
        green_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        green_frame[:, :, 1] = 255
        blue_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        blue_frame[:, :, 0] = 255

        trigger._check_scene_change(red_frame, [])
        trigger._check_scene_change(green_frame, [])  # fires, sets cooldown
        trigger._check_scene_change(blue_frame, [])    # should NOT fire (cooldown)

        assert q.qsize() == 1  # only one event


# ---------------------------------------------------------------------------
# Mejora 4: Count Change
# ---------------------------------------------------------------------------

class TestCountChange:
    def test_no_change_no_event(self):
        trigger = _make_trigger()
        trigger._last_class_counts = {"person": 2}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = trigger._check_count_change({"person": 2}, [], frame)
        assert result is False

    def test_small_change_no_event(self):
        """Delta of 1 should NOT fire with default delta=2."""
        trigger = _make_trigger()
        trigger._last_class_counts = {"person": 2}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = trigger._check_count_change({"person": 3}, [], frame)
        assert result is False

    def test_significant_increase_fires_event(self):
        q = queue.Queue()
        trigger = _make_trigger(q=q)
        trigger._last_class_counts = {"person": 1}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = trigger._check_count_change({"person": 4}, [], frame)
        assert result is True

        event = q.get_nowait()
        assert event.metadata["trigger_class"] == "count_change"
        assert event.metadata["changed_class"] == "person"
        assert event.metadata["prev_count"] == 1
        assert event.metadata["new_count"] == 4

    def test_significant_decrease_fires_event(self):
        q = queue.Queue()
        trigger = _make_trigger(q=q)
        trigger._last_class_counts = {"person": 5}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = trigger._check_count_change({"person": 2}, [], frame)
        assert result is True

    def test_new_class_appearing_with_delta(self):
        """0 -> 3 persons should fire (delta=3 >= 2)."""
        q = queue.Queue()
        trigger = _make_trigger(q=q)
        trigger._last_class_counts = {}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = trigger._check_count_change({"person": 3}, [], frame)
        assert result is True

    def test_count_change_respects_cooldown(self):
        q = queue.Queue()
        trigger = _make_trigger(q=q)
        trigger._last_class_counts = {"person": 1}
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        trigger._check_count_change({"person": 5}, [], frame)  # fires
        trigger._last_class_counts = {"person": 5}
        trigger._check_count_change({"person": 1}, [], frame)  # cooldown

        assert q.qsize() == 1  # only one


# ---------------------------------------------------------------------------
# Mejora 5: Periodic Summary
# ---------------------------------------------------------------------------

class TestPeriodicSummary:
    def test_fires_after_interval(self):
        q = queue.Queue()
        trigger = _make_trigger(q=q)
        trigger._last_summary_time = time.time() - 60  # 60s ago
        trigger._summary_interval = 30
        dets = [_det("person")]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = trigger._check_summary(frame, dets)
        assert result is True

        event = q.get_nowait()
        assert event.metadata["trigger_class"] == "scene_summary"

    def test_no_fire_before_interval(self):
        trigger = _make_trigger()
        trigger._last_summary_time = time.time()
        trigger._summary_interval = 30
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = trigger._check_summary(frame, [_det("person")])
        assert result is False

    def test_no_fire_without_detections(self):
        trigger = _make_trigger()
        trigger._last_summary_time = time.time() - 60
        trigger._summary_interval = 30
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = trigger._check_summary(frame, [])
        assert result is False


# ---------------------------------------------------------------------------
# Mejora 6: Context Frames
# ---------------------------------------------------------------------------

class TestContextFrames:
    def test_select_5_from_large_buffer(self):
        cam = _make_camera(buffer_size=100)
        trigger = _make_trigger(camera=cam)
        trigger._context_frames_count = 5
        frames = trigger._select_context_frames()
        assert len(frames) == 5

    def test_select_all_from_small_buffer(self):
        cam = _make_camera(buffer_size=3)
        trigger = _make_trigger(camera=cam)
        trigger._context_frames_count = 5
        frames = trigger._select_context_frames()
        assert len(frames) == 3

    def test_empty_buffer_returns_empty(self):
        cam = _make_camera(buffer_size=0)
        cam.get_buffer_snapshot.return_value = []
        trigger = _make_trigger(camera=cam)
        frames = trigger._select_context_frames()
        assert frames == []

    def test_fire_event_includes_context_and_priority(self):
        q = queue.Queue()
        cam = _make_camera(buffer_size=10)
        trigger = _make_trigger(q=q, camera=cam)
        trigger._context_frames_count = 3
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        trigger._fire_event(frame, "person", [_det("person")])

        event = q.get_nowait()
        assert event.metadata["priority"] == "important"
        assert event.metadata["context_frame_count"] == 3
        assert event.frame_buffer is not None
        assert len(event.frame_buffer) == 3


# ---------------------------------------------------------------------------
# Heuristicas base (no deben haberse roto)
# ---------------------------------------------------------------------------

class TestBaseHeuristics:
    def test_clase_nueva_puede_disparar(self):
        trigger = _make_trigger(_make_config(cooldown=0, stability_frames=1))
        trigger._stability_window.append({"person"})
        trigger._last_classes = set()
        result = trigger._evaluate_heuristics({"person"})
        assert "person" in result

    def test_clase_ya_vista_no_dispara(self):
        trigger = _make_trigger(_make_config(cooldown=0, stability_frames=1))
        trigger._stability_window.append({"person"})
        trigger._last_classes = {"person"}
        result = trigger._evaluate_heuristics({"person"})
        assert "person" not in result

    def test_clase_inestable_no_dispara(self):
        trigger = _make_trigger(_make_config(cooldown=0, stability_frames=3, stability_window=5))
        trigger._last_classes = set()
        for s in [{"person"}, set(), {"person"}, set(), set()]:
            trigger._stability_window.append(s)
        result = trigger._evaluate_heuristics({"person"})
        assert "person" not in result


# ---------------------------------------------------------------------------
# Integracion (thread real, modelo mockeado)
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_detecta_clase_nueva_con_spatial_y_priority(self):
        q = queue.Queue()
        config = _make_config(
            whitelist=["person"],
            cooldown=0,
            stability_frames=1,
            stability_window=1,
            analysis_fps=30,
        )
        cam = _make_camera()
        trigger = YoloTrigger(q, config, cam)
        trigger._model = MagicMock()
        trigger._model.detect.return_value = [
            _det("person", bbox=[300, 100, 400, 400])
        ]

        trigger._stop_event.clear()
        trigger._thread = threading.Thread(
            target=trigger._analysis_loop, daemon=True
        )
        trigger._thread.start()
        time.sleep(0.3)
        trigger.stop()

        assert not q.empty()
        event = q.get_nowait()
        assert event.event_type == EventType.YOLO
        assert event.metadata["priority"] == "important"
        # Spatial metadata should be present in detections
        det = event.metadata["detections"][0]
        assert "zone" in det
        assert "proximity" in det
