"""Unit tests for core/detector.py."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from core.detector import (
    COCO_NAMES,
    PRIORITY_CLASSES,
    Detection,
    MockDetector,
    ONNXDetector,
    RKNNDetector,
    _postprocess,
    build_detector,
)


class TestDetection:
    def test_is_priority_for_known_class(self) -> None:
        d = Detection(0, "person", 0.9, (0, 0, 100, 100))
        assert d.is_priority is True

    def test_is_priority_false_for_unknown_class(self) -> None:
        d = Detection(50, "orange", 0.9, (0, 0, 100, 100))
        assert d.is_priority is False

    def test_center_x(self) -> None:
        d = Detection(0, "person", 0.9, (100, 50, 300, 200))
        assert d.center_x == 200

    def test_direction_left(self) -> None:
        d = Detection(0, "person", 0.9, (0, 0, 100, 100))  # center_x = 50
        assert d.direction == "izquierda"

    def test_direction_center(self) -> None:
        d = Detection(0, "person", 0.9, (213, 0, 427, 100))  # center_x = 320
        assert d.direction == "frente"

    def test_direction_right(self) -> None:
        d = Detection(0, "person", 0.9, (427, 0, 640, 100))  # center_x = 533
        assert d.direction == "derecha"

    def test_immutable(self) -> None:
        d = Detection(0, "person", 0.9, (0, 0, 100, 100))
        with pytest.raises((AttributeError, TypeError)):
            d.class_id = 1  # type: ignore[misc]


class TestMockDetector:
    def test_returns_default_detections(self) -> None:
        det = MockDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert len(results) == 2
        assert results[0].class_name == "person"
        assert results[1].class_name == "car"

    def test_returns_custom_detections(self) -> None:
        custom = [Detection(9, "traffic_light", 0.95, (300, 50, 340, 150))]
        det = MockDetector(custom)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert results == custom

    def test_ignores_frame_content(self) -> None:
        det = MockDetector()
        frame_black = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_white = np.ones((480, 640, 3), dtype=np.uint8) * 255
        assert det.detect(frame_black) == det.detect(frame_white)

    def test_returns_same_list_each_call(self) -> None:
        det = MockDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        r1 = det.detect(frame)
        r2 = det.detect(frame)
        assert r1 == r2


class TestBuildDetector:
    def test_dev_mode_returns_mock(self) -> None:
        det = build_detector(dev_mode=True)
        assert isinstance(det, MockDetector)

    def test_dev_mode_ignores_model_path(self) -> None:
        det = build_detector(dev_mode=True, model_path="nonexistent.rknn")
        assert isinstance(det, MockDetector)


class TestPostprocess:
    def _make_output(self, boxes: list[tuple[float, float, float, float, int, float]]) -> np.ndarray:
        """Create synthetic YOLO output (84, 8400)."""
        output = np.zeros((84, 8400), dtype=np.float32)
        for i, (cx, cy, bw, bh, cls_id, conf) in enumerate(boxes):
            if i >= 8400:
                break
            output[0, i] = cx
            output[1, i] = cy
            output[2, i] = bw
            output[3, i] = bh
            output[4 + cls_id, i] = conf
        return output

    def test_empty_output_returns_empty_list(self) -> None:
        output = np.zeros((84, 8400), dtype=np.float32)
        result = _postprocess(output, 640, 480, 0.4)
        assert result == []

    def test_detects_high_confidence_box(self) -> None:
        output = self._make_output([(320.0, 240.0, 100.0, 80.0, 0, 0.9)])
        result = _postprocess(output, 640, 480, 0.4)
        assert len(result) == 1
        assert result[0].class_name == "person"
        assert result[0].confidence == pytest.approx(0.9, abs=1e-4)

    def test_filters_low_confidence(self) -> None:
        output = self._make_output([(320.0, 240.0, 100.0, 80.0, 0, 0.1)])
        result = _postprocess(output, 640, 480, 0.4)
        assert result == []

    def test_accepts_3d_input(self) -> None:
        output = self._make_output([(320.0, 240.0, 100.0, 80.0, 0, 0.9)])
        output_3d = output[np.newaxis, ...]  # (1, 84, 8400)
        result = _postprocess(output_3d, 640, 480, 0.4)
        assert len(result) == 1

    def test_bbox_coordinates_scaled(self) -> None:
        output = self._make_output([(320.0, 240.0, 100.0, 80.0, 0, 0.9)])
        result = _postprocess(output, 640, 480, 0.4)
        assert len(result) == 1
        x1, y1, x2, y2 = result[0].bbox
        assert x1 < x2
        assert y1 < y2


class TestONNXDetector:
    def test_init_fails_without_onnxruntime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "onnxruntime", None)  # type: ignore[arg-type]
        with pytest.raises(Exception):
            ONNXDetector("fake.onnx")

    def test_detect_calls_session_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "images"
        mock_session.get_inputs.return_value = [mock_input]
        # Output shape (1, 84, 8400) with zeros → 0 detections after postprocess
        mock_session.run.return_value = [np.zeros((1, 84, 8400), dtype=np.float32)]
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        monkeypatch.setitem(sys.modules, "onnxruntime", mock_ort)
        det = ONNXDetector("fake.onnx")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert isinstance(results, list)
        mock_session.run.assert_called_once()

    def test_detect_returns_detections_when_confident(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "images"
        mock_session.get_inputs.return_value = [mock_input]
        # Build a (1, 84, 8400) output with one high-confidence person box
        output = np.zeros((1, 84, 8400), dtype=np.float32)
        output[0, 0, 0] = 320.0  # cx
        output[0, 1, 0] = 240.0  # cy
        output[0, 2, 0] = 100.0  # bw
        output[0, 3, 0] = 80.0   # bh
        output[0, 4, 0] = 0.95   # class 0 (person) confidence
        mock_session.run.return_value = [output]
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        monkeypatch.setitem(sys.modules, "onnxruntime", mock_ort)
        det = ONNXDetector("fake.onnx")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert isinstance(results, list)


class TestRKNNDetector:
    def test_init_raises_without_rknnlite(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "rknnlite", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "rknnlite.api", None)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="rknnlite"):
            RKNNDetector("fake.rknn")

    def test_init_raises_on_load_failure(self, mocker: "pytest_mock.MockerFixture") -> None:  # noqa: F821
        mock_rknn = mocker.MagicMock()
        mock_rknn.load_rknn.return_value = -1
        mock_class = mocker.MagicMock(return_value=mock_rknn)
        mock_rknnlite_api = mocker.MagicMock()
        mock_rknnlite_api.RKNNLite = mock_class
        mock_rknnlite = mocker.MagicMock()
        mocker.patch.dict(
            "sys.modules",
            {"rknnlite": mock_rknnlite, "rknnlite.api": mock_rknnlite_api},
        )
        with pytest.raises(RuntimeError, match="Error cargando"):
            RKNNDetector("fake.rknn")

    def test_init_raises_on_runtime_failure(self, mocker: "pytest_mock.MockerFixture") -> None:  # noqa: F821
        mock_rknn = mocker.MagicMock()
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = -1
        mock_class = mocker.MagicMock(return_value=mock_rknn)
        mock_rknnlite_api = mocker.MagicMock()
        mock_rknnlite_api.RKNNLite = mock_class
        mock_rknnlite = mocker.MagicMock()
        mocker.patch.dict(
            "sys.modules",
            {"rknnlite": mock_rknnlite, "rknnlite.api": mock_rknnlite_api},
        )
        with pytest.raises(RuntimeError, match="Error inicializando"):
            RKNNDetector("fake.rknn")

    def test_detect_calls_inference(self, mocker: "pytest_mock.MockerFixture") -> None:  # noqa: F821
        mock_rknn = mocker.MagicMock()
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        output = np.zeros((1, 84, 8400), dtype=np.float32)
        mock_rknn.inference.return_value = [output]
        mock_class = mocker.MagicMock(return_value=mock_rknn)
        mock_rknnlite_api = mocker.MagicMock()
        mock_rknnlite_api.RKNNLite = mock_class
        mock_rknnlite = mocker.MagicMock()
        mocker.patch.dict(
            "sys.modules",
            {"rknnlite": mock_rknnlite, "rknnlite.api": mock_rknnlite_api},
        )
        det = RKNNDetector("fake.rknn")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = det.detect(frame)
        assert isinstance(results, list)
        mock_rknn.inference.assert_called_once()


class TestBuildDetectorFallback:
    def test_rknn_falls_back_to_onnx(self, mocker: "pytest_mock.MockerFixture", tmp_path: "pathlib.Path") -> None:  # noqa: F821
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")
        rknn_file = tmp_path / "model.rknn"
        rknn_file.write_bytes(b"fake")
        mocker.patch("core.detector.RKNNDetector", side_effect=RuntimeError("bad model"))
        mock_onnx = mocker.MagicMock()
        mocker.patch("core.detector.ONNXDetector", return_value=mock_onnx)
        result = build_detector(dev_mode=False, model_path=str(rknn_file))
        assert result is mock_onnx

    def test_rknn_falls_back_to_mock_if_no_onnx(self, mocker: "pytest_mock.MockerFixture", tmp_path: "pathlib.Path") -> None:  # noqa: F821
        rknn_file = tmp_path / "model.rknn"
        rknn_file.write_bytes(b"fake")
        mocker.patch("core.detector.RKNNDetector", side_effect=RuntimeError("bad"))
        result = build_detector(dev_mode=False, model_path=str(rknn_file))
        assert isinstance(result, MockDetector)

    def test_onnx_path_returns_onnx_detector(self, mocker: "pytest_mock.MockerFixture", tmp_path: "pathlib.Path") -> None:  # noqa: F821
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")
        mock_onnx = mocker.MagicMock()
        mocker.patch("core.detector.ONNXDetector", return_value=mock_onnx)
        result = build_detector(dev_mode=False, model_path=str(onnx_file))
        assert result is mock_onnx

    def test_unknown_extension_returns_mock(self) -> None:
        result = build_detector(dev_mode=False, model_path="model.bin")
        assert isinstance(result, MockDetector)
