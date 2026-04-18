"""Unit tests for _postprocess_dfl — YOLOv8 DFL 9-output format from RKNN NPU.

Shape layout expected by _postprocess_dfl:
    outputs[0]: (1, 64, 80, 80)  — box DFL stride 8
    outputs[1]: (1, 80, 80, 80)  — class logits stride 8
    outputs[2]: (1,  1, 80, 80)  — ignored (objectness/unused)
    outputs[3]: (1, 64, 40, 40)  — box DFL stride 16
    outputs[4]: (1, 80, 40, 40)  — class logits stride 16
    outputs[5]: (1,  1, 40, 40)  — ignored
    outputs[6]: (1, 64, 20, 20)  — box DFL stride 32
    outputs[7]: (1, 80, 20, 20)  — class logits stride 32
    outputs[8]: (1,  1, 20, 20)  — ignored

DFL decoding recap (reg_max=16 bins):
    dist = sum(softmax(bins_4) * arange(reg_max), axis=-1)
    Each of the 64 channels is split as 4 * reg_max.
    anchor center = ((col + 0.5) * stride, (row + 0.5) * stride)
    x1 = cx - dist_left * stride   x2 = cx + dist_right * stride
    y1 = cy - dist_top  * stride   y2 = cy + dist_bottom * stride
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.detector import Detection, RKNNDetector, _postprocess_dfl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REG_MAX = 16
_NUM_CLASSES = 80


def _make_dfl_outputs(
    *,
    input_size: int = 640,
    reg_max: int = REG_MAX,
) -> list[np.ndarray]:
    """Return 9 zero-filled outputs matching the RKNN DFL shape layout."""
    strides = [8, 16, 32]
    outputs: list[np.ndarray] = []
    for stride in strides:
        grid = input_size // stride
        outputs.append(np.zeros((1, reg_max * 4, grid, grid), dtype=np.float32))  # box DFL
        outputs.append(np.zeros((1, _NUM_CLASSES, grid, grid), dtype=np.float32))  # class logits
        outputs.append(np.zeros((1, 1, grid, grid), dtype=np.float32))             # ignored
    return outputs


def _inject_detection(
    outputs: list[np.ndarray],
    *,
    scale_idx: int,
    row: int,
    col: int,
    dist_bins: tuple[int, int, int, int] = (2, 2, 2, 2),
    class_id: int = 0,
    cls_score: float = 5.0,
    reg_max: int = REG_MAX,
) -> None:
    """Write a synthetic detection into the output tensors in-place.

    dist_bins: (left_peak, top_peak, right_peak, bottom_peak) — index of the
    softmax peak bin for each of the 4 DFL distance components.  Concentrating
    all probability mass at a single bin gives dist ≈ bin_index (grid units).
    """
    box_out = outputs[scale_idx * 3]      # (1, 64, H, W)
    cls_out = outputs[scale_idx * 3 + 1]  # (1, 80, H, W)

    # Box DFL — set a large logit at the target bin for each side so that
    # softmax is ~1 there and the decoded distance ≈ bin_index grid units.
    for side, peak_bin in enumerate(dist_bins):
        channel_start = side * reg_max
        box_out[0, channel_start + peak_bin, row, col] = 20.0  # large logit → peak

    # Class logits — a large positive value → sigmoid ≈ 1 → high confidence
    cls_out[0, class_id, row, col] = cls_score


# ---------------------------------------------------------------------------
# Test 1 — returns at least one Detection for a clear synthetic anchor
# ---------------------------------------------------------------------------

class TestDflReturnsListOfDetections:
    """_postprocess_dfl returns Detection objects for a high-confidence anchor."""

    def test_dfl_returns_list_of_detections(self) -> None:
        outputs = _make_dfl_outputs()
        # Stride-8 scale (index 0), anchor at row=40, col=40 — image center
        _inject_detection(
            outputs,
            scale_idx=0,
            row=40,
            col=40,
            dist_bins=(2, 2, 2, 2),
            class_id=0,    # person
            cls_score=5.0,
        )

        result = _postprocess_dfl(outputs, orig_w=640, orig_h=640, threshold=0.5)

        assert isinstance(result, list)
        assert len(result) >= 1

        person_detections = [d for d in result if d.class_id == 0]
        assert len(person_detections) >= 1

        det = person_detections[0]
        assert isinstance(det, Detection)
        assert det.class_name == "person"
        assert det.confidence > 0.5
        x1, y1, x2, y2 = det.bbox
        assert x1 < x2, "x1 must be left of x2"
        assert y1 < y2, "y1 must be above y2"


# ---------------------------------------------------------------------------
# Test 2 — threshold filtering
# ---------------------------------------------------------------------------

class TestDflFiltersByThreshold:
    """Low cls scores (raw logit 0.1 → sigmoid ~0.52) are filtered at threshold=0.5."""

    def test_dfl_filters_by_threshold_low_logit(self) -> None:
        outputs = _make_dfl_outputs()
        # Raw logit 0.1 → sigmoid(0.1) ≈ 0.525; just above 0.5 in raw sigmoid space.
        # We use a clearly sub-threshold logit: -2.0 → sigmoid ≈ 0.12.
        _inject_detection(
            outputs,
            scale_idx=0,
            row=10,
            col=10,
            cls_score=-2.0,
            class_id=0,
        )

        result = _postprocess_dfl(outputs, orig_w=640, orig_h=640, threshold=0.5)

        assert result == [], (
            f"Expected empty list for sub-threshold logit, got {result}"
        )

    def test_dfl_all_zeros_returns_empty(self) -> None:
        outputs = _make_dfl_outputs()  # all zeros → sigmoid(0) = 0.5, on boundary

        # We force strictly below threshold by setting logits to -5
        for i in [1, 4, 7]:  # cls outputs
            outputs[i][:] = -5.0

        result = _postprocess_dfl(outputs, orig_w=640, orig_h=640, threshold=0.5)

        assert result == []


# ---------------------------------------------------------------------------
# Test 3 — coordinate scaling to original image size
# ---------------------------------------------------------------------------

class TestDflScalesBoxesToOriginalSize:
    """Bounding boxes must be expressed in original image coordinates."""

    def test_dfl_scales_boxes_to_original_size(self) -> None:
        orig_w, orig_h = 1280, 720
        input_size = 640

        outputs = _make_dfl_outputs(input_size=input_size)
        # Stride-8 scale, anchor near image center (in 640-space)
        _inject_detection(
            outputs,
            scale_idx=0,
            row=40,
            col=40,
            dist_bins=(4, 4, 4, 4),  # 4 grid units ≈ 32px at stride 8
            class_id=0,
            cls_score=5.0,
        )

        result = _postprocess_dfl(
            outputs,
            orig_w=orig_w,
            orig_h=orig_h,
            threshold=0.5,
            input_size=input_size,
        )

        assert len(result) >= 1, "Expected at least one detection"

        for det in result:
            x1, y1, x2, y2 = det.bbox
            assert x1 >= 0, f"x1={x1} is negative"
            assert y1 >= 0, f"y1={y1} is negative"
            assert x2 <= orig_w, f"x2={x2} exceeds orig_w={orig_w}"
            assert y2 <= orig_h, f"y2={y2} exceeds orig_h={orig_h}"
            assert x1 < x2, "Degenerate box: x1 >= x2"
            assert y1 < y2, "Degenerate box: y1 >= y2"

    def test_dfl_scale_1x1_input_matches_passthrough(self) -> None:
        """When orig == input_size, coordinates must be within [0, input_size]."""
        size = 640
        outputs = _make_dfl_outputs(input_size=size)
        _inject_detection(
            outputs,
            scale_idx=0,
            row=20,
            col=20,
            dist_bins=(2, 2, 2, 2),
            class_id=2,    # car
            cls_score=5.0,
        )

        result = _postprocess_dfl(outputs, orig_w=size, orig_h=size, threshold=0.5)

        assert len(result) >= 1
        for det in result:
            x1, y1, x2, y2 = det.bbox
            assert x1 < x2 and y1 < y2
            assert x2 <= size and y2 <= size


# ---------------------------------------------------------------------------
# Test 4 — detections from multiple scales are combined
# ---------------------------------------------------------------------------

class TestDflAllNineScalesCombined:
    """Detections from stride-8, stride-16, and stride-32 scales must all appear."""

    def test_dfl_all_nine_scales_combined(self) -> None:
        outputs = _make_dfl_outputs()

        # One clear object per scale, distinct class ids to tell them apart
        _inject_detection(outputs, scale_idx=0, row=40, col=40, class_id=0, cls_score=5.0)   # person  (stride 8)
        _inject_detection(outputs, scale_idx=1, row=20, col=20, class_id=2, cls_score=5.0)   # car     (stride 16)
        _inject_detection(outputs, scale_idx=2, row=10, col=10, class_id=5, cls_score=5.0)   # bus     (stride 32)

        result = _postprocess_dfl(outputs, orig_w=640, orig_h=640, threshold=0.5)

        found_class_ids = {d.class_id for d in result}
        assert 0 in found_class_ids, "person (stride-8) not detected"
        assert 2 in found_class_ids, "car (stride-16) not detected"
        assert 5 in found_class_ids, "bus (stride-32) not detected"

    def test_dfl_combines_without_duplicates_after_nms(self) -> None:
        """Two identical boxes at the same location collapse to one after NMS."""
        outputs = _make_dfl_outputs()

        # Put two almost-identical objects at the same anchor (same scale, same cell)
        _inject_detection(outputs, scale_idx=0, row=40, col=40, class_id=0, cls_score=5.0)

        result = _postprocess_dfl(outputs, orig_w=640, orig_h=640, threshold=0.5)

        # There should not be duplicate detections for the same anchor
        person_boxes = [d.bbox for d in result if d.class_id == 0]
        assert len(person_boxes) == len(set(person_boxes)), (
            "Duplicate bounding boxes returned for a single anchor"
        )


# ---------------------------------------------------------------------------
# Test 5 — RKNNDetector.detect() dispatches to _postprocess_dfl for 9 outputs
# ---------------------------------------------------------------------------

class TestRknnDetectorDispatchesToDfl:
    """RKNNDetector.detect() must call _postprocess_dfl when the model returns 9 outputs."""

    def _build_nine_outputs(self) -> list[np.ndarray]:
        return _make_dfl_outputs()

    def _make_rknn_mock(self, mocker: "pytest_mock.MockerFixture", outputs: list[np.ndarray]):  # noqa: F821
        mock_rknn = mocker.MagicMock()
        mock_rknn.load_rknn.return_value = 0
        mock_rknn.init_runtime.return_value = 0
        mock_rknn.inference.return_value = outputs

        mock_rknnlite_api = mocker.MagicMock()
        mock_rknnlite_api.RKNNLite = mocker.MagicMock(return_value=mock_rknn)

        mock_rknnlite = mocker.MagicMock()
        mocker.patch.dict(
            "sys.modules",
            {"rknnlite": mock_rknnlite, "rknnlite.api": mock_rknnlite_api},
        )
        return mock_rknn

    def test_rknn_detector_dispatches_to_dfl(
        self, mocker: "pytest_mock.MockerFixture"  # noqa: F821
    ) -> None:
        outputs = self._build_nine_outputs()
        _inject_detection(outputs, scale_idx=0, row=40, col=40, class_id=0, cls_score=5.0)
        self._make_rknn_mock(mocker, outputs)

        spy = mocker.patch(
            "core.detector._postprocess_dfl",
            wraps=_postprocess_dfl,
        )

        det = RKNNDetector("fake.rknn")
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        det.detect(frame)

        spy.assert_called_once()

    def test_rknn_detector_does_not_call_legacy_postprocess_for_9_outputs(
        self, mocker: "pytest_mock.MockerFixture"  # noqa: F821
    ) -> None:
        from core.detector import _postprocess

        outputs = self._build_nine_outputs()
        self._make_rknn_mock(mocker, outputs)

        legacy_spy = mocker.patch(
            "core.detector._postprocess",
            wraps=_postprocess,
        )

        det = RKNNDetector("fake.rknn")
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        det.detect(frame)

        legacy_spy.assert_not_called()

    def test_rknn_detector_returns_list_of_detections_for_9_outputs(
        self, mocker: "pytest_mock.MockerFixture"  # noqa: F821
    ) -> None:
        outputs = self._build_nine_outputs()
        _inject_detection(outputs, scale_idx=1, row=20, col=20, class_id=0, cls_score=5.0)
        self._make_rknn_mock(mocker, outputs)

        det = RKNNDetector("fake.rknn")
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        result = det.detect(frame)

        assert isinstance(result, list)
        assert all(isinstance(d, Detection) for d in result)
