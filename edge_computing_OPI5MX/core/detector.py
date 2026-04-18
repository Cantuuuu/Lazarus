"""Wrapper de detección de objetos con YOLO.

Backends:
- MockDetector   — dev local, sin modelo real
- ONNXDetector   — CPU con OpenCV DNN (desarrollo/fallback)
- RKNNDetector   — NPU Orange Pi RK3588 (producción)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Clases COCO relevantes para Lazarillo
PRIORITY_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic_light",
    11: "stop_sign",
    15: "cat",
    16: "dog",
}

COCO_NAMES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic_light", "fire_hydrant", "stop_sign",
    "parking_meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", "kite",
    "baseball_bat", "baseball_glove", "skateboard", "surfboard", "tennis_racket",
    "bottle", "wine_glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot_dog", "pizza",
    "donut", "cake", "chair", "couch", "potted_plant", "bed", "dining_table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell_phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy_bear", "hair_drier", "toothbrush",
]


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2

    @property
    def is_priority(self) -> bool:
        return self.class_id in PRIORITY_CLASSES

    @property
    def center_x(self) -> int:
        return (self.bbox[0] + self.bbox[2]) // 2

    @property
    def direction(self) -> str:
        """Dirección relativa en el frame (640px ancho)."""
        cx = self.center_x
        if cx < 213:
            return "izquierda"
        if cx < 427:
            return "frente"
        return "derecha"


class DetectorBackend(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class MockDetector:
    """Detector falso para pruebas — retorna detecciones predecibles."""

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._detections = detections or [
            Detection(0, "person", 0.92, (200, 100, 280, 300)),
            Detection(2, "car", 0.87, (400, 300, 520, 420)),
        ]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return self._detections


class ONNXDetector:
    """Detector YOLO con onnxruntime — corre en CPU, sin dependencias de NPU."""

    INPUT_SIZE = 640

    def __init__(self, model_path: str, confidence_threshold: float = 0.4) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime no disponible — instala con: pip install onnxruntime"
            ) from e

        self._threshold = confidence_threshold
        # RK3588 tiene 4 big cores (A76) + 4 efficiency cores (A55).
        # intra_op_num_threads=4 usa los big cores para operaciones internas
        # (matmul, conv); inter_op_num_threads=2 paraleliza nodos independientes.
        # ORT_PARALLEL permite que ambos niveles de paralelismo coexistan.
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4  # 4 big cores A76
        sess_options.inter_op_num_threads = 2
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        self._session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (self.INPUT_SIZE, self.INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
        outputs = self._session.run(None, {self._input_name: blob})
        if len(outputs) == 9:
            return _postprocess_dfl(outputs, w, h, self._threshold)
        return _postprocess(outputs[0][0], w, h, self._threshold)


class RKNNDetector:
    """Detector YOLO en NPU RK3588 — solo disponible en Orange Pi."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.4) -> None:
        try:
            from rknnlite.api import RKNNLite  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "rknnlite no disponible — usa ONNXDetector o MockDetector en dev"
            ) from e

        self._threshold = confidence_threshold
        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Error cargando modelo RKNN: {ret}")
        ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            raise RuntimeError(f"Error inicializando runtime RKNN: {ret}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
        outputs = self._rknn.inference(inputs=[blob])
        if outputs is None or len(outputs) == 0:
            return []
        if len(outputs) == 9:
            return _postprocess_dfl(outputs, w, h, self._threshold)
        return _postprocess(outputs[0][0], w, h, self._threshold)


def build_detector(
    *,
    dev_mode: bool = True,
    model_path: str = "",
    confidence_threshold: float = 0.4,
) -> DetectorBackend:
    """Factory — elige el backend correcto según el entorno."""
    if dev_mode:
        return MockDetector()
    if model_path.endswith(".rknn"):
        try:
            return RKNNDetector(model_path, confidence_threshold)
        except RuntimeError as e:
            import os
            onnx_path = model_path.replace(".rknn", ".onnx")
            if os.path.exists(onnx_path):
                logger.warning("RKNNDetector falló (%s), usando ONNXDetector: %s", e, onnx_path)
                return ONNXDetector(onnx_path, confidence_threshold)
            logger.warning("RKNNDetector falló y no hay ONNX fallback, usando MockDetector")
            return MockDetector()
    if model_path.endswith(".onnx"):
        return ONNXDetector(model_path, confidence_threshold)
    return MockDetector()


# ---------------------------------------------------------------------------
# Post-procesado YOLO (formato YOLOv8 — [batch, 84, 8400])
# ---------------------------------------------------------------------------

def _postprocess(
    output: np.ndarray,
    orig_w: int,
    orig_h: int,
    threshold: float,
) -> list[Detection]:
    """Convierte salida raw YOLO a lista de Detection con NMS."""
    # output shape: (84, 8400) — 4 coords + 80 clases
    if output.ndim == 3:
        output = output[0]

    rows = output.T  # (8400, 84)
    boxes, scores, class_ids = [], [], []

    x_scale = orig_w / 640
    y_scale = orig_h / 640

    for row in rows:
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        if confidence < threshold:
            continue

        cx, cy, bw, bh = row[:4]
        if np.isnan(cx) or np.isnan(cy) or np.isnan(bw) or np.isnan(bh):
            continue
        x1 = int((cx - bw / 2) * x_scale)
        y1 = int((cy - bh / 2) * y_scale)
        x2 = int((cx + bw / 2) * x_scale)
        y2 = int((cy + bh / 2) * y_scale)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(confidence)
        class_ids.append(class_id)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, 0.45)
    detections = []
    for i in indices:
        idx = int(i)
        x, y, w, h = boxes[idx]
        cid = class_ids[idx]
        name = COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid)
        detections.append(Detection(
            class_id=cid,
            class_name=name,
            confidence=scores[idx],
            bbox=(x, y, x + w, y + h),
        ))

    return detections


def _postprocess_dfl(
    outputs: list[np.ndarray],
    orig_w: int,
    orig_h: int,
    threshold: float,
    input_size: int = 640,
    reg_max: int = 16,
) -> list[Detection]:
    """Postprocessing para 9-outputs DFL YOLOv8 RKNN.

    Estructura por escala (3 escalas × 3 tensores):
      outputs[i*3+0]: (1, 64, H, W)  — box DFL (4 * reg_max distancias)
      outputs[i*3+1]: (1, 80, H, W)  — class logits
      outputs[i*3+2]: (1,  1, H, W)  — ignorado
    Escalas: stride 8 (80×80), stride 16 (40×40), stride 32 (20×20).
    """
    strides = [8, 16, 32]
    x_scale = orig_w / input_size
    y_scale = orig_h / input_size
    bins = np.arange(reg_max, dtype=np.float32)

    all_boxes: list[list[int]] = []
    all_scores: list[float] = []
    all_class_ids: list[int] = []

    for i, stride in enumerate(strides):
        box_dfl = outputs[i * 3][0]      # (64, H, W)
        cls_raw = outputs[i * 3 + 1][0]  # (80, H, W)
        _, H, W = cls_raw.shape

        # Anchor grid — center points in input_size coords
        gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        cx = (gx.ravel() + 0.5) * stride  # (H*W,)
        cy = (gy.ravel() + 0.5) * stride

        # DFL decode: (64, H, W) → (4, reg_max, H*W) → softmax → weighted sum → distances
        flat = box_dfl.reshape(4, reg_max, H * W)
        shifted = flat - flat.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        soft = exp / exp.sum(axis=1, keepdims=True)
        dist = (soft * bins[np.newaxis, :, np.newaxis]).sum(axis=1) * stride  # (4, H*W)

        x1 = (cx - dist[0]) * x_scale
        y1 = (cy - dist[1]) * y_scale
        x2 = (cx + dist[2]) * x_scale
        y2 = (cy + dist[3]) * y_scale

        # Class scores are already sigmoid-activated by the RKNN toolkit (values in [0,1]).
        # Do NOT apply sigmoid again — use directly as probabilities.
        cls_flat = cls_raw.reshape(80, H * W)
        max_cls = cls_flat.max(axis=0)

        pre_mask = max_cls >= threshold
        if not pre_mask.any():
            continue

        idx_keep = np.where(pre_mask)[0]
        cls_sub = cls_flat[:, idx_keep]
        class_ids_arr = np.argmax(cls_sub, axis=0)
        max_scores = cls_sub[class_ids_arr, np.arange(len(idx_keep))]

        mask = max_scores >= threshold
        if not mask.any():
            continue

        for local_j in np.where(mask)[0]:
            global_j = int(idx_keep[local_j])
            bx1, by1, bx2, by2 = int(x1[global_j]), int(y1[global_j]), int(x2[global_j]), int(y2[global_j])
            all_boxes.append([bx1, by1, bx2 - bx1, by2 - by1])
            all_scores.append(float(max_scores[local_j]))
            all_class_ids.append(int(class_ids_arr[local_j]))

    if not all_boxes:
        return []

    indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, threshold, 0.45)
    result = []
    for idx in indices:
        idx = int(idx)
        x, y, w, h = all_boxes[idx]
        cid = all_class_ids[idx]
        name = COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid)
        result.append(Detection(
            class_id=cid,
            class_name=name,
            confidence=all_scores[idx],
            bbox=(x, y, x + w, y + h),
        ))
    return result
