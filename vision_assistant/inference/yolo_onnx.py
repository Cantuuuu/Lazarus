"""
inference/yolo_onnx.py — Wrapper de inferencia ONNX para YOLOv8.

Abstrae completamente el backend: en desarrollo usa CUDAExecutionProvider,
en Orange Pi 5 usará CPUExecutionProvider (o RKNN tras conversión).
El código que llama a este módulo no necesita cambiar al portar.

Pipeline:
    imagen BGR (numpy) -> preprocess -> onnxruntime -> postprocess (NMS) -> detecciones

Formato de salida del modelo YOLOv8n exportado:
    (1, 84, 8400)
    84 = 4 coords (cx, cy, w, h) + 80 scores de clase COCO
    8400 = anchors (80x80 + 40x40 + 20x20 = 8400 para imgsz=640)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from loguru import logger

# Nombres de las 80 clases COCO en el orden del modelo
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


# Tipo de retorno de detect()
Detection = dict[str, Any]
# Ejemplo: {"class_name": "person", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}


class YoloOnnx:
    """Wrapper de inferencia YOLOv8 sobre onnxruntime.

    Args:
        model_path: Ruta al archivo .onnx exportado por export_yolo.py.
        confidence_threshold: Score mínimo para aceptar una detección.
        iou_threshold: Umbral IoU para NMS (suprimir cajas superpuestas).

    Example:
        model = YoloOnnx("models/yolov8n.onnx", confidence_threshold=0.6)
        detections = model.detect(frame_bgr)
        # [{"class_name": "person", "confidence": 0.91, "bbox": [x1,y1,x2,y2]}, ...]
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ) -> None:
        self._model_path = Path(model_path)
        self._conf_thresh = confidence_threshold
        self._iou_thresh = iou_threshold
        self._imgsz = 640  # fijo en la exportación; cambiar si se re-exporta

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Modelo ONNX no encontrado: {self._model_path.resolve()}\n"
                f"Ejecuta primero: python scripts/export_yolo.py"
            )

        self._session = self._load_session()
        self._input_name: str = self._session.get_inputs()[0].name
        logger.info(
            f"YoloOnnx cargado: {self._model_path.name} "
            f"conf={self._conf_thresh} iou={self._iou_thresh} "
            f"provider={self._session.get_providers()[0]}"
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Ejecuta detección sobre un frame BGR de OpenCV.

        Args:
            frame_bgr: Imagen en formato BGR (H, W, 3), uint8.

        Returns:
            Lista de detecciones ordenadas por confidence descendente.
            Cada detección es un dict con:
                - class_name (str): nombre de la clase COCO
                - confidence (float): score [0, 1]
                - bbox (list[int]): [x1, y1, x2, y2] en coords del frame original
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        original_h, original_w = frame_bgr.shape[:2]

        # 1. Preprocess
        input_tensor, scale, pad_x, pad_y = self._preprocess(frame_bgr)

        # 2. Inferencia
        outputs = self._session.run(None, {self._input_name: input_tensor})
        raw_output = outputs[0]  # shape: (1, 84, 8400)

        # 3. Postprocess
        detections = self._postprocess(
            raw_output, scale, pad_x, pad_y, original_w, original_h
        )

        return detections

    # ------------------------------------------------------------------
    # Carga de sesión
    # ------------------------------------------------------------------

    def _load_session(self) -> ort.InferenceSession:
        """Carga la sesión onnxruntime con el mejor provider disponible.

        DECISION: intentamos CUDA primero, fallback a CPU.
        En Orange Pi 5 el provider será CPU (o un custom RKNN provider).
        El código cliente no necesita saber cuál se usa.
        """
        available = ort.get_available_providers()
        providers: list[str] = []

        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
            logger.info("YoloOnnx: usando CUDAExecutionProvider")
        else:
            logger.info("YoloOnnx: CUDA no disponible, usando CPUExecutionProvider")

        providers.append("CPUExecutionProvider")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # DECISION: inter_op = 1 hilo para no interferir con los otros threads
        # del sistema (CameraService, dispatcher, etc.). En el Pi también es
        # preferible dado que la NPU maneja su propio paralelismo.
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2

        return ort.InferenceSession(
            str(self._model_path),
            sess_options=opts,
            providers=providers,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(
        self, frame_bgr: np.ndarray
    ) -> tuple[np.ndarray, float, int, int]:
        """Convierte un frame BGR a tensor de entrada ONNX.

        Aplica letterbox: escala con aspect ratio preservado y rellena
        con gris (114) hasta llegar a imgsz×imgsz.

        Returns:
            input_tensor: float32 (1, 3, imgsz, imgsz) normalizado [0,1]
            scale: factor de escala aplicado (para revertir en postprocess)
            pad_x: píxeles de padding horizontal (izquierda)
            pad_y: píxeles de padding vertical (arriba)
        """
        h, w = frame_bgr.shape[:2]
        scale = min(self._imgsz / w, self._imgsz / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        import cv2
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Canvas gris y centrado
        canvas = np.full((self._imgsz, self._imgsz, 3), 114, dtype=np.uint8)
        pad_x = (self._imgsz - new_w) // 2
        pad_y = (self._imgsz - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        # BGR -> RGB, HWC -> CHW, uint8 -> float32 [0,1]
        rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1))[np.newaxis]  # (1,3,H,W)

        return tensor, scale, pad_x, pad_y

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        raw: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        orig_w: int,
        orig_h: int,
    ) -> list[Detection]:
        """Decodifica la salida cruda del modelo y aplica NMS.

        Args:
            raw: Array (1, 84, 8400) directo del modelo.
            scale, pad_x, pad_y: Parámetros del letterbox para revertir coords.
            orig_w, orig_h: Dimensiones del frame original.

        Returns:
            Lista de Detection dicts tras NMS, ordenados por confidence desc.
        """
        # (1, 84, 8400) -> (8400, 84)
        preds = raw[0].T  # (8400, 84)

        # Separar coords y scores
        boxes_cxcywh = preds[:, :4]       # (8400, 4) — cx, cy, w, h en imgsz coords
        class_scores = preds[:, 4:]        # (8400, 80)

        # Confianza = score máximo de clase (YOLOv8 no tiene objectness separado)
        confidences = class_scores.max(axis=1)   # (8400,)
        class_ids   = class_scores.argmax(axis=1) # (8400,)

        # Filtro por threshold antes de NMS para reducir trabajo
        mask = confidences >= self._conf_thresh
        if not mask.any():
            return []

        boxes_f   = boxes_cxcywh[mask]
        confs_f   = confidences[mask]
        classes_f = class_ids[mask]

        # cx,cy,w,h (imgsz) -> x1,y1,x2,y2 (imgsz) -> coords originales
        x1y1x2y2 = self._cxcywh_to_xyxy(boxes_f)
        x1y1x2y2 = self._letterbox_to_original(x1y1x2y2, scale, pad_x, pad_y, orig_w, orig_h)

        # NMS por clase
        keep = self._nms(x1y1x2y2, confs_f, self._iou_thresh)

        detections: list[Detection] = []
        for idx in keep:
            x1, y1, x2, y2 = x1y1x2y2[idx].tolist()
            detections.append({
                "class_name":  COCO_CLASSES[int(classes_f[idx])],
                "confidence":  float(confs_f[idx]),
                "bbox":        [int(x1), int(y1), int(x2), int(y2)],
            })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    # ------------------------------------------------------------------
    # Helpers geométricos
    # ------------------------------------------------------------------

    @staticmethod
    def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        """Convierte (cx, cy, w, h) -> (x1, y1, x2, y2)."""
        out = np.empty_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return out

    @staticmethod
    def _letterbox_to_original(
        boxes: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        orig_w: int,
        orig_h: int,
    ) -> np.ndarray:
        """Revierte el letterbox: quita padding y deshace el scale."""
        out = boxes.copy()
        out[:, 0] = (out[:, 0] - pad_x) / scale  # x1
        out[:, 1] = (out[:, 1] - pad_y) / scale  # y1
        out[:, 2] = (out[:, 2] - pad_x) / scale  # x2
        out[:, 3] = (out[:, 3] - pad_y) / scale  # y2

        # Clip a los límites del frame original
        out[:, 0] = out[:, 0].clip(0, orig_w)
        out[:, 1] = out[:, 1].clip(0, orig_h)
        out[:, 2] = out[:, 2].clip(0, orig_w)
        out[:, 3] = out[:, 3].clip(0, orig_h)
        return out

    @staticmethod
    def _nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> list[int]:
        """Non-Maximum Suppression en numpy puro (portable, sin CUDA requerido).

        Args:
            boxes:  (N, 4) xyxy float.
            scores: (N,) float.
            iou_threshold: cajas con IoU > umbral se suprimen.

        Returns:
            Índices de las cajas a conservar.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)

        order = scores.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break

            rest = order[1:]
            ix1 = np.maximum(x1[i], x1[rest])
            iy1 = np.maximum(y1[i], y1[rest])
            ix2 = np.minimum(x2[i], x2[rest])
            iy2 = np.minimum(y2[i], y2[rest])

            inter = (ix2 - ix1).clip(0) * (iy2 - iy1).clip(0)
            union = areas[i] + areas[rest] - inter
            iou   = np.where(union > 0, inter / union, 0.0)

            order = rest[iou <= iou_threshold]

        return keep
