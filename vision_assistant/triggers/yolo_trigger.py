"""
triggers/yolo_trigger.py — Trigger de deteccion automatica con YOLOv8 ONNX.

Corre en su propio thread, analiza frames de la camara a analysis_fps,
y dispara Event(type=YOLO) con metadata enriquecida para que Gemini
genere descripciones espaciales utiles para personas con discapacidad visual.

Heuristicas base:
  1. Whitelist de clases
  2. Confidence threshold (en YoloOnnx)
  3. Clase nueva (set diff vs frame anterior)
  4. Cooldown por clase con prioridad (urgent/important/informative)
  5. Estabilidad (M de K frames)

Mejoras para asistencia visual:
  - Metadata espacial (zona + proximidad) en cada deteccion
  - Prioridades por clase con cooldowns diferenciados
  - Deteccion de cambio de escena (histograma HSV)
  - Deteccion de cambio en cantidad de objetos
  - Resumen periodico del entorno
  - Seleccion de context frames para Gemini
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
from loguru import logger

from core.events import Event, EventType
from inference.yolo_onnx import YoloOnnx
from services.camera import CameraService
from triggers.base import BaseTrigger


class YoloTrigger(BaseTrigger):
    """Trigger de deteccion automatica usando YOLOv8 ONNX.

    Args:
        event_queue: Cola compartida de eventos.
        config: Seccion "yolo_trigger" del config.yaml.
        camera: Instancia de CameraService para obtener frames.
    """

    def __init__(
        self,
        event_queue: queue.Queue,
        config: dict[str, Any],
        camera: CameraService,
    ) -> None:
        super().__init__(event_queue, config)
        self._camera = camera

        # Parametros del modelo
        self._model_path: str     = config.get("model_path", "models/yolov8n.onnx")
        self._conf_thresh: float  = config.get("confidence_threshold", 0.6)
        self._analysis_fps: int   = config.get("analysis_fps", 10)

        # Heuristicas base
        self._cooldown_secs: float  = config.get("cooldown_seconds", 5.0)
        self._stability_m: int      = config.get("stability_frames", 3)
        self._stability_k: int      = config.get("stability_window", 5)
        self._whitelist: set[str]   = set(config.get("class_whitelist", []))

        # Estado de heuristicas base
        self._cooldown_map: dict[str, float] = {}
        self._stability_window: deque[set[str]] = deque(maxlen=self._stability_k)
        self._last_classes: set[str] = set()

        # Mejora 2: Prioridades
        self._priority_cooldowns: dict[str, float] = config.get(
            "priority_cooldowns", {"urgent": 2, "important": 5, "informative": 15}
        )
        self._class_priority: dict[str, str] = config.get("class_priority", {})

        # Mejora 3: Scene change
        self._scene_change_enabled: bool = config.get("scene_change_enabled", False)
        self._scene_change_threshold: float = config.get("scene_change_threshold", 0.45)
        self._scene_change_cooldown: float = config.get("scene_change_cooldown", 10.0)
        self._last_histogram: np.ndarray | None = None
        self._last_scene_change_time: float = 0.0

        # Mejora 4: Count change
        self._count_change_enabled: bool = config.get("count_change_enabled", False)
        self._count_change_delta: int = config.get("count_change_delta", 2)
        self._count_change_cooldown: float = config.get("count_change_cooldown", 10.0)
        self._last_class_counts: dict[str, int] = {}
        self._last_count_change_time: float = 0.0

        # Mejora 5: Resumen periodico
        self._summary_enabled: bool = config.get("summary_enabled", False)
        self._summary_interval: float = config.get("summary_interval_seconds", 30.0)
        self._last_summary_time: float = 0.0

        # Mejora 6: Context frames
        self._context_frames_count: int = config.get("context_frames_count", 5)

        self._model: YoloOnnx | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Carga el modelo ONNX e inicia el thread de analisis."""
        logger.info(f"YoloTrigger: cargando modelo '{self._model_path}'...")
        self._model = YoloOnnx(
            model_path=self._model_path,
            confidence_threshold=self._conf_thresh,
        )
        self._stop_event.clear()
        self._last_summary_time = time.time()
        self._thread = threading.Thread(
            target=self._analysis_loop,
            name="YoloTrigger",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"YoloTrigger iniciado: analysis_fps={self._analysis_fps} "
            f"stability={self._stability_m}/{self._stability_k} "
            f"whitelist={sorted(self._whitelist)}"
        )

    def stop(self) -> None:
        """Detiene el thread de analisis."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("YoloTrigger detenido")

    # ------------------------------------------------------------------
    # Loop de analisis
    # ------------------------------------------------------------------

    def _analysis_loop(self) -> None:
        """Loop principal con las 6 mejoras integradas."""
        frame_interval = 1.0 / self._analysis_fps

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            frame = self._camera.get_latest_frame()
            if frame is None:
                time.sleep(frame_interval)
                continue

            try:
                detections = self._model.detect(frame)
            except Exception:
                logger.exception("[YoloTrigger] Error en inferencia")
                time.sleep(frame_interval)
                continue

            # Filtrar por whitelist
            relevant = [d for d in detections if d["class_name"] in self._whitelist]

            # Mejora 1: Enriquecer con metadata espacial
            frame_h, frame_w = frame.shape[:2]
            self._enrich_spatial(relevant, frame_h, frame_w)

            # Heuristicas base: estabilidad + clase nueva + cooldown con prioridad
            current_classes = {d["class_name"] for d in relevant}
            self._stability_window.append(current_classes)

            classes_to_fire = self._evaluate_heuristics(current_classes)

            for class_name in classes_to_fire:
                class_dets = [d for d in relevant if d["class_name"] == class_name]
                self._fire_event(frame, class_name, class_dets)
                self._cooldown_map[class_name] = time.time()

            self._last_classes = current_classes

            # Mejora 3: Scene change
            if self._scene_change_enabled:
                self._check_scene_change(frame, relevant)

            # Mejora 4: Count change
            if self._count_change_enabled:
                current_counts: dict[str, int] = {}
                for d in relevant:
                    current_counts[d["class_name"]] = current_counts.get(d["class_name"], 0) + 1
                self._check_count_change(current_counts, relevant, frame)

            # Mejora 5: Resumen periodico
            if self._summary_enabled:
                self._check_summary(frame, relevant)

            # Mantener ritmo de analysis_fps
            elapsed = time.monotonic() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.debug("YoloTrigger _analysis_loop finalizado")

    # ------------------------------------------------------------------
    # Mejora 1: Metadata espacial
    # ------------------------------------------------------------------

    @staticmethod
    def _enrich_spatial(
        detections: list[dict], frame_h: int, frame_w: int
    ) -> list[dict]:
        """Enriquece cada deteccion con zona y proximidad.

        Args:
            detections: Lista de dicts con key "bbox" [x1,y1,x2,y2].
            frame_h: Alto del frame original.
            frame_w: Ancho del frame original.

        Returns:
            La misma lista, mutada con keys "zone" y "proximity" agregadas.
        """
        frame_area = frame_h * frame_w
        third = frame_w / 3.0

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            center_x = (x1 + x2) / 2.0
            bbox_area = (x2 - x1) * (y2 - y1)
            ratio = bbox_area / frame_area if frame_area > 0 else 0

            # Zona horizontal
            if center_x < third:
                d["zone"] = "left"
            elif center_x > 2 * third:
                d["zone"] = "right"
            else:
                d["zone"] = "center"

            # Proximidad relativa por area del bbox
            if ratio > 0.15:
                d["proximity"] = "near"
            elif ratio > 0.03:
                d["proximity"] = "medium"
            else:
                d["proximity"] = "far"

        return detections

    # ------------------------------------------------------------------
    # Mejora 2: Prioridades
    # ------------------------------------------------------------------

    def _get_priority_for_class(self, class_name: str) -> str:
        """Retorna el nivel de prioridad de una clase."""
        return self._class_priority.get(class_name, "informative")

    def _get_cooldown_for_class(self, class_name: str) -> float:
        """Retorna el cooldown correspondiente a la prioridad de la clase."""
        priority = self._get_priority_for_class(class_name)
        return self._priority_cooldowns.get(priority, self._cooldown_secs)

    # ------------------------------------------------------------------
    # Heuristicas (con prioridad integrada)
    # ------------------------------------------------------------------

    def _evaluate_heuristics(self, current_classes: set[str]) -> set[str]:
        """Aplica heuristicas y retorna clases que deben disparar."""
        candidates: set[str] = set()
        now = time.time()

        for class_name in current_classes:
            # Clase nueva (no estaba en el frame anterior)
            if class_name in self._last_classes:
                continue

            # Cooldown por clase con prioridad (Mejora 2)
            last_fired = self._cooldown_map.get(class_name, 0.0)
            cooldown = self._get_cooldown_for_class(class_name)
            if (now - last_fired) < cooldown:
                logger.debug(
                    f"[YoloTrigger] '{class_name}' en cooldown "
                    f"({now - last_fired:.1f}s / {cooldown}s [{self._get_priority_for_class(class_name)}])"
                )
                continue

            # Estabilidad: aparece en >= M de los ultimos K frames
            appearances = sum(
                1 for frame_classes in self._stability_window
                if class_name in frame_classes
            )
            if appearances < self._stability_m:
                logger.debug(
                    f"[YoloTrigger] '{class_name}' inestable "
                    f"({appearances}/{self._stability_m} frames)"
                )
                continue

            candidates.add(class_name)

        return candidates

    # ------------------------------------------------------------------
    # Mejora 3: Scene change
    # ------------------------------------------------------------------

    def _check_scene_change(
        self, frame: np.ndarray, all_detections: list[dict]
    ) -> bool:
        """Detecta cambio de escena via comparacion de histogramas HSV.

        Dispara evento con trigger_class="scene_change" si la diferencia
        supera el threshold y el cooldown lo permite.

        Returns:
            True si se disparo un evento de scene_change.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        fired = False

        if self._last_histogram is not None:
            diff = cv2.compareHist(
                self._last_histogram, hist, cv2.HISTCMP_BHATTACHARYYA
            )
            now = time.time()
            if (
                diff > self._scene_change_threshold
                and (now - self._last_scene_change_time) >= self._scene_change_cooldown
            ):
                logger.info(
                    f"[YoloTrigger] Cambio de escena detectado: diff={diff:.3f} "
                    f"(threshold={self._scene_change_threshold})"
                )
                self._fire_event(frame, "scene_change", all_detections)
                self._last_scene_change_time = now
                fired = True

        self._last_histogram = hist
        return fired

    # ------------------------------------------------------------------
    # Mejora 4: Count change
    # ------------------------------------------------------------------

    def _check_count_change(
        self,
        current_counts: dict[str, int],
        all_detections: list[dict],
        frame: np.ndarray,
    ) -> bool:
        """Detecta cambios significativos en la cantidad de objetos por clase.

        Dispara evento con trigger_class="count_change" si el delta
        supera count_change_delta y el cooldown lo permite.

        Returns:
            True si se disparo al menos un evento de count_change.
        """
        now = time.time()
        fired = False

        if (now - self._last_count_change_time) < self._count_change_cooldown:
            self._last_class_counts = current_counts.copy()
            return False

        # Juntar todas las clases (actuales y anteriores) para detectar
        # tanto incrementos como decrementos
        all_classes = set(current_counts.keys()) | set(self._last_class_counts.keys())

        for class_name in all_classes:
            prev = self._last_class_counts.get(class_name, 0)
            curr = current_counts.get(class_name, 0)
            delta = abs(curr - prev)

            if delta >= self._count_change_delta:
                logger.info(
                    f"[YoloTrigger] Cambio de cantidad: '{class_name}' "
                    f"{prev} -> {curr} (delta={delta})"
                )
                self._fire_event(
                    frame,
                    "count_change",
                    all_detections,
                    extra_metadata={
                        "changed_class": class_name,
                        "prev_count": prev,
                        "new_count": curr,
                    },
                )
                self._last_count_change_time = now
                fired = True
                break  # Un evento de count_change por ciclo

        self._last_class_counts = current_counts.copy()
        return fired

    # ------------------------------------------------------------------
    # Mejora 5: Resumen periodico
    # ------------------------------------------------------------------

    def _check_summary(
        self, frame: np.ndarray, all_detections: list[dict]
    ) -> bool:
        """Emite un resumen periodico de todas las detecciones.

        Dispara cada summary_interval_seconds si hay al menos 1 deteccion.

        Returns:
            True si se disparo un evento de scene_summary.
        """
        now = time.time()
        if (now - self._last_summary_time) < self._summary_interval:
            return False
        if not all_detections:
            return False

        logger.info(
            f"[YoloTrigger] Resumen periodico: "
            f"{len(all_detections)} detecciones en escena"
        )
        self._fire_event(frame, "scene_summary", all_detections)
        self._last_summary_time = now
        return True

    # ------------------------------------------------------------------
    # Mejora 6: Context frames
    # ------------------------------------------------------------------

    def _select_context_frames(self) -> list[np.ndarray]:
        """Selecciona frames equidistantes del buffer de la camara.

        Retorna N frames espaciados uniformemente para dar contexto
        temporal a Gemini (movimiento, direccion, cambios).
        """
        buffer = self._camera.get_buffer_snapshot()
        if not buffer:
            return []
        if len(buffer) <= self._context_frames_count:
            return buffer
        indices = np.linspace(
            0, len(buffer) - 1, self._context_frames_count, dtype=int
        )
        return [buffer[i] for i in indices]

    # ------------------------------------------------------------------
    # Emision de evento
    # ------------------------------------------------------------------

    def _fire_event(
        self,
        frame: np.ndarray,
        class_name: str,
        detections: list[dict],
        extra_metadata: dict | None = None,
    ) -> None:
        """Construye y emite el Event con metadata enriquecida."""
        # Mejora 6: context frames en vez de buffer completo
        context_frames = self._select_context_frames()

        # Mejora 2: prioridad en metadata
        priority = self._get_priority_for_class(class_name)

        metadata: dict[str, Any] = {
            "trigger_class": class_name,
            "detections": detections,
            "priority": priority,
            "context_frame_count": len(context_frames),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        event = Event(
            event_type=EventType.YOLO,
            frame=frame.copy(),
            frame_buffer=context_frames if context_frames else None,
            metadata=metadata,
        )
        logger.info(
            f"[YoloTrigger] Evento: clase='{class_name}' priority={priority} "
            f"detecciones={len(detections)} ctx_frames={len(context_frames)}"
        )
        self._emit_event(event)
