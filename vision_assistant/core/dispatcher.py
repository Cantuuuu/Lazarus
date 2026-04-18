"""
core/dispatcher.py — Consumidor central de la cola de eventos.

EventDispatcher corre en su propio thread daemon. Consume Event objetos
de una queue.Queue compartida, aplica anti-spam por ventana temporal
y los pasa al handler stub (guarda frame + metadata a disco).

TODO: reemplazar _handle_event() por llamadas a endpoint Gemini + ElevenLabs
      cuando esté lista la integración de la siguiente fase.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from core.events import Event, EventType


class EventDispatcher:
    """Consumidor de cola de eventos con anti-spam y handler stub.

    Args:
        event_queue: Cola compartida donde los triggers empujan eventos.
        config: Sección "dispatcher" del config.yaml.

    Example:
        q = queue.Queue()
        dispatcher = EventDispatcher(q, config["dispatcher"])
        dispatcher.start()
        # ... triggers empujan a q ...
        dispatcher.stop()
    """

    def __init__(self, event_queue: queue.Queue, config: dict[str, Any]) -> None:
        self._queue = event_queue
        self._dedupe_window: float = config.get("dedupe_window_seconds", 1.0)
        self._output_dir = Path(config.get("output_dir", "output"))
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Anti-spam: guarda el último timestamp por EventType
        self._last_seen: dict[EventType, float] = {}
        self._last_seen_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Inicia el thread consumidor."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._consume_loop,
            name="EventDispatcher",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"EventDispatcher iniciado: dedupe={self._dedupe_window}s "
            f"output='{self._output_dir}'"
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Señala al thread que pare, espera a que drene la cola y termine.

        Args:
            timeout: Segundos máximos a esperar para el join.
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("EventDispatcher detenido")

    # ------------------------------------------------------------------
    # Loop interno
    # ------------------------------------------------------------------

    def _consume_loop(self) -> None:
        """Loop principal: extrae eventos de la cola y los procesa."""
        while not self._stop_event.is_set():
            try:
                # Timeout corto para poder chequear stop_event regularmente
                event: Event = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if self._is_spam(event):
                    logger.debug(
                        f"[DISPATCHER] Evento descartado (anti-spam): {event}"
                    )
                else:
                    self._handle_event(event)
            except Exception:
                logger.exception(f"Error procesando evento {event.event_id}")
            finally:
                self._queue.task_done()

        # Drena los eventos restantes al hacer stop
        self._drain()
        logger.debug("_consume_loop finalizado")

    def _drain(self) -> None:
        """Procesa los eventos que quedaron en la cola al hacer stop."""
        drained = 0
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                self._handle_event(event)
                self._queue.task_done()
                drained += 1
            except queue.Empty:
                break
        if drained:
            logger.info(f"[DISPATCHER] Drenados {drained} eventos pendientes al cerrar")

    # ------------------------------------------------------------------
    # Anti-spam
    # ------------------------------------------------------------------

    def _is_spam(self, event: Event) -> bool:
        """Retorna True si el evento debe descartarse por anti-spam.

        Ignora eventos del mismo EventType que lleguen dentro de la
        ventana dedupe_window_seconds.
        """
        now = time.time()
        with self._last_seen_lock:
            last = self._last_seen.get(event.event_type, 0.0)
            if (now - last) < self._dedupe_window:
                return True
            self._last_seen[event.event_type] = now
            return False

    # ------------------------------------------------------------------
    # Handler stub
    # ------------------------------------------------------------------

    def _handle_event(self, event: Event) -> None:
        """Guarda frame y metadata a disco. Stub de la integración futura.

        Nombre de archivos: output/{timestamp}_{type}.jpg / .json

        TODO: aquí integrar con endpoint Gemini + ElevenLabs.
              Reemplazar este método (o llamarlo antes) con:
                response = gemini_client.describe(frame, metadata)
                elevenlabs_client.speak(response)
        """
        ts = f"{event.timestamp:.3f}"
        stem = f"{ts}_{event.event_type.name}"

        logger.info(
            f"[DISPATCHER] Evento recibido: type={event.event_type.name} "
            f"id={event.event_id[:8]} ts={ts}"
        )

        # Guardar frame como JPG si existe
        if event.frame is not None:
            jpg_path = self._output_dir / f"{stem}.jpg"
            ok = cv2.imwrite(str(jpg_path), event.frame)
            if ok:
                logger.info(f"[DISPATCHER] Frame guardado: {jpg_path}")
            else:
                logger.warning(f"[DISPATCHER] No se pudo guardar frame en {jpg_path}")
        else:
            logger.debug("[DISPATCHER] Evento sin frame, omitiendo guardado de JPG")

        # Guardar context frames (Mejora 6: multi-frame para Gemini)
        ctx_saved = 0
        if event.frame_buffer:
            for idx, ctx_frame in enumerate(event.frame_buffer):
                ctx_path = self._output_dir / f"{stem}_ctx_{idx}.jpg"
                cv2.imwrite(str(ctx_path), ctx_frame)
                ctx_saved += 1
            if ctx_saved:
                logger.debug(
                    f"[DISPATCHER] {ctx_saved} context frames guardados"
                )

        # Guardar metadata como JSON siempre
        json_path = self._output_dir / f"{stem}.json"
        payload = {
            "event_id":   event.event_id,
            "event_type": event.event_type.name,
            "timestamp":  event.timestamp,
            "metadata":   _serialize_metadata(event.metadata),
            "has_frame":  event.frame is not None,
            "buffer_frames": len(event.frame_buffer) if event.frame_buffer else 0,
            "context_frames_saved": ctx_saved,
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.debug(f"[DISPATCHER] Metadata guardada: {json_path}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _serialize_metadata(meta: dict) -> dict:
    """Convierte valores no-JSON-serializables del metadata a strings.

    Cubre: numpy arrays, enums, y otros tipos custom.
    """
    result = {}
    for k, v in meta.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif hasattr(v, "value"):       # Enum con .value
            result[k] = v.value
        elif hasattr(v, "name"):        # Enum sin .value pero con .name
            result[k] = v.name
        else:
            result[k] = v
    return result
