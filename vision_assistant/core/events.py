"""
core/events.py — Definición del evento unificado del sistema de triggers.

Todos los triggers (manual, YOLO, voz) producen instancias de Event
y las empujan a la misma queue.Queue para que el dispatcher las consuma.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


class EventType(Enum):
    MANUAL = auto()   # Hotkey global presionada por el usuario
    YOLO   = auto()   # Objeto relevante detectado por el modelo
    VOICE  = auto()   # Wake word + comando de voz reconocido


class VoiceCommand(Enum):
    SNAPSHOT          = "snapshot"           # "captura", "foto", "snapshot"
    START_RECORDING   = "start_recording"    # "graba", "grabación", "empieza"
    STOP_RECORDING    = "stop_recording"     # "para", "detén", "stop"
    DESCRIBE          = "describe"           # default: transcripción completa


@dataclass
class Event:
    """Evento unificado producido por cualquier trigger.

    Args:
        event_type: Origen del evento (MANUAL, YOLO, VOICE).
        frame: Frame capturado en el momento del evento. Puede ser None
               si la cámara no estaba disponible.
        frame_buffer: Snapshot del ring buffer de la cámara (últimos N
                      segundos). None si no aplica o cámara no disponible.
        metadata: Datos adicionales dependientes del tipo de evento:
                  - YOLO: {"detections": [{"class_name", "confidence", "bbox"}]}
                  - VOICE: {"command": VoiceCommand, "transcript": str}
                  - MANUAL: {} (vacío)
        event_id: UUID generado automáticamente. No pasar manualmente.
        timestamp: Unix timestamp en segundos. Generado automáticamente.
    """

    event_type: EventType
    frame: np.ndarray | None = None
    frame_buffer: list[np.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Generados automáticamente — no modificar desde fuera
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: __import__("time").time())

    def __repr__(self) -> str:
        has_frame = self.frame is not None
        buf_len = len(self.frame_buffer) if self.frame_buffer else 0
        return (
            f"Event(id={self.event_id[:8]}…, type={self.event_type.name}, "
            f"frame={has_frame}, buffer_frames={buf_len}, meta={list(self.metadata.keys())})"
        )
