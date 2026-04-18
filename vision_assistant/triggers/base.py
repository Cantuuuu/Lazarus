"""
triggers/base.py — Interfaz abstracta para todos los triggers.

Cada trigger concreto hereda de BaseTrigger e implementa start() y stop().
El método _emit_event() es el único punto de entrada a la cola compartida,
garantizando que todos los triggers usen el mismo mecanismo de despacho.
"""

from __future__ import annotations

import queue
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from core.events import Event


class BaseTrigger(ABC):
    """Interfaz común para ManualTrigger, YoloTrigger y VoiceTrigger.

    Args:
        event_queue: Cola compartida donde se empujan los Event producidos.
        config: Sección de configuración específica del trigger (del config.yaml).
    """

    def __init__(self, event_queue: queue.Queue, config: dict[str, Any]) -> None:
        self._queue = event_queue
        self._config = config

    @abstractmethod
    def start(self) -> None:
        """Inicia el trigger (abre listeners, threads, streams, etc.)."""

    @abstractmethod
    def stop(self) -> None:
        """Detiene el trigger de forma limpia."""

    def _emit_event(self, event: Event) -> None:
        """Empuja un evento a la cola compartida.

        Nunca bloquea: usa put_nowait() y loggea si la cola está llena.

        Args:
            event: Evento a emitir hacia el dispatcher.
        """
        try:
            self._queue.put_nowait(event)
            logger.debug(f"[{self.__class__.__name__}] Evento emitido: {event}")
        except queue.Full:
            logger.warning(
                f"[{self.__class__.__name__}] Cola llena, evento descartado: {event}"
            )
