"""
triggers/manual_trigger.py — Trigger de hotkey global.

Usa pynput.GlobalHotKeys para escuchar la combinación de teclas
aunque la ventana no tenga foco. Al presionar la hotkey, toma el
frame actual de la cámara y emite un Event(type=MANUAL).
"""

from __future__ import annotations

import queue
import threading
from typing import Any

from loguru import logger
from pynput import keyboard

from core.events import Event, EventType
from services.camera import CameraService
from triggers.base import BaseTrigger


class ManualTrigger(BaseTrigger):
    """Trigger activado por hotkey global (por defecto Ctrl+Shift+G).

    Args:
        event_queue: Cola compartida de eventos.
        config: Sección "manual_trigger" del config.yaml.
        camera: Instancia de CameraService para capturar el frame actual.

    Example:
        trigger = ManualTrigger(q, config["manual_trigger"], camera)
        trigger.start()
        # ... usuario presiona Ctrl+Shift+G ...
        trigger.stop()
    """

    def __init__(
        self,
        event_queue: queue.Queue,
        config: dict[str, Any],
        camera: CameraService,
    ) -> None:
        super().__init__(event_queue, config)
        self._camera = camera
        self._hotkey_str: str = config.get("hotkey", "<ctrl>+<shift>+g")
        self._listener: keyboard.GlobalHotKeys | None = None

        # DECISION: usamos threading.Event para poder hacer join limpio en stop().
        # GlobalHotKeys.stop() pide que el listener esté corriendo; el Event
        # nos permite chequear el estado sin race conditions.
        self._running = threading.Event()

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Registra la hotkey global e inicia el listener en background."""
        hotkeys = {self._hotkey_str: self._on_hotkey}

        self._listener = keyboard.GlobalHotKeys(hotkeys)
        self._listener.daemon = True
        self._listener.start()
        self._running.set()

        logger.info(
            f"ManualTrigger iniciado: hotkey='{self._hotkey_str}'"
        )

    def stop(self) -> None:
        """Detiene el listener de hotkeys."""
        if self._listener and self._running.is_set():
            self._listener.stop()
            self._running.clear()
            logger.info("ManualTrigger detenido")

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def _on_hotkey(self) -> None:
        """Callback ejecutado por pynput al detectar la hotkey.

        Captura frame + buffer y emite el evento. Corre en el thread
        del listener de pynput, por lo que debe ser rápido y no bloquear.
        """
        logger.info(f"[ManualTrigger] Hotkey '{self._hotkey_str}' detectada")

        frame = self._camera.get_latest_frame()
        buffer = self._camera.get_buffer_snapshot()

        if frame is None:
            logger.warning("[ManualTrigger] Cámara sin frames disponibles")

        event = Event(
            event_type=EventType.MANUAL,
            frame=frame,
            frame_buffer=buffer if buffer else None,
            metadata={},
        )
        self._emit_event(event)
