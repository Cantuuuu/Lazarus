"""
main.py — Punto de entrada del sistema de triggers.

Arranca en orden:
  1. Logging
  2. CameraService (ring buffer)
  3. EventDispatcher (consumidor de cola)
  4. ManualTrigger  (hotkey global)
  5. [futuro] YoloTrigger
  6. [futuro] VoiceTrigger

Shutdown limpio con Ctrl+C: detiene todos los componentes en orden inverso
y drena la cola antes de salir.
"""

from __future__ import annotations

import queue
import sys
import time
from pathlib import Path

from loguru import logger

# Asegura que el directorio raíz del proyecto esté en sys.path
# cuando se ejecuta con: python main.py  (desde vision_assistant/)
sys.path.insert(0, str(Path(__file__).parent))

from core.config_loader import load_config
from core.dispatcher import EventDispatcher
from services.camera import CameraService
from triggers.manual_trigger import ManualTrigger
from triggers.voice_trigger import VoiceTrigger
from triggers.yolo_trigger import YoloTrigger
from utils.logger import setup_logging


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Config
    # ------------------------------------------------------------------
    config = load_config("config.yaml")

    # ------------------------------------------------------------------
    # 2. Logging
    # ------------------------------------------------------------------
    setup_logging(config)
    logger.info("=== Vision Assistant — Sistema de Triggers ===")
    logger.info(f"Hotkey manual: {config['manual_trigger']['hotkey']}")

    # ------------------------------------------------------------------
    # 3. Cola compartida
    # ------------------------------------------------------------------
    event_queue: queue.Queue = queue.Queue(maxsize=100)

    # ------------------------------------------------------------------
    # 4. CameraService
    # ------------------------------------------------------------------
    camera = CameraService(config["camera"])
    camera.start()

    # Pequeña espera para que el buffer tenga al menos un frame
    # antes de que el usuario pueda presionar la hotkey
    time.sleep(0.5)

    # ------------------------------------------------------------------
    # 5. EventDispatcher
    # ------------------------------------------------------------------
    dispatcher = EventDispatcher(event_queue, config["dispatcher"])
    dispatcher.start()

    # ------------------------------------------------------------------
    # 6. ManualTrigger
    # ------------------------------------------------------------------
    manual = ManualTrigger(event_queue, config["manual_trigger"], camera)
    manual.start()

    # ------------------------------------------------------------------
    # 7. YoloTrigger
    # ------------------------------------------------------------------
    yolo: YoloTrigger | None = None
    if config["yolo_trigger"].get("enabled", True):
        yolo = YoloTrigger(event_queue, config["yolo_trigger"], camera)
        yolo.start()
    else:
        logger.info("YoloTrigger deshabilitado en config")

    # ------------------------------------------------------------------
    # 8. VoiceTrigger
    # ------------------------------------------------------------------
    voice: VoiceTrigger | None = None
    if config["voice_trigger"].get("enabled", True):
        voice = VoiceTrigger(event_queue, config["voice_trigger"], camera)
        voice.start()
    else:
        logger.info("VoiceTrigger deshabilitado en config")

    logger.info("Sistema listo. Presiona Ctrl+C para salir.")
    logger.info(f"Hotkey activa: {config['manual_trigger']['hotkey']}")

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------
    try:
        while True:
            time.sleep(0.5)

            # Log periódico de estado (cada 30 s aprox)
            # DECISION: usamos un contador simple en vez de un timer
            # separado para no añadir complejidad innecesaria aquí.
            if not camera.is_running:
                logger.error("CameraService se detuvo inesperadamente, saliendo...")
                break

    except KeyboardInterrupt:
        logger.info("Ctrl+C recibido — iniciando shutdown limpio...")

    finally:
        _shutdown(manual, yolo, voice, dispatcher, camera)


def _shutdown(
    manual: ManualTrigger,
    yolo: YoloTrigger | None,
    voice: VoiceTrigger | None,
    dispatcher: EventDispatcher,
    camera: CameraService,
) -> None:
    """Detiene todos los componentes en orden inverso al inicio."""
    logger.info("Deteniendo ManualTrigger...")
    manual.stop()

    if yolo is not None:
        logger.info("Deteniendo YoloTrigger...")
        yolo.stop()

    if voice is not None:
        logger.info("Deteniendo VoiceTrigger...")
        voice.stop()

    logger.info("Deteniendo EventDispatcher (drenando cola)...")
    dispatcher.stop(timeout=5.0)

    logger.info("Deteniendo CameraService...")
    camera.stop()

    logger.info("Shutdown completo. Hasta luego.")


if __name__ == "__main__":
    main()
