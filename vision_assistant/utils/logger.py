"""
utils/logger.py — Setup centralizado de loguru.

Llamar setup_logging() una vez desde main.py.
El resto de módulos importan directamente: from loguru import logger
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger


def setup_logging(config: dict[str, Any]) -> None:
    """Configura loguru con los parámetros del config.yaml.

    Agrega dos sinks:
    - stderr: salida en consola con colores
    - archivo: rotación diaria, retención 7 días

    Args:
        config: Dict de configuración completo (sección "logging" se usa aquí).
    """
    log_cfg = config.get("logging", {})
    level: str = log_cfg.get("level", "INFO").upper()
    log_file: str = log_cfg.get("file", "logs/app.log")

    # Asegura que el directorio de logs exista
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Elimina el sink por defecto de loguru para reconfigurar desde cero
    logger.remove()

    # Sink consola
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
        ),
        colorize=True,
    )

    # Sink archivo con rotación diaria
    logger.add(
        log_file,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
        rotation="00:00",      # Rota a medianoche
        retention="7 days",
        encoding="utf-8",
    )

    logger.info(f"Logging configurado: level={level}, archivo={log_file}")
