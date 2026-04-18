"""
core/config_loader.py — Carga y validación de config.yaml.

Expone una función load_config() que devuelve el dict de configuración.
Falla rápido con mensajes claros si faltan claves obligatorias.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# Claves obligatorias que deben existir en el YAML raíz
_REQUIRED_SECTIONS = ("camera", "manual_trigger", "yolo_trigger", "voice_trigger", "dispatcher", "logging")


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Carga config.yaml y valida que las secciones obligatorias existan.

    Args:
        path: Ruta al archivo YAML. Por defecto busca en el directorio
              de trabajo actual.

    Returns:
        Dict con la configuración completa.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan secciones obligatorias.
        yaml.YAMLError: Si el archivo tiene sintaxis inválida.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("config.yaml debe ser un mapping YAML en el nivel raíz")

    missing = [s for s in _REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ValueError(f"Secciones faltantes en config.yaml: {missing}")

    return config


def get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Acceso seguro a claves anidadas del config.

    Ejemplo:
        fps = get(config, "camera", "fps", default=30)

    Args:
        config: Dict de configuración raíz.
        *keys: Secuencia de claves para navegar el dict.
        default: Valor a retornar si alguna clave no existe.

    Returns:
        El valor encontrado o `default`.
    """
    node = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
