"""
utils/audio.py — Helpers de audio para VoiceTrigger.
"""

from __future__ import annotations

import numpy as np


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convierte audio float32 [-1, 1] a int16 [-32768, 32767].

    faster-whisper y openWakeWord esperan formatos distintos:
    - openWakeWord: float32 normalizado
    - faster-whisper: puede recibir float32 directamente

    Esta función es útil si algún componente necesita int16.
    """
    return (audio * 32767).clip(-32768, 32767).astype(np.int16)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normaliza audio int16 a float32 en [-1, 1]."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def parse_voice_command(transcript: str) -> str:
    """Determina el comando a partir de la transcripción.

    Args:
        transcript: Texto transcrito por Whisper (en español).

    Returns:
        Uno de: "snapshot", "start_recording", "stop_recording", "describe"
    """
    text = transcript.lower().strip()

    snapshot_keywords  = ["captura", "foto", "fotografía", "snapshot", "imagen", "toma"]
    start_keywords     = ["graba", "grabación", "grabar", "empieza", "inicia", "comienza", "video"]
    stop_keywords      = ["para", "detén", "detener", "stop", "termina", "finaliza", "alto"]

    # DECISION: stop se evalúa antes que start porque "grabación" contiene
    # "graba", lo que causaría falsos positivos en start si se evalúa primero.
    for kw in stop_keywords:
        if kw in text:
            return "stop_recording"
    for kw in snapshot_keywords:
        if kw in text:
            return "snapshot"
    for kw in start_keywords:
        if kw in text:
            return "start_recording"

    return "describe"
