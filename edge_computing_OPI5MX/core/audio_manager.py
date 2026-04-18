"""Cola de audio con prioridad para alertas de voz."""
from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _AudioItem:
    """Item de audio encolado. Menor priority = mayor urgencia."""

    priority: int
    text: str = field(compare=False)
    audio: bytes = field(compare=False)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class AudioManager:
    """Reproduce alertas de voz en orden de prioridad mediante una PriorityQueue."""

    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[_AudioItem] = asyncio.PriorityQueue()
        self._current_task: asyncio.Task | None = None
        self._drain_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    async def speak(self, text: str, audio: bytes, *, priority: int = 5) -> None:
        """Encola audio para reproducción. Número menor de priority = se reproduce antes."""
        item = _AudioItem(priority=priority, text=text, audio=audio)
        await self._queue.put(item)
        logger.debug("Encolado: %r (prioridad=%d)", text, priority)

    async def stop_current(self) -> None:
        """Cancela el audio que se está reproduciendo actualmente."""
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
            self._current_task = None
            logger.debug("Reproducción actual cancelada.")

    async def start(self) -> None:
        """Inicia el loop de drenado de la cola en background."""
        if self._drain_task is not None and not self._drain_task.done():
            return
        self._drain_task = asyncio.create_task(self._drain_loop())
        logger.debug("AudioManager iniciado.")

    async def stop(self) -> None:
        """Detiene el loop de drenado y cancela cualquier reproducción activa."""
        await self.stop_current()
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
        self._drain_task = None
        logger.debug("AudioManager detenido.")

    # ------------------------------------------------------------------
    # Loop interno
    # ------------------------------------------------------------------

    async def _drain_loop(self) -> None:
        """Loop de background: extrae y reproduce items en orden de prioridad."""
        while True:
            try:
                item = await self._queue.get()
                self._current_task = asyncio.create_task(self._play_audio(item))
                try:
                    await self._current_task
                except asyncio.CancelledError:
                    logger.debug("Reproducción cancelada: %r", item.text)
                except Exception as exc:
                    logger.error("Error reproduciendo audio %r: %s", item.text, exc)
                finally:
                    self._queue.task_done()
                    self._current_task = None
            except asyncio.CancelledError:
                break

    async def _play_audio(self, item: _AudioItem) -> None:
        """Reproduce un item de audio. Usa aplay/ffplay si hay hardware; si no, loguea."""
        if not item.audio:
            logger.info("[dev] Audio: %r", item.text)
            return

        await self._play_bytes(item.audio, item.text)

    async def _play_bytes(self, audio: bytes, label: str) -> None:
        """Escribe los bytes en un archivo temporal y los reproduce con ffplay o aplay."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio)
            tmp_path = tmp.name

        try:
            player = _find_player()
            if player is None:
                logger.warning("No se encontró reproductor de audio para: %r", label)
                return

            cmd = _build_play_command(player, tmp_path)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await proc.wait()
        except Exception as exc:
            logger.error("Error al reproducir %r: %s", label, exc)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Utilidades de reproducción
# ---------------------------------------------------------------------------

_PLAYERS = ("ffplay", "aplay", "mpg123")


def _find_player() -> str | None:
    """Retorna el primer reproductor disponible en el PATH, o None."""
    import shutil

    for player in _PLAYERS:
        if shutil.which(player):
            return player
    return None


def _build_play_command(player: str, path: str) -> list[str]:
    """Construye el comando para el reproductor dado."""
    commands: dict[str, list[str]] = {
        "ffplay": ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
        "aplay": ["aplay", path],
        "mpg123": ["mpg123", "-q", path],
    }
    return commands.get(player, [player, path])
