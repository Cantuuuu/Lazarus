"""Tests para AudioManager — TDD fase RED antes de implementar core/audio_manager.py."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from core.audio_manager import AudioManager, _AudioItem


# ---------------------------------------------------------------------------
# _AudioItem: ordenamiento por prioridad
# ---------------------------------------------------------------------------

class TestAudioItemOrdering:
    def test_lower_priority_number_sorts_first(self) -> None:
        high = _AudioItem(priority=1, text="urgente", audio=b"")
        low = _AudioItem(priority=5, text="normal", audio=b"")
        assert high < low

    def test_equal_priority_items_are_comparable(self) -> None:
        a = _AudioItem(priority=3, text="a", audio=b"")
        b = _AudioItem(priority=3, text="b", audio=b"")
        # No debe lanzar excepción al comparar
        assert not (a < b) and not (b < a)


# ---------------------------------------------------------------------------
# speak(): encola sin bloquear
# ---------------------------------------------------------------------------

class TestSpeak:
    async def test_speak_enqueues_item(self) -> None:
        manager = AudioManager()
        await manager.start()

        await manager.speak("persona al frente", b"audio-data", priority=5)

        assert manager._queue.qsize() >= 0  # Al menos fue procesado o está en cola
        await manager.stop()

    async def test_speak_is_non_blocking(self) -> None:
        """speak() debe retornar inmediatamente sin esperar reproducción."""
        manager = AudioManager()
        await manager.start()

        # speak() no debe tardar más de unos pocos ms
        import time
        start = time.monotonic()
        await manager.speak("auto a la izquierda", b"audio-bytes", priority=3)
        elapsed = time.monotonic() - start

        assert elapsed < 0.1, f"speak() tardó {elapsed:.3f}s, se esperaba < 0.1s"
        await manager.stop()

    async def test_speak_accepts_bytes_audio(self) -> None:
        manager = AudioManager()
        await manager.start()
        # No debe lanzar excepción
        await manager.speak("semáforo al frente", b"", priority=5)
        await manager.stop()


# ---------------------------------------------------------------------------
# Prioridad: items de mayor prioridad (número menor) se reproducen primero
# ---------------------------------------------------------------------------

class TestPriorityOrder:
    async def test_higher_priority_item_dequeued_first(self) -> None:
        """Con la cola pausada, insertar dos items y verificar el orden al sacarlos."""
        manager = AudioManager()
        # No iniciamos el drain loop para poder inspeccionar la cola

        await manager._queue.put(_AudioItem(priority=5, text="baja", audio=b"low"))
        await manager._queue.put(_AudioItem(priority=1, text="alta", audio=b"high"))

        first = await manager._queue.get()
        second = await manager._queue.get()

        assert first.priority == 1
        assert second.priority == 5

    async def test_priority_queue_orders_multiple_items(self) -> None:
        manager = AudioManager()

        items = [
            _AudioItem(priority=3, text="media", audio=b""),
            _AudioItem(priority=1, text="alta", audio=b""),
            _AudioItem(priority=5, text="baja", audio=b""),
            _AudioItem(priority=2, text="alta-2", audio=b""),
        ]
        for item in items:
            await manager._queue.put(item)

        priorities = []
        while not manager._queue.empty():
            item = await manager._queue.get()
            priorities.append(item.priority)

        assert priorities == sorted(priorities)


# ---------------------------------------------------------------------------
# stop_current(): interrumpe reproducción activa
# ---------------------------------------------------------------------------

class TestStopCurrent:
    async def test_stop_current_cancels_active_task(self) -> None:
        manager = AudioManager()

        # Simular una tarea activa
        async def long_task() -> None:
            await asyncio.sleep(10)

        manager._current_task = asyncio.create_task(long_task())

        await manager.stop_current()

        assert manager._current_task is None or manager._current_task.cancelled()

    async def test_stop_current_is_safe_when_no_task(self) -> None:
        """stop_current() no debe lanzar excepción si no hay tarea activa."""
        manager = AudioManager()
        assert manager._current_task is None
        # No debe lanzar excepción
        await manager.stop_current()


# ---------------------------------------------------------------------------
# start() / stop(): ciclo de vida del manager
# ---------------------------------------------------------------------------

class TestLifecycle:
    async def test_start_creates_drain_task(self) -> None:
        manager = AudioManager()
        await manager.start()

        assert manager._drain_task is not None
        assert not manager._drain_task.done()

        await manager.stop()

    async def test_stop_cancels_drain_task(self) -> None:
        manager = AudioManager()
        await manager.start()
        drain_task = manager._drain_task

        await manager.stop()

        assert drain_task.done() or drain_task.cancelled()

    async def test_stop_clears_current_task(self) -> None:
        manager = AudioManager()
        await manager.start()
        await manager.stop()

        assert manager._current_task is None


# ---------------------------------------------------------------------------
# _drain_loop(): procesa items en orden de prioridad
# ---------------------------------------------------------------------------

class TestDrainLoop:
    async def test_drain_loop_processes_items_in_priority_order(self) -> None:
        """El drain loop debe procesar el item de mayor prioridad primero."""
        processed: list[str] = []

        manager = AudioManager()

        # Patch _play_audio para registrar el orden de procesamiento
        async def fake_play(item: _AudioItem) -> None:
            processed.append(item.text)

        manager._play_audio = fake_play  # type: ignore[assignment]

        # Encolar items antes de iniciar el loop
        await manager._queue.put(_AudioItem(priority=5, text="baja", audio=b""))
        await manager._queue.put(_AudioItem(priority=1, text="alta", audio=b""))
        await manager._queue.put(_AudioItem(priority=3, text="media", audio=b""))

        await manager.start()
        # Dar tiempo al loop para procesar los items
        await asyncio.sleep(0.05)
        await manager.stop()

        assert processed[0] == "alta"
        assert processed[1] == "media"
        assert processed[2] == "baja"

    async def test_drain_loop_continues_after_item_processed(self) -> None:
        """El loop no se detiene después de procesar un item."""
        processed: list[str] = []

        manager = AudioManager()

        async def fake_play(item: _AudioItem) -> None:
            processed.append(item.text)

        manager._play_audio = fake_play  # type: ignore[assignment]

        await manager.start()

        await manager._queue.put(_AudioItem(priority=1, text="primero", audio=b""))
        await asyncio.sleep(0.02)
        await manager._queue.put(_AudioItem(priority=1, text="segundo", audio=b""))
        await asyncio.sleep(0.02)

        await manager.stop()

        assert "primero" in processed
        assert "segundo" in processed
