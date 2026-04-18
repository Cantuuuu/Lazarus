"""
triggers/voice_trigger.py — Trigger de wake word + comando de voz.

Pipeline:
  1. sounddevice.InputStream captura audio a 16 kHz en chunks de 1280 samples
  2. openWakeWord evalúa cada chunk buscando el wake word configurado
  3. Al detectar: graba N segundos adicionales de audio
  4. faster-whisper transcribe el audio grabado
  5. parse_voice_command() extrae el intent (snapshot / record / stop / describe)
  6. Se emite Event(type=VOICE, metadata={command, transcript})

Portabilidad:
  - sounddevice: funciona en Windows y Linux (Orange Pi)
  - openWakeWord: usa ONNX internamente, portable
  - faster-whisper: eficiente en CPU; mismo código en Pi
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

import numpy as np
from loguru import logger

from core.events import Event, EventType, VoiceCommand
from services.camera import CameraService
from triggers.base import BaseTrigger
from utils.audio import normalize_audio, parse_voice_command

# Importaciones lazy para dar mensajes de error claros si no están instaladas
try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

try:
    import openwakeword
    from openwakeword.model import Model as OwwModel
    _OWW_AVAILABLE = True
except ImportError:
    _OWW_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False


# Mapa de string de comando a VoiceCommand enum
_COMMAND_MAP: dict[str, VoiceCommand] = {
    "snapshot":        VoiceCommand.SNAPSHOT,
    "start_recording": VoiceCommand.START_RECORDING,
    "stop_recording":  VoiceCommand.STOP_RECORDING,
    "describe":        VoiceCommand.DESCRIBE,
}


class VoiceTrigger(BaseTrigger):
    """Trigger activado por wake word seguido de un comando de voz.

    Args:
        event_queue: Cola compartida de eventos.
        config: Sección "voice_trigger" del config.yaml.
        camera: Instancia de CameraService para capturar frame al disparar.

    Example:
        trigger = VoiceTrigger(q, config["voice_trigger"], camera)
        trigger.start()
        # ... usuario dice "hey jarvis" + "captura" ...
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

        self._wake_word_model: str  = config.get("wake_word_model", "hey_jarvis")
        self._wake_threshold: float = config.get("wake_word_threshold", 0.5)
        self._recording_secs: float = config.get("recording_seconds", 4.0)
        self._whisper_model: str    = config.get("whisper_model", "base")
        self._whisper_lang: str     = config.get("whisper_language", "es")
        self._sample_rate: int      = config.get("sample_rate", 16000)

        # DECISION: chunk_size = 1280 samples = 80ms a 16kHz.
        # openWakeWord recomienda chunks de 80ms para su ventana de contexto.
        self._chunk_size: int = 1280

        self._oww_model: Any = None
        self._whisper: Any = None

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Verifica dependencias, carga modelos e inicia el thread."""
        self._check_dependencies()
        self._load_models()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._listen_loop,
            name="VoiceTrigger",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"VoiceTrigger iniciado: wake_word='{self._wake_word_model}' "
            f"threshold={self._wake_threshold} "
            f"whisper='{self._whisper_model}' lang='{self._whisper_lang}'"
        )

    def stop(self) -> None:
        """Detiene el thread de escucha."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("VoiceTrigger detenido")

    # ------------------------------------------------------------------
    # Carga de modelos
    # ------------------------------------------------------------------

    def _check_dependencies(self) -> None:
        missing = []
        if not _SD_AVAILABLE:
            missing.append("sounddevice")
        if not _OWW_AVAILABLE:
            missing.append("openwakeword")
        if not _WHISPER_AVAILABLE:
            missing.append("faster-whisper")
        if missing:
            raise ImportError(
                f"VoiceTrigger requiere: {missing}\n"
                f"Instala con: pip install {' '.join(missing)}"
            )

    def _load_models(self) -> None:
        """Carga openWakeWord y faster-whisper."""
        logger.info(f"Cargando openWakeWord (modelo: '{self._wake_word_model}')...")
        # DECISION: inference_framework="onnx" para portabilidad.
        # openWakeWord descarga automáticamente el modelo si no existe en caché.
        self._oww_model = OwwModel(
            wakeword_models=[self._wake_word_model],
            inference_framework="onnx",
        )
        logger.info(f"Cargando faster-whisper (modelo: '{self._whisper_model}')...")
        # DECISION: device="cpu" + int8 para compatibilidad universal.
        # En dev con GPU el modelo base es tan rápido en CPU que no vale la pena
        # la complejidad de detección de CUDA aquí.
        self._whisper = WhisperModel(
            self._whisper_model,
            device="cpu",
            compute_type="int8",
        )
        logger.info("Modelos de voz cargados")

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------

    def _listen_loop(self) -> None:
        """Abre el stream de micrófono y escucha continuamente."""
        try:
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._chunk_size,
            ) as stream:
                logger.info("VoiceTrigger: escuchando micrófono...")
                while not self._stop_event.is_set():
                    self._process_chunk(stream)
        except Exception:
            logger.exception("VoiceTrigger: error en el stream de audio")

        logger.debug("VoiceTrigger _listen_loop finalizado")

    def _process_chunk(self, stream: Any) -> None:
        """Lee un chunk del stream y lo pasa a openWakeWord."""
        try:
            audio_chunk, _ = stream.read(self._chunk_size)
        except Exception:
            logger.warning("VoiceTrigger: error leyendo chunk de audio")
            return

        # audio_chunk shape: (chunk_size, 1) -> (chunk_size,)
        audio_flat = audio_chunk[:, 0]

        # openWakeWord espera int16 o float32 normalizado
        prediction = self._oww_model.predict(audio_flat)

        # prediction es un dict: {wake_word_name: score}
        score = self._get_wake_score(prediction)

        if score >= self._wake_threshold:
            logger.info(
                f"[VoiceTrigger] Wake word detectado: "
                f"'{self._wake_word_model}' score={score:.3f}"
            )
            self._on_wake_word_detected(stream)

    def _get_wake_score(self, prediction: dict) -> float:
        """Extrae el score del wake word configurado del dict de predicciones."""
        # openWakeWord puede usar el nombre del modelo con o sin extensión
        for key, val in prediction.items():
            if self._wake_word_model.lower().replace("_", " ") in key.lower().replace("_", " "):
                return float(val)
        # Fallback: tomar el máximo score disponible
        if prediction:
            return float(max(prediction.values()))
        return 0.0

    # ------------------------------------------------------------------
    # Detección de wake word -> grabación -> transcripción
    # ------------------------------------------------------------------

    def _on_wake_word_detected(self, stream: Any) -> None:
        """Graba audio, transcribe y emite el evento."""
        # Limpiar el buffer interno de openWakeWord para evitar re-trigger
        self._oww_model.reset()

        # Graba N segundos
        logger.info(f"[VoiceTrigger] Grabando {self._recording_secs}s de comando...")
        audio_data = self._record_audio(stream, self._recording_secs)

        if audio_data is None or len(audio_data) == 0:
            logger.warning("[VoiceTrigger] Grabación vacía, ignorando")
            return

        # Transcribir
        transcript = self._transcribe(audio_data)
        if not transcript:
            logger.warning("[VoiceTrigger] Transcripción vacía, ignorando")
            return

        logger.info(f"[VoiceTrigger] Transcripción: '{transcript}'")

        # Parsear intent
        command_str = parse_voice_command(transcript)
        command = _COMMAND_MAP.get(command_str, VoiceCommand.DESCRIBE)
        logger.info(f"[VoiceTrigger] Comando: {command.value}")

        # Capturar frame
        frame = self._camera.get_latest_frame()
        buffer = self._camera.get_buffer_snapshot()

        event = Event(
            event_type=EventType.VOICE,
            frame=frame,
            frame_buffer=buffer if buffer else None,
            metadata={
                "command":    command,
                "transcript": transcript,
            },
        )
        self._emit_event(event)

    def _record_audio(self, stream: Any, duration: float) -> np.ndarray | None:
        """Graba `duration` segundos del stream de micrófono.

        Args:
            stream: sounddevice.InputStream ya abierto.
            duration: Segundos a grabar.

        Returns:
            Array float32 de shape (samples,) o None si falla.
        """
        total_samples = int(self._sample_rate * duration)
        chunks: list[np.ndarray] = []
        recorded = 0

        while recorded < total_samples and not self._stop_event.is_set():
            remaining = total_samples - recorded
            read_size = min(self._chunk_size, remaining)
            try:
                chunk, _ = stream.read(read_size)
                chunks.append(chunk[:, 0])
                recorded += read_size
            except Exception:
                logger.warning("[VoiceTrigger] Error leyendo chunk durante grabación")
                break

        if not chunks:
            return None
        return np.concatenate(chunks)

    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio float32 con faster-whisper.

        Args:
            audio: Array float32 normalizado a 16 kHz.

        Returns:
            Texto transcrito (puede ser vacío si no se detectó habla).
        """
        try:
            segments, info = self._whisper.transcribe(
                audio,
                language=self._whisper_lang,
                beam_size=5,
                vad_filter=True,          # filtra silencios automáticamente
                vad_parameters={"min_silence_duration_ms": 300},
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            logger.debug(
                f"[VoiceTrigger] Whisper: lang={info.language} "
                f"prob={info.language_probability:.2f} texto='{text}'"
            )
            return text
        except Exception:
            logger.exception("[VoiceTrigger] Error en transcripción")
            return ""
