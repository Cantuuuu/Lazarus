"""
scripts/process_video.py — Procesa un video pregrabado con el pipeline YOLO + voz.

Uso:
    python scripts/process_video.py --video ruta/al/video.mp4
    python scripts/process_video.py --video video.mp4 --config config.yaml
    python scripts/process_video.py --video video.mp4 --show   # preview en vivo
    python scripts/process_video.py --video video.mp4 --wake-word lazarus

Salida en output/run_YYYYMMDD_HHMMSS/:
    processed.mp4                    Video completo con bounding boxes dibujados
    frame_{N:06d}_{clase}.jpg        Frames importantes (los que dispararían trigger)
    frame_{N:06d}_{clase}.json       Metadata de cada frame importante
    frame_{N:06d}_{clase}_ctx_{i}.jpg  Frames de contexto temporal
    summary.json                     Resumen de la ejecucion completa
    voice_requests.json              Peticiones de voz detectadas (wake word + YOLO)
    voice_{N:04d}_ctx_{i}.jpg        Frames de contexto de cada peticion de voz

Reutiliza:
    - inference/yolo_onnx.py  (misma inferencia que en produccion)
    - triggers/yolo_trigger.py (mismas heuristicas + spatial enrichment)
    - config.yaml              (mismos parametros)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import deque
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

try:
    import requests as _requests_lib
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

import cv2
import numpy as np

# Agregar raiz del proyecto al path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.config_loader import load_config
from inference.yolo_onnx import YoloOnnx
from triggers.yolo_trigger import YoloTrigger


# ------------------------------------------------------------------
# Colores por prioridad (BGR)
# ------------------------------------------------------------------
PRIORITY_COLORS = {
    "urgent":      (0,   0,   255),   # rojo
    "important":   (0,   200, 255),   # amarillo
    "informative": (0,   200, 0),     # verde
    "scene_change": (255, 0,  200),   # magenta
    "count_change": (255, 128, 0),    # naranja
}
DEFAULT_COLOR = (180, 180, 180)

# Variaciones que Whisper produce al transcribir "Lazarus" en español.
# Se expande con cada video nuevo que confirme una variante.
# Formato de los valores: normalized (sin acentos, minúsculas).
#
# Confirmadas experimentalmente:
#   "laseros"  → video 1 (palabra única)
#   "lasalud"  → video 2 (bigrama "La" + "salud")
_WAKE_WORD_VARIANTS: set[str] = {
    # Forma canónica
    "lazarus",
    # Variantes de una sola palabra
    "lazaro", "lazaros", "lazaro's",
    "lasarus", "lasaro", "lasaros",
    "lazeros", "laseros", "laseiros",
    "lasseros", "laceros",
    # Variantes de bigrama (dos palabras unidas tras normalizar)
    "lasalud",   # "La salud" — confirmado video IMG_3280
    "lasaluz", "lasalu", "lasalus",  # variantes cercanas
}


# ------------------------------------------------------------------
# Argumentos
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Procesa un video con el pipeline YOLO + voz del asistente visual."
    )
    p.add_argument("--video",      required=True, help="Ruta al video de entrada")
    p.add_argument("--config",     default=str(ROOT / "config.yaml"), help="config.yaml")
    p.add_argument("--show",       action="store_true", help="Mostrar preview en tiempo real")
    p.add_argument("--output-fps", type=float, default=None,
                   help="FPS del video de salida (default: mismo que el original)")
    p.add_argument("--wake-word",  default="lazarus",
                   help="Palabra de activacion de voz (default: lazarus)")
    p.add_argument("--whisper-model", default=None,
                   help="Modelo Whisper a usar (default: el de config.yaml o 'base')")
    p.add_argument("--n8n-url", default=None,
                   help="URL del webhook n8n. Si se omite, no se hace POST.")
    p.add_argument("--user-id", default="lazarus_user_001",
                   help="ID de usuario para el POST (default: lazarus_user_001)")
    p.add_argument("--mode", default="voice", choices=["voice", "manual"],
                   help="voice: detecta wake word y hace un POST por comando. "
                        "manual: transcribe todo el audio y hace un solo POST. "
                        "(default: voice)")
    p.add_argument("--yolo-every", type=int, default=1,
                   help="Ejecutar YOLO cada N frames (default: 1 = todos). "
                        "Usar 3 o 5 para videos largos.")
    return p.parse_args()


# ------------------------------------------------------------------
# Helpers de texto
# ------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Minusculas + elimina acentos/diacriticos."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _is_wake_word(word: str, wake_word: str, threshold: float = 0.60) -> bool:
    """True si la palabra es el wake word (exacto o variacion fuzzy).

    Primero chequea contra el conjunto de variantes conocidas,
    luego usa SequenceMatcher para tolerar pequeños errores de Whisper.
    """
    clean = _normalize(word.strip(".,!?¿¡\"'"))
    target = _normalize(wake_word)

    # Chequeo exacto contra variantes conocidas
    if clean in _WAKE_WORD_VARIANTS:
        return True

    # Fuzzy match: util cuando Whisper inventa algo parecido
    ratio = SequenceMatcher(None, clean, target).ratio()
    return ratio >= threshold


def _is_wake_word_bigram(word1: str, word2: str, wake_word: str, threshold: float = 0.55) -> bool:
    """True si la UNION de dos palabras consecutivas es el wake word.

    Cubre casos como 'La salud' -> 'lasalud' ~ 'lazarus' (Whisper parte
    la palabra en dos al transcribir en español).
    """
    joined = _normalize(word1.strip(".,!?¿¡\"'") + word2.strip(".,!?¿¡\"'"))
    target = _normalize(wake_word)

    if joined in _WAKE_WORD_VARIANTS:
        return True

    ratio = SequenceMatcher(None, joined, target).ratio()
    return ratio >= threshold


# ------------------------------------------------------------------
# Clase principal
# ------------------------------------------------------------------

class VideoProcessor:
    """Procesa un video frame a frame con las mismas heuristicas que YoloTrigger
    y, opcionalmente, extrae peticiones de voz si el video tiene audio.

    Args:
        video_path: Ruta al video de entrada.
        config: Dict de configuracion (yolo_trigger section).
        output_dir: Carpeta donde guardar los resultados.
        show_preview: Si True, abre ventana con el video procesado.
        output_fps: FPS del video de salida. None = mismo que el original.
        wake_word: Palabra de activacion de voz (default: "lazarus").
        whisper_model: Nombre del modelo Whisper (base, small, medium).
    """

    def __init__(
        self,
        video_path: Path,
        config: dict,
        output_dir: Path,
        show_preview: bool = False,
        output_fps: float | None = None,
        wake_word: str = "lazarus",
        whisper_model: str = "base",
        n8n_url: str | None = None,
        user_id: str = "lazarus_user_001",
        mode: str = "voice",
        yolo_every: int = 1,
    ) -> None:
        self._video_path = video_path
        self._cfg = config
        self._output_dir = output_dir
        self._show = show_preview
        self._output_fps = output_fps
        self._wake_word = wake_word.lower()
        self._whisper_model = whisper_model
        self._n8n_url = n8n_url
        self._user_id = user_id
        self._mode = mode
        self._yolo_every = max(1, yolo_every)

        # Configuracion YOLO (misma que YoloTrigger)
        yolo_cfg = config["yolo_trigger"]
        self._whitelist: set[str] = set(yolo_cfg.get("class_whitelist", []))
        self._cooldown_secs: float = yolo_cfg.get("cooldown_seconds", 5.0)
        self._stability_m: int = yolo_cfg.get("stability_frames", 3)
        self._stability_k: int = yolo_cfg.get("stability_window", 5)
        self._class_priority: dict[str, str] = yolo_cfg.get("class_priority", {})
        self._priority_cooldowns: dict[str, float] = yolo_cfg.get(
            "priority_cooldowns", {"urgent": 2, "important": 5, "informative": 15}
        )
        self._scene_change_enabled: bool = yolo_cfg.get("scene_change_enabled", True)
        self._scene_change_threshold: float = yolo_cfg.get("scene_change_threshold", 0.45)
        self._scene_change_cooldown: float = yolo_cfg.get("scene_change_cooldown", 10.0)
        self._count_change_enabled: bool = yolo_cfg.get("count_change_enabled", True)
        self._count_change_delta: int = yolo_cfg.get("count_change_delta", 2)
        self._count_change_cooldown: float = yolo_cfg.get("count_change_cooldown", 10.0)
        self._context_count: int = yolo_cfg.get("context_frames_count", 5)

        # Estado de heuristicas
        self._cooldown_map: dict[str, float] = {}
        self._stability_window: deque[set[str]] = deque(maxlen=self._stability_k)
        self._last_classes: set[str] = set()
        self._last_histogram: np.ndarray | None = None
        self._last_scene_change_time: float = 0.0
        self._last_class_counts: dict[str, int] = {}
        self._last_count_change_time: float = 0.0

        # Ring buffer de frames para contexto (equivalente al de CameraService)
        buffer_maxlen = int(30 * config.get("camera", {}).get("buffer_seconds", 10))
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=buffer_maxlen)

        # Detecciones indexadas por frame (para cruzar con eventos de voz)
        # {frame_idx: [deteccion_dict, ...]}
        self._detections_by_frame: dict[int, list[dict]] = {}

        # Estadisticas
        self._trigger_log: list[dict] = []

        # Se asigna en run() para que _process_audio pueda usarlo
        self._src_fps: float = 30.0

    def run(self) -> dict:
        """Ejecuta el procesamiento completo del video (YOLO + voz).

        Returns:
            Diccionario con el resumen de la ejecucion.
        """
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {self._video_path}")

        # Propiedades del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_fps      = self._output_fps or src_fps
        duration_s   = total_frames / src_fps if src_fps > 0 else 0

        self._src_fps = src_fps  # Necesario para process_audio

        print(f"\n  Video    : {self._video_path.name}")
        print(f"  Resolucion: {width}x{height}  FPS: {src_fps:.1f}  Duracion: {duration_s:.1f}s  Frames: {total_frames}")
        print(f"  Output   : {self._output_dir}\n")

        # Writer del video de salida
        out_path = self._output_dir / "processed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (width, height))

        # Cargar modelo
        yolo_cfg = self._cfg["yolo_trigger"]
        model_path = ROOT / yolo_cfg.get("model_path", "models/yolov8n.onnx")
        print(f"  Cargando modelo: {model_path.name}...")
        model = YoloOnnx(
            model_path=model_path,
            confidence_threshold=yolo_cfg.get("confidence_threshold", 0.6),
        )
        yolo_every_str = f"cada {self._yolo_every} frames" if self._yolo_every > 1 else "todos los frames"
        print(f"  Modelo listo. Modo: {self._mode.upper()}  YOLO: {yolo_every_str}\n")

        t_start = time.time()
        frame_idx = 0
        triggers_count = 0
        last_detections: list = []  # para dibujo continuo sin parpadeo

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Tiempo de video en segundos (para cooldowns)
            video_time = frame_idx / src_fps if src_fps > 0 else frame_idx

            # Acumular buffer de contexto
            self._frame_buffer.append(frame.copy())

            # YOLO: solo en frames multiplo de yolo_every
            if frame_idx % self._yolo_every == 0:
                detections = model.detect(frame)
                relevant   = [d for d in detections if d["class_name"] in self._whitelist]
                YoloTrigger._enrich_spatial(relevant, height, width)
                self._detections_by_frame[frame_idx] = [d.copy() for d in relevant]
                last_detections = detections  # actualizar para dibujo
            else:
                # Reusar ultima deteccion para dibujar sin parpadeo
                detections = last_detections
                relevant   = []

            # Heuristicas
            current_classes = {d["class_name"] for d in relevant}
            self._stability_window.append(current_classes)
            classes_to_fire = self._evaluate_heuristics(current_classes, video_time)

            # Scene change
            scene_fired = False
            if self._scene_change_enabled:
                scene_fired = self._check_scene_change(frame, relevant, video_time)

            # Count change
            count_fired = False
            if self._count_change_enabled:
                current_counts: dict[str, int] = {}
                for d in relevant:
                    current_counts[d["class_name"]] = current_counts.get(d["class_name"], 0) + 1
                count_fired = self._check_count_change(current_counts, relevant, frame, frame_idx, video_time)

            self._last_classes = current_classes

            # Guardar frames importantes
            for class_name in classes_to_fire:
                class_dets = [d for d in relevant if d["class_name"] == class_name]
                self._save_trigger_frame(frame, frame_idx, class_name, class_dets, video_time)
                self._cooldown_map[class_name] = video_time
                triggers_count += 1

            # Dibujar bounding boxes sobre el frame
            annotated = self._draw_detections(frame.copy(), detections, classes_to_fire)
            annotated = self._draw_hud(annotated, frame_idx, total_frames, video_time, triggers_count)

            writer.write(annotated)

            if self._show:
                cv2.imshow("Vision Assistant - Video Processor", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n  [q] Interrumpido por el usuario")
                    break

            # Progreso
            frame_idx += 1
            if frame_idx % 30 == 0 or frame_idx == total_frames:
                pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                elapsed = time.time() - t_start
                print(f"  [{pct:5.1f}%] frame {frame_idx}/{total_frames}  "
                      f"triggers={triggers_count}  t={elapsed:.1f}s", end="\r")

        cap.release()
        writer.release()
        if self._show:
            cv2.destroyAllWindows()

        elapsed_total = time.time() - t_start
        print(f"\n\n  Procesamiento YOLO completo en {elapsed_total:.1f}s")

        # ------------------------------------------------------------------
        # Procesamiento de audio
        # ------------------------------------------------------------------
        voice_summary = []

        if self._mode == "manual":
            manual_req = self._process_audio_manual(src_fps, frame_idx)
            if manual_req:
                voice_summary = [manual_req]
                mr_path = self._output_dir / "manual_request.json"
                with mr_path.open("w", encoding="utf-8") as f:
                    json.dump(manual_req, f, indent=2, ensure_ascii=False)
                print(f"\n  [MANUAL] Transcripcion lista  ->  {mr_path.name}")
                print(f"    Texto ({len(manual_req['transcription'])} chars): "
                      f"\"{manual_req['transcription'][:120]}...\"")
                print(f"    Imagenes: {len(manual_req['context_frame_paths'])} frames")
                if self._n8n_url:
                    self._post_to_n8n_manual(manual_req)
        else:
            voice_requests = self._process_audio()
            if voice_requests is not None:
                voice_summary = voice_requests
                vr_path = self._output_dir / "voice_requests.json"
                with vr_path.open("w", encoding="utf-8") as f:
                    json.dump(voice_requests, f, indent=2, ensure_ascii=False)
                print(f"\n  Peticiones de voz: {len(voice_requests)}  ->  {vr_path.name}")
                for i, vr in enumerate(voice_requests):
                    print(f"    [{i+1}] t={vr['timestamp_s']:.1f}s  \"{vr['command']}\"")
                    if vr["yolo_detections_at_trigger"]:
                        dets_str = ", ".join(
                            f"{d['class_name']} ({d.get('zone','?')}/{d.get('proximity','?')})"
                            for d in vr["yolo_detections_at_trigger"]
                        )
                        print(f"         YOLO: {dets_str}")
                if self._n8n_url:
                    for vr in voice_requests:
                        self._post_to_n8n(vr)
            else:
                print("\n  (Sin procesamiento de audio: ffmpeg no disponible o video sin audio)")

        # Guardar summary
        summary = {
            "video":           str(self._video_path),
            "frames_total":    frame_idx,
            "duration_s":      round(frame_idx / src_fps, 2) if src_fps > 0 else 0,
            "triggers_total":  triggers_count,
            "processed_at":    datetime.now().isoformat(),
            "output_dir":      str(self._output_dir),
            "triggers":        self._trigger_log,
            "voice_requests":  voice_summary,
        }
        summary_path = self._output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary

    # ------------------------------------------------------------------
    # Procesamiento de audio
    # ------------------------------------------------------------------

    def _process_audio(self) -> list[dict] | None:
        """Extrae audio del video, transcribe, y encuentra peticiones de voz.

        Returns:
            Lista de dicts con cada peticion detectada, o None si no es posible.
        """
        # 1. Verificar que ffmpeg esté disponible
        ffmpeg_exe = _get_ffmpeg_exe()
        if not ffmpeg_exe:
            print("\n  [AUDIO] ffmpeg no encontrado. Instala ffmpeg para procesar audio.")
            return None

        # 2. Verificar que faster-whisper esté disponible
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("\n  [AUDIO] faster-whisper no instalado. Omitiendo audio.")
            return None

        print(f"\n  [AUDIO] Extrayendo audio del video...")

        # 3. Extraer audio a WAV temporal (16kHz mono — formato que espera Whisper)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            result = subprocess.run(
                [
                    ffmpeg_exe, "-y",
                    "-i", str(self._video_path),
                    "-ar", "16000",    # 16 kHz
                    "-ac", "1",        # mono
                    "-vn",             # sin video
                    str(tmp_path),
                ],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            print("\n  [AUDIO] ffmpeg no encontrado en PATH.")
            return None

        if result.returncode != 0:
            # ffmpeg falla cuando el video no tiene pista de audio
            if "does not contain any stream" in result.stderr or "Invalid data" in result.stderr:
                print("\n  [AUDIO] El video no tiene pista de audio. Omitiendo.")
            else:
                print(f"\n  [AUDIO] Error extrayendo audio:\n{result.stderr[-300:]}")
            tmp_path.unlink(missing_ok=True)
            return None

        if not tmp_path.exists() or tmp_path.stat().st_size < 1000:
            print("\n  [AUDIO] Audio extraido vacio o inexistente.")
            tmp_path.unlink(missing_ok=True)
            return None

        print(f"  [AUDIO] Audio extraido. Transcribiendo con Whisper ({self._whisper_model})...")

        # 4. Transcribir con word timestamps
        try:
            whisper = WhisperModel(self._whisper_model, device="cpu", compute_type="int8")
            segments, info = whisper.transcribe(
                str(tmp_path),
                language="es",
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            segments = list(segments)  # materializar generador antes de borrar tmp
        except Exception as e:
            print(f"\n  [AUDIO] Error en transcripcion: {e}")
            tmp_path.unlink(missing_ok=True)
            return None
        finally:
            tmp_path.unlink(missing_ok=True)

        print(f"  [AUDIO] Transcripcion lista. Buscando wake word '{self._wake_word}'...")

        # 5. Buscar wake word en las palabras transcritas
        voice_events = self._find_wake_word_events(segments)

        if not voice_events:
            print(f"  [AUDIO] No se encontro '{self._wake_word}' en el audio.")
            return []

        # 6. Cruzar con detecciones YOLO y generar payload
        requests = []
        for idx, event in enumerate(voice_events):
            ts = event["timestamp_s"]
            command = event["command"]
            raw_word = event["raw_word"]

            # Frame en ese timestamp
            frame_num = int(ts * self._src_fps)

            # Detecciones YOLO activas en ese frame (o el mas cercano registrado)
            yolo_dets = self._get_detections_near_frame(frame_num)

            # Contexto YOLO (solo para el JSON local, no va en el input del POST)
            yolo_context = _format_yolo_context(yolo_dets)

            # Extraer y guardar frames de contexto del video (max 5)
            ctx_frame_paths = self._extract_voice_context_frames(idx, ts)

            payload = {
                "request_index":              idx,
                "timestamp_s":                round(ts, 2),
                "wake_word_detected":         raw_word,
                "command":                    command,
                "user_id":                    self._user_id,
                "yolo_detections_at_trigger": yolo_dets,
                "yolo_context_text":          yolo_context,
                "context_frame_paths":        ctx_frame_paths,
            }
            requests.append(payload)

            print(f"\n  [VOZ #{idx+1}] t={ts:.1f}s")
            print(f"    Wake word : '{raw_word}'")
            print(f"    Comando   : \"{command}\"")
            print(f"    User ID   : {self._user_id}")
            print(f"    YOLO ctx  : {yolo_context or '(sin detecciones)'}")
            print(f"    Fotos     : {len(ctx_frame_paths)} frames de contexto")

        return requests

    def _process_audio_manual(self, src_fps: float, total_frames: int) -> dict | None:
        """Modo manual: transcribe el audio completo del video sin buscar wake word.

        Returns:
            Dict con transcripcion completa + paths de imagenes representativas,
            o None si no hay audio o faltan dependencias.
        """
        ffmpeg_exe = _get_ffmpeg_exe()
        if not ffmpeg_exe:
            print("\n  [AUDIO] ffmpeg no encontrado.")
            return None

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("\n  [AUDIO] faster-whisper no instalado.")
            return None

        print("\n  [MANUAL] Extrayendo audio...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            result = subprocess.run(
                [ffmpeg_exe, "-y", "-i", str(self._video_path),
                 "-ar", "16000", "-ac", "1", "-vn", str(tmp_path)],
                capture_output=True, text=True,
            )
            if result.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size < 1000:
                print("\n  [MANUAL] No se pudo extraer audio (video sin pista de audio?).")
                tmp_path.unlink(missing_ok=True)
                return None

            print(f"  [MANUAL] Transcribiendo con Whisper ({self._whisper_model})...")
            whisper = WhisperModel(self._whisper_model, device="cpu", compute_type="int8")
            segments, info = whisper.transcribe(
                str(tmp_path),
                language="es",
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            transcription = " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            tmp_path.unlink(missing_ok=True)

        if not transcription:
            print("\n  [MANUAL] Transcripcion vacia.")
            return None

        # Frames representativos: primero los triggers YOLO, luego rellena con
        # frames equidistantes hasta tener 5
        trigger_jpgs = sorted(self._output_dir.glob("frame_*.jpg"))
        # Excluir ctx frames
        trigger_jpgs = [p for p in trigger_jpgs if "_ctx_" not in p.name]

        selected_paths: list[str] = [p.name for p in trigger_jpgs[:5]]

        # Si hay menos de 5 triggers, completar con frames equidistantes del video
        if len(selected_paths) < 5:
            needed = 5 - len(selected_paths)
            cap = cv2.VideoCapture(str(self._video_path))
            indices = list(np.linspace(0, total_frames - 1, needed + 2, dtype=int))[1:-1]
            for i, fidx in enumerate(indices[:needed]):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
                ret, frame = cap.read()
                if ret:
                    fname = f"manual_frame_{i:03d}.jpg"
                    cv2.imwrite(str(self._output_dir / fname), frame)
                    selected_paths.append(fname)
            cap.release()

        return {
            "mode":                 "manual",
            "user_id":              self._user_id,
            "video":                self._video_path.name,
            "transcription":        transcription,
            "transcription_chars":  len(transcription),
            "context_frame_paths":  selected_paths,
            "yolo_triggers":        self._trigger_log,
        }

    def _post_to_n8n_manual(self, manual_request: dict) -> None:
        """Envia el trigger manual a n8n.

        Payload multipart:
            input   : transcripcion completa del video
            user_id : identificador del usuario
            img0..4 : hasta 5 frames representativos
        """
        if not _REQUESTS_AVAILABLE:
            print("\n  [POST] 'requests' no instalado.")
            return

        transcription = manual_request["transcription"]
        ctx_paths = manual_request.get("context_frame_paths", [])[:5]

        data = {
            "input":   transcription,
            "user_id": self._user_id,
        }

        file_handles = []
        files = []
        try:
            for i, fname in enumerate(ctx_paths):
                full_path = self._output_dir / fname
                if full_path.exists():
                    fh = open(full_path, "rb")
                    file_handles.append(fh)
                    files.append((f"img{i}", (fname, fh, "image/jpeg")))

            print(f"\n  [POST] Enviando trigger manual a n8n...")
            print(f"         user_id   : {self._user_id}")
            print(f"         input     : \"{transcription[:100]}...\"")
            print(f"         imagenes  : {len(files)}")

            resp = _requests_lib.post(
                self._n8n_url,
                data=data,
                files=files if files else None,
                timeout=30,
            )
            print(f"         status    : {resp.status_code}")
            print(f"         respuesta : {resp.text[:300]}")

        except Exception as e:
            print(f"\n  [POST] Error: {e}")
        finally:
            for fh in file_handles:
                fh.close()

    def _post_to_n8n(self, voice_request: dict) -> None:
        """Envia una peticion de voz al webhook de n8n.

        Payload multipart:
            input   : texto del comando
            user_id : identificador del usuario
            img0..4 : hasta 5 frames de contexto (JPGs)
        """
        if not _REQUESTS_AVAILABLE:
            print("\n  [POST] 'requests' no instalado. Corre: pip install requests")
            return

        command = voice_request["command"]
        ctx_paths = voice_request.get("context_frame_paths", [])[:5]

        data = {
            "input":   command,
            "user_id": self._user_id,
        }

        file_handles = []
        files = []
        try:
            for i, fname in enumerate(ctx_paths):
                full_path = self._output_dir / fname
                if full_path.exists():
                    fh = open(full_path, "rb")
                    file_handles.append(fh)
                    files.append((f"img{i}", (fname, fh, "image/jpeg")))

            print(f"\n  [POST] Enviando a n8n...")
            print(f"         input   : \"{command}\"")
            print(f"         user_id : {self._user_id}")
            print(f"         imagenes: {len(files)}")

            resp = _requests_lib.post(
                self._n8n_url,
                data=data,
                files=files if files else None,
                timeout=30,
            )

            print(f"         status  : {resp.status_code}")
            print(f"         respuesta: {resp.text[:300]}")

        except Exception as e:
            print(f"\n  [POST] Error: {e}")
        finally:
            for fh in file_handles:
                fh.close()

    def _find_wake_word_events(self, segments: list) -> list[dict]:
        """Encuentra ocurrencias del wake word con su comando y timestamp.

        Itera palabra por palabra. Al encontrar el wake word, toma todas
        las palabras siguientes del mismo segmento como el "comando".

        Returns:
            Lista de {timestamp_s, raw_word, command}
        """
        events = []
        for seg in segments:
            if not hasattr(seg, "words") or not seg.words:
                continue

            words = seg.words
            for i, w in enumerate(words):
                word_text = getattr(w, "word", "").strip()
                if not word_text:
                    continue

                # Chequeo de palabra simple
                single_match = _is_wake_word(word_text, self._wake_word)

                # Chequeo de bigrama (palabra actual + siguiente)
                bigram_match = False
                if not single_match and i + 1 < len(words):
                    next_text = getattr(words[i + 1], "word", "").strip()
                    bigram_match = _is_wake_word_bigram(word_text, next_text, self._wake_word)

                if not single_match and not bigram_match:
                    continue

                wake_time = getattr(w, "start", seg.start)

                # Si fue bigrama, el comando empieza desde la palabra i+2
                cmd_start = i + 2 if bigram_match else i + 1

                # Comando = palabras que siguen al wake word en el mismo segmento
                remaining = [
                    getattr(rw, "word", "").strip()
                    for rw in words[cmd_start:]
                    if getattr(rw, "word", "").strip()
                ]
                command = " ".join(remaining).strip(" .,!?¿¡")

                # Si no hay palabras en el mismo segmento, usar el texto del segmento
                if not command and seg.text:
                    raw_seg = seg.text.strip()
                    lower_seg = raw_seg.lower()
                    for variant in _WAKE_WORD_VARIANTS | {self._wake_word}:
                        if lower_seg.startswith(variant):
                            command = raw_seg[len(variant):].strip(" .,!?¿¡")
                            break
                    if not command:
                        command = raw_seg

                events.append({
                    "timestamp_s": wake_time,
                    "raw_word":    word_text,
                    "command":     command if command else "(sin comando)",
                })

        return events

    def _get_detections_near_frame(self, frame_num: int) -> list[dict]:
        """Retorna detecciones YOLO del frame mas cercano al indicado."""
        if frame_num in self._detections_by_frame:
            return self._detections_by_frame[frame_num]

        # Buscar el frame registrado mas cercano
        if not self._detections_by_frame:
            return []

        closest = min(
            self._detections_by_frame.keys(),
            key=lambda k: abs(k - frame_num),
        )
        return self._detections_by_frame[closest]

    def _extract_voice_context_frames(
        self, request_idx: int, timestamp_s: float, before_s: float = 3.0
    ) -> list[str]:
        """Extrae N frames de contexto alrededor del timestamp y los guarda como JPG.

        Abre el video, busca frames desde (timestamp - before_s) hasta timestamp,
        selecciona context_count frames equidistantes y los guarda.

        Returns:
            Lista de rutas relativas (str) de los JPGs guardados.
        """
        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            return []

        start_s = max(0.0, timestamp_s - before_s)
        start_frame = int(start_s * self._src_fps)
        end_frame = int(timestamp_s * self._src_fps)

        # Seleccionar indices equidistantes
        n = self._context_count
        if end_frame <= start_frame:
            frame_indices = [end_frame]
        else:
            frame_indices = list(
                np.linspace(start_frame, end_frame, min(n, end_frame - start_frame + 1), dtype=int)
            )

        saved_paths: list[str] = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])

        prev_idx = frame_indices[0]
        for i, fidx in enumerate(frame_indices):
            # Avanzar al frame correcto
            skip = fidx - prev_idx
            for _ in range(skip):
                cap.read()
            prev_idx = fidx + 1

            ret, frame = cap.read()
            if not ret:
                break

            fname = f"voice_{request_idx:04d}_ctx_{i}.jpg"
            fpath = self._output_dir / fname
            cv2.imwrite(str(fpath), frame)
            saved_paths.append(fname)

        cap.release()
        return saved_paths

    # ------------------------------------------------------------------
    # Heuristicas (equivalentes a YoloTrigger, con tiempo de video)
    # ------------------------------------------------------------------

    def _evaluate_heuristics(self, current_classes: set[str], now: float) -> set[str]:
        candidates: set[str] = set()
        for class_name in current_classes:
            if class_name in self._last_classes:
                continue
            last_fired = self._cooldown_map.get(class_name, -999.0)
            cooldown = self._get_cooldown(class_name)
            if (now - last_fired) < cooldown:
                continue
            appearances = sum(1 for fc in self._stability_window if class_name in fc)
            if appearances < self._stability_m:
                continue
            candidates.add(class_name)
        return candidates

    def _get_cooldown(self, class_name: str) -> float:
        priority = self._class_priority.get(class_name, "informative")
        return self._priority_cooldowns.get(priority, self._cooldown_secs)

    def _get_priority(self, class_name: str) -> str:
        return self._class_priority.get(class_name, "informative")

    def _check_scene_change(
        self, frame: np.ndarray, dets: list[dict], now: float
    ) -> bool:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        fired = False
        if self._last_histogram is not None:
            diff = cv2.compareHist(self._last_histogram, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > self._scene_change_threshold and (now - self._last_scene_change_time) >= self._scene_change_cooldown:
                frame_idx = len(self._frame_buffer) - 1
                self._save_trigger_frame(frame, frame_idx, "scene_change", dets, now)
                self._last_scene_change_time = now
                fired = True
        self._last_histogram = hist
        return fired

    def _check_count_change(
        self,
        current_counts: dict[str, int],
        dets: list[dict],
        frame: np.ndarray,
        frame_idx: int,
        now: float,
    ) -> bool:
        if (now - self._last_count_change_time) < self._count_change_cooldown:
            self._last_class_counts = current_counts.copy()
            return False
        fired = False
        all_classes = set(current_counts) | set(self._last_class_counts)
        for cls in all_classes:
            prev = self._last_class_counts.get(cls, 0)
            curr = current_counts.get(cls, 0)
            if abs(curr - prev) >= self._count_change_delta:
                self._save_trigger_frame(
                    frame, frame_idx, "count_change", dets, now,
                    extra={"changed_class": cls, "prev_count": prev, "new_count": curr},
                )
                self._last_count_change_time = now
                fired = True
                break
        self._last_class_counts = current_counts.copy()
        return fired

    # ------------------------------------------------------------------
    # Guardado de frames importantes
    # ------------------------------------------------------------------

    def _save_trigger_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        class_name: str,
        detections: list[dict],
        video_time: float,
        extra: dict | None = None,
    ) -> None:
        """Guarda el frame + context frames + JSON de un trigger."""
        stem = f"frame_{frame_idx:06d}_{class_name}"
        priority = self._get_priority(class_name)

        # Frame principal anotado
        annotated = self._draw_detections(frame.copy(), detections, {class_name})
        jpg_path = self._output_dir / f"{stem}.jpg"
        cv2.imwrite(str(jpg_path), annotated)

        # Context frames del buffer
        ctx_frames = self._select_context_frames()
        for i, cf in enumerate(ctx_frames):
            cv2.imwrite(str(self._output_dir / f"{stem}_ctx_{i}.jpg"), cf)

        # Metadata JSON
        metadata = {
            "frame_index":         frame_idx,
            "video_time_s":        round(video_time, 3),
            "trigger_class":       class_name,
            "priority":            priority,
            "context_frames":      len(ctx_frames),
            "detections":          detections,
        }
        if extra:
            metadata.update(extra)

        json_path = self._output_dir / f"{stem}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Log
        log_entry = {"frame": frame_idx, "time_s": round(video_time, 2),
                     "trigger": class_name, "priority": priority}
        self._trigger_log.append(log_entry)
        print(f"\n  TRIGGER  frame={frame_idx:6d}  t={video_time:6.1f}s  "
              f"clase='{class_name}'  priority={priority}  "
              f"n_dets={len(detections)}", end="")

    def _select_context_frames(self) -> list[np.ndarray]:
        buf = list(self._frame_buffer)
        if not buf:
            return []
        if len(buf) <= self._context_count:
            return [f.copy() for f in buf]
        indices = np.linspace(0, len(buf) - 1, self._context_count, dtype=int)
        return [buf[i].copy() for i in indices]

    # ------------------------------------------------------------------
    # Dibujo de anotaciones
    # ------------------------------------------------------------------

    def _draw_detections(
        self,
        frame: np.ndarray,
        detections: list[dict],
        triggered_classes: set[str],
    ) -> np.ndarray:
        """Dibuja bboxes sobre el frame. Los triggers tienen borde mas grueso."""
        for d in detections:
            class_name = d["class_name"]
            conf       = d["confidence"]
            x1, y1, x2, y2 = d["bbox"]
            zone       = d.get("zone", "")
            proximity  = d.get("proximity", "")
            priority   = self._get_priority(class_name)
            color      = PRIORITY_COLORS.get(priority, DEFAULT_COLOR)
            thickness  = 3 if class_name in triggered_classes else 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"{class_name} {conf:.2f}"
            if zone:
                label += f" | {zone}"
            if proximity:
                label += f" | {proximity}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - 6, th + 4)
            cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 4, label_y), color, -1)
            cv2.putText(
                frame, label,
                (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA,
            )

        return frame

    def _draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        total: int,
        video_time: float,
        triggers: int,
    ) -> np.ndarray:
        """Dibuja HUD con estadisticas en la esquina superior izquierda."""
        lines = [
            f"Frame: {frame_idx}/{total}",
            f"Time:  {video_time:.1f}s",
            f"Triggers: {triggers}",
        ]
        for i, line in enumerate(lines):
            y = 20 + i * 20
            cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return frame


# ------------------------------------------------------------------
# Helpers de audio
# ------------------------------------------------------------------

def _get_ffmpeg_exe() -> str | None:
    """Retorna la ruta al ejecutable ffmpeg, o None si no está disponible.

    Busca en orden:
    1. ffmpeg en el PATH del sistema
    2. ffmpeg empaquetado con imageio-ffmpeg (instalado via pip)
    """
    # 1. Sistema
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        if r.returncode == 0:
            return "ffmpeg"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 2. imageio-ffmpeg
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        exe = get_ffmpeg_exe()
        if exe:
            return exe
    except Exception:
        pass

    return None


def _ffmpeg_available() -> bool:
    """True si ffmpeg está disponible (sistema o imageio-ffmpeg)."""
    return _get_ffmpeg_exe() is not None


def _format_yolo_context(detections: list[dict]) -> str:
    """Formatea detecciones YOLO como texto legible para el campo 'input' de n8n.

    Ejemplo: "person (left/near), door (right/medium), chair (center/far)"
    """
    if not detections:
        return ""
    parts = []
    for d in detections:
        name = d.get("class_name", "?")
        zone = d.get("zone", "?")
        prox = d.get("proximity", "?")
        parts.append(f"{name} ({zone}/{prox})")
    return ", ".join(parts)


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: No se encontro el video: {video_path}")
        sys.exit(1)

    config = load_config(args.config)

    # Modelo Whisper: argumento > config > default "base"
    whisper_model = (
        args.whisper_model
        or config.get("voice_trigger", {}).get("whisper_model", "base")
    )

    # Carpeta de salida unica por ejecucion
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "output" / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Vision Assistant - Video Processor")
    print("=" * 55)

    processor = VideoProcessor(
        video_path=video_path,
        config=config,
        output_dir=output_dir,
        show_preview=args.show,
        output_fps=args.output_fps,
        wake_word=args.wake_word,
        whisper_model=whisper_model,
        n8n_url=args.n8n_url,
        user_id=args.user_id,
        mode=args.mode,
        yolo_every=args.yolo_every,
    )

    summary = processor.run()

    print(f"\n  Resultados en: {output_dir}")
    print(f"  Video procesado: {output_dir / 'processed.mp4'}")
    print(f"  Triggers YOLO:   {summary['triggers_total']}")
    print(f"  Peticiones voz:  {len(summary.get('voice_requests', []))}")
    print(f"  Summary: {output_dir / 'summary.json'}")
    if (output_dir / "voice_requests.json").exists():
        print(f"  Voz:     {output_dir / 'voice_requests.json'}")
    print()

    if summary["triggers"]:
        print("  Eventos YOLO detectados:")
        for t in summary["triggers"]:
            print(f"    t={t['time_s']:6.1f}s  frame={t['frame']:6d}  "
                  f"{t['trigger']:<20} [{t['priority']}]")

    if summary.get("voice_requests"):
        print("\n  Peticiones de voz/manual:")
        for vr in summary["voice_requests"]:
            if vr.get("mode") == "manual":
                txt = vr.get("transcription", "")
                print(f"    [MANUAL] {len(txt)} chars: \"{txt[:100]}...\"")
            else:
                print(f"    t={vr['timestamp_s']:6.1f}s  \"{vr['command']}\"")
                if vr.get("yolo_context_text"):
                    print(f"             YOLO: {vr['yolo_context_text']}")


if __name__ == "__main__":
    main()
