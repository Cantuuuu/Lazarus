# Codemap: Sistema de Edge Computing — Lazarillo

**Last Updated:** 2026-04-18  
**Entry Points:** `main.py`, `core/orchestrator.py`, `modes/proactive.py`

## Visión General

Lazarillo es un asistente visual de IA offline-first para personas ciegas. La arquitectura de edge computing optimiza la latencia (inferencia local en NPU) combinando:

- **Hardware:** Orange Pi 5 Max (RK3588 NPU + 8 cores ARM)
- **Modelos:** YOLOv8n (12MB ONNX → 4MB RKNN)
- **Stack:** FastAPI + asyncio + onnxruntime/rknnlite
- **LLM Cloud:** Gemini 1.5 Flash (consultas, fallback local)

---

## Arquitectura General

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│ • Raspberry Pi Zero W → MJPEG stream                           │
│ • StreamReader (services/stream_reader.py) → httpx async       │
│ • Buffer JPEG + descompresión con cv2.imdecode()              │
└────────────────────────────────┬────────────────────────────────┘
                                 │ np.ndarray (640×480, BGR)
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│              INFERENCE ENGINE (core/detector.py)                │
├─────────────────────────────────────────────────────────────────┤
│ DetectorBackend (Protocol)                                      │
│ ├─ RKNNDetector → 30ms @ 640×640 (producción)                  │
│ ├─ ONNXDetector → 205ms @ 640×640 (fallback)                   │
│ └─ MockDetector → <1ms (desarrollo)                            │
│                                                                  │
│ Pipeline:                                                        │
│   Resize(640×640) → RGB normalization → Inference              │
│   Output: (batch=1, channels=84, anchors=8400)                 │
│   ↓                                                              │
│   Post-procesado: _postprocess() o _postprocess_dfl()          │
│   • Threshold (conf≥0.4), NMS (IoU=0.45), scaling              │
│   ↓                                                              │
│   list[Detection]                                               │
└────────────────────────────────┬────────────────────────────────┘
                                 │ Detection dataclass (frozen)
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│           DECISION LAYER (core/alert_engine.py)                 │
├─────────────────────────────────────────────────────────────────┤
│ should_alert(detection) → bool                                  │
│ • Cooldown por clase_name (car=5s, person=8s, etc.)            │
│ • Timestamp con time.monotonic()                                │
│                                                                  │
│ direction_message(detection) → str                              │
│ • Traduce class_name al español                                 │
│ • Agrega dirección (izq/frente/der) basada en center_x         │
│ • Output: "persona al frente" (ejemplo)                        │
└────────────────────────────────┬────────────────────────────────┘
                                 │ Alert messages
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│         ORCHESTRATION (core/orchestrator.py + modes/)            │
├─────────────────────────────────────────────────────────────────┤
│ Orchestrator (main.py lifespan context)                        │
│ ├─ ProactiveMode: YOLO loop @ 5 FPS                            │
│ │   ├─ ElevenLabsClient.generate(msg) → MP3 bytes             │
│ │   └─ AudioManager.speak(..., priority) → queue asyncio       │
│ ├─ ReactiveMode: Pregunta → Gemini → respuesta                │
│ │   ├─ GeminiClient.describe_scene(frame, question)           │
│ │   └─ Resume ProactiveMode después                            │
│ └─ Modo PAUSED: Espera trigger_button()                        │
│                                                                  │
│ Guarantees:                                                      │
│ • Mutex: Solo 1 modo activo simultáneamente                    │
│ • Fallback: Si APIs fallan, alert_engine funciona sin LLM      │
└────────────────────────────────┬────────────────────────────────┘
                                 │ Audio/LLM responses
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│              OUTPUT LAYER (core/audio_manager.py)               │
├─────────────────────────────────────────────────────────────────┤
│ AudioManager (asyncio.PriorityQueue)                            │
│ ├─ _AudioItem: (priority, text, audio_bytes)                   │
│ ├─ _drain_loop(): Reproduce en background thread               │
│ ├─ _play_bytes(): Escribe MP3 temporal + ffplay/aplay/mpg123   │
│ └─ Bluetooth audio (futuro)                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Módulos Principales

### 1. Input: StreamReader (`services/stream_reader.py`)

**Responsabilidad:** Conectar con stream MJPEG, extraer frames, manejar desconexiones.

| Componente | Código | Función |
|-----------|--------|---------|
| **Clase principal** | `StreamReader` | Async wrapper para httpx stream MJPEG |
| **Método start** | `async start()` | Arranca loop de captura en background |
| **Método get_frame** | `async get_frame() → np.ndarray \| None` | Retorna último frame decodificado |
| **Método frames** | `async frames() → AsyncIterator[np.ndarray]` | Itera nuevos frames |
| **Internal: _capture_loop** | Reconexión auto con backoff exponencial (1s→30s) |
| **Internal: _read_stream** | Descarga chunks MJPEG del stream HTTP |
| **Utilidades** | `_extract_frames(buf)` | Busca JPEG markers (0xFFD8...0xFFD9) |
| **Utilidades** | `_decode_jpeg(data)` | cv2.imdecode → np.ndarray |

**Características técnicas:**
```python
# Fallback automático para boundary markers:
_BOUNDARY_CANDIDATES = [b"--frame", b"--mjpegstream", b"--myboundary"]

# Thread-safe con asyncio.Lock
self._lock = asyncio.Lock()
self._latest_frame: np.ndarray | None = None

# Reconexión: max_retries=10, timeout=10.0
# Espera exponencial: min(2^retries, 30) segundos
```

**Endpoints HTTP que lo usan:**
- `/health` — verifica `stream_reader.get_frame()`
- `/preview` — captura frame actual
- `/detections` — obtiene detecciones del frame actual
- `/stream` — MJPEG vivo anotado

---

### 2. Inference: Detector (`core/detector.py`)

**Responsabilidad:** Envolver YOLO en múltiples backends (RKNN/ONNX/Mock), abstraer hardware.

#### Datos

```python
@dataclass(frozen=True)  # Immutable
class Detection:
    class_id: int                          # 0=person, 2=car, etc
    class_name: str                        # COCO label ("person", "car", ...)
    confidence: float                      # 0.0-1.0
    bbox: tuple[int, int, int, int]       # (x1, y1, x2, y2) en píxeles

    @property
    def is_priority(self) -> bool:
        """¿Alerta automática?"""
        return self.class_id in PRIORITY_CLASSES

    @property
    def center_x(self) -> int:
        """Centro X para calcular dirección"""
        return (self.bbox[0] + self.bbox[2]) // 2

    @property
    def direction(self) -> str:
        """Dirección relativa en frame de 640px"""
        # izquierda: x < 213, frente: 213-427, derecha: > 427
```

#### Backends

| Backend | Clase | Hardware | Latencia | Status |
|---------|-------|----------|----------|--------|
| Mock | `MockDetector` | N/A | <1ms | ✅ Desarrollo |
| ONNX CPU | `ONNXDetector` | ARM/x86 | ~205ms | ✅ Fallback validado |
| RKNN NPU | `RKNNDetector` | RK3588 | ~30ms | ✅ Producción |

#### Factory Pattern

```python
def build_detector(
    *,
    dev_mode: bool = True,
    model_path: str = "",
    confidence_threshold: float = 0.4,
) -> DetectorBackend:
    """Fallback chain: RKNN → ONNX → Mock"""
    if dev_mode:
        return MockDetector()
    if model_path.endswith(".rknn"):
        try:
            return RKNNDetector(model_path, confidence_threshold)
        except RuntimeError:
            # Fallback 1: ONNX
            onnx_path = model_path.replace(".rknn", ".onnx")
            if os.path.exists(onnx_path):
                logger.warning("RKNN failed, using ONNX")
                return ONNXDetector(onnx_path, confidence_threshold)
            # Fallback 2: Mock
            logger.warning("RKNN and ONNX failed, using Mock")
            return MockDetector()
    # ... similar para .onnx
```

#### Post-procesado YOLO

**Entrada:** Output raw de YOLO (shape: `[batch=1, 84, 8400]`)
- 84 = 4 coords (cx, cy, w, h) + 80 class scores
- 8400 = 3 escalas (80×80 stride-8, 40×40 stride-16, 20×20 stride-32)

**Función:** `_postprocess(output, orig_w, orig_h, threshold)`

```
1. Threshold: Filtrar confidence < 0.4
2. Scale: Escalar bboxes de 640×640 → original resolution
3. NMS: cv2.dnn.NMSBoxes() con IoU=0.45 para eliminar overlaps
4. Map: class_id → COCO label
5. Retorna: list[Detection]
```

**COCO Classes soportadas (PRIORITY_CLASSES):**
```python
{
    0: "person",      1: "bicycle",    2: "car",
    3: "motorcycle",  5: "bus",        7: "truck",
    9: "traffic_light", 11: "stop_sign",
    15: "cat",       16: "dog",
}
# + 70 clases no-priority (botella, silla, tabla, etc)
```

---

### 3. Decision: AlertEngine (`core/alert_engine.py`)

**Responsabilidad:** Throttle alertas por clase, generar mensajes en español.

#### Cooldowns por Clase

```python
ALERT_COOLDOWNS: dict[str, float] = {
    "car": 5.0,
    "person": 8.0,
    "traffic_light": 3.0,
    "dog": 10.0,
    "stairs": 3.0,
    "bicycle": 5.0,
    "bus": 5.0,
    "truck": 5.0,
    "motorcycle": 5.0,
    "stop_sign": 5.0,
}
DEFAULT_COOLDOWN = 5.0  # Clases no mapeadas
```

#### Métodos

| Método | Firma | Descripción |
|--------|-------|-------------|
| **should_alert** | `(detection: Detection) → bool` | ¿Ha expirado cooldown? |
| **mark_alerted** | `(detection: Detection) → None` | Registra timestamp |
| **direction_message** | `(detection: Detection) → str` | Mensaje en español |

#### Ejemplos

```python
engine = AlertEngine()

# Detección 1: person @ center
det1 = Detection(0, "person", 0.95, (200, 100, 280, 300))
if engine.should_alert(det1):  # → True (primera vez)
    msg = engine.direction_message(det1)  # → "persona al frente"
    engine.mark_alerted(det1)

# Detección 2: mismo person @ otro lugar (mismo timestamp)
det2 = Detection(0, "person", 0.92, (400, 100, 480, 300))
if engine.should_alert(det2):  # → False (cooldown 8s activo)
    # ... no se alerta

# Esperar 8s+ y reintentar
await asyncio.sleep(8.1)
if engine.should_alert(det2):  # → True (cooldown expirado)
    engine.mark_alerted(det2)
```

#### Invariantes

- ✅ Immutable state: `mark_alerted` NO muta `self._last_alerted`, crea nuevo dict
- ✅ `time.monotonic()` (no afectado por ajustes de reloj)
- ✅ Cooldown es por `class_name`, no por bbox/posición

---

### 4. Modos de Operación

#### ProactiveMode (`modes/proactive.py`)

**Flujo:** Loop → StreamReader → Detector → AlertEngine → TTS → AudioManager

```python
class ProactiveMode:
    async def _loop(self) -> None:
        """Main loop @ fps_limit (default 5 FPS)"""
        frame_interval = 1.0 / self.fps_limit
        while self._running:
            frame = await self._stream.get_frame()
            if frame is not None:
                await self._process_frame(frame)
            await asyncio.sleep(frame_interval)

    async def _process_frame(self, frame: np.ndarray) -> None:
        detections = self._detector.detect(frame)
        for det in detections:
            if not self._alert_engine.should_alert(det):
                continue
            self._alert_engine.mark_alerted(det)
            message = _build_message(frame, det, self._alert_engine)
            priority = _detection_priority(det)
            audio_bytes = await self._tts.generate(message)
            await self._audio.speak(message, audio_bytes, priority=priority)
```

**Prioridades de Audio:**
```python
PRIORITY_ALERT = 3       # traffic_light, stairs, stop_sign
PRIORITY_NORMAL = 5      # person, car, bicycle
PRIORITY_LOW = 8         # otros
```

#### ReactiveMode (`modes/reactive.py`)

**Flujo:** Pregunta → Captura frame → Gemini → TTS → Audio

```python
class ReactiveMode:
    async def handle_question(self, question: str) -> str:
        """Pausa proactivo, responde, reanuda proactivo"""
        frame = await self._stream.get_frame()
        if frame is None:
            return await self._speak_and_return("No tengo imagen")

        answer = await self._gemini.describe_scene(frame, question)
        audio_bytes = await self._tts.generate(answer)
        await self._audio.speak(answer, audio_bytes, priority=1)  # Máxima prioridad
        return answer
```

#### Orchestrator (`core/orchestrator.py`)

**Responsabilidad:** Coordinar ProactiveMode ↔ ReactiveMode, garantizar que no se solapen.

```python
class Orchestrator:
    def __init__(self, stream, detector, alert_engine, tts, audio, gemini):
        self._proactive = ProactiveMode(stream, detector, alert_engine, tts, audio)
        self._reactive = ReactiveMode(stream, gemini, tts, audio)
        self._mode = Mode.PAUSED
        self._lock = asyncio.Lock()  # Mutex

    async def handle_question(self, question: str) -> str:
        """Thread-safe: pausa proactivo durante pregunta"""
        async with self._lock:
            was_proactive = self._mode == Mode.PROACTIVE
            if was_proactive:
                await self._stop_proactive()
                self._mode = Mode.REACTIVE

        answer = await self._reactive.handle_question(question)

        async with self._lock:
            if was_proactive:
                await self._start_proactive()
                self._mode = Mode.PROACTIVE
        return answer
```

---

### 5. Output: AudioManager (`core/audio_manager.py`)

**Responsabilidad:** Cola de audio con prioridad, reproducción asincrónica.

#### Estructura

```python
@dataclass(order=True)
class _AudioItem:
    priority: int           # Menor = más urgente
    text: str = field(compare=False)
    audio: bytes = field(compare=False)

class AudioManager:
    def __init__(self):
        self._queue: asyncio.PriorityQueue[_AudioItem] = asyncio.PriorityQueue()
        self._current_task: asyncio.Task | None = None
        self._drain_task: asyncio.Task | None = None
```

#### Métodos

| Método | Firma | Descripción |
|--------|-------|-------------|
| **speak** | `async speak(text, audio, priority=5)` | Encola audio para reproducción |
| **stop_current** | `async stop_current()` | Cancela reproducción actual |
| **start** | `async start()` | Inicia loop de drenado |
| **stop** | `async stop()` | Para loop y cancela reproducción |

#### Flujo de Reproducción

```python
async def _drain_loop(self) -> None:
    """Background loop: saca items de cola y reproduce"""
    while True:
        item = await self._queue.get()  # Espera item
        self._current_task = asyncio.create_task(self._play_audio(item))
        try:
            await self._current_task
        except asyncio.CancelledError:
            logger.debug("Reproducción cancelada")
        finally:
            self._queue.task_done()

async def _play_bytes(self, audio: bytes, label: str) -> None:
    """Escribe MP3 temporal + reproduce con ffplay/aplay/mpg123"""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio)
        tmp_path = tmp.name

    try:
        player = _find_player()  # ffplay > aplay > mpg123
        if player is None:
            logger.warning("No reproductor encontrado")
            return

        cmd = _build_play_command(player, tmp_path)
        proc = await asyncio.create_subprocess_exec(*cmd, ...)
        await proc.wait()
    finally:
        os.unlink(tmp_path)
```

---

### 6. Servicios Externos

#### ElevenLabsClient (`services/elevenlabs_client.py`)

**Responsabilidad:** Sintetizar texto a audio MP3, cache LRU.

```python
class ElevenLabsClient:
    async def generate(self, text: str) -> bytes:
        """Retorna MP3 bytes. En dev_mode, retorna bytes vacíos."""
        if self._dev_mode:
            return b""

        cached = self._cache_get(text)
        if cached:
            return cached

        audio = await self._fetch_audio(text)
        self._cache_put(text, audio)
        return audio

    async def _fetch_audio(self, text: str) -> bytes:
        """POST a https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"""
        # Retorna MP3 bytes si status_code == 200
        # Lanza RuntimeError si falla
```

**Cache LRU:**
- Max 50 entradas
- Evicción: entrada más antigua cuando se supera límite
- Hit: mueve entrada al final (MRU)

#### GeminiClient (`services/gemini_client.py`)

**Responsabilidad:** Describir escena con visión, rate limiting.

```python
class GeminiClient:
    async def describe_scene(self, frame: np.ndarray, question: str = "¿Qué hay?") -> str:
        """Envía frame + pregunta a Gemini, retorna descripción en español"""
        if self._dev_mode:
            return "Modo desarrollo. Frente a ti hay una simulación..."

        await self._rate_limit()  # Respeta rpm_limit (default 10)
        return await self._call_api(frame, question)

    async def _call_api(self, frame: np.ndarray, question: str) -> str:
        """POST a generativelanguage.googleapis.com, codifica frame como JPEG base64"""
        # Fallback a _FALLBACK si error
```

**Rate limiting:**
- `rpm_limit=10` (máx 10 requests/minuto)
- Deque de timestamps (últimos 60s)
- Sleep si se alcanza límite

---

### 7. FastAPI App (`main.py`)

**Responsabilidad:** Servir endpoints HTTP, lifespan management.

#### Lifespan

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await stream_reader.start()
    await audio.start()
    await orchestrator.start()
    logger.info("Lazarillo iniciado — stream: %s", settings.pi_zero_stream_url)
    yield
    # Shutdown
    await orchestrator.stop()
    await audio.stop()
    await stream_reader.stop()
```

#### Endpoints

| Método | Path | Descripción | Retorna |
|--------|------|-------------|---------|
| `GET` | `/health` | Status del sistema | `{"status": "ok", "stream": "connected", "mode": "proactive", "env": "dev"}` |
| `GET` | `/detections` | Detecciones JSON | `{"count": 2, "detections": [...]}` |
| `GET` | `/preview` | Frame actual con bboxes | JPEG binario |
| `GET` | `/stream` | MJPEG vivo anotado | Multipart/MJPEG |
| `POST` | `/trigger/button` | Alterna proactivo/pausado | `{"mode": "paused"}` |
| `POST` | `/mode/{mode}` | Cambia modo | `{"mode": "proactive"}` |
| `POST` | `/ask` | Pregunta sobre escena | `{"answer": "..."}` |

#### Annotación de Detecciones

```python
def _draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        color = (0, 255, 0) if d.is_priority else (200, 200, 200)
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.0%} [{d.direction}]"
        cv2.putText(out, label, (x1, y1 - 6), ...)
    return out
```

---

## Pipeline de Datos

### Latencia (Fase 2 Proactivo)

```
Frame capture (MJPEG)     : 33ms (10-15 FPS stream)
YOLO inference (RKNN)     : 30ms @ 640×640
Post-procesado (NMS)      : 2ms
AlertEngine.should_alert(): <1ms
TTS generate (cached)     : 0ms (LRU cache hit)
TTS generate (fresh)      : 500-1000ms (API)
AudioManager queue        : <1ms
Audio playback            : 100-500ms (Bluetooth/speaker)
─────────────────────────────────────────────
TOTAL (cached):           ~165ms ✅
TOTAL (fresh):            ~600ms ✅ (aceptable)
```

### Fallbacks en Cascada

```
Stream desconectado?
├─ Sí: StreamReader._capture_loop reintenta (backoff exponencial)
└─ No: continúa

RKNN no disponible?
├─ Sí: intenta ONNXDetector
├─ Fallback también falla: usa MockDetector
└─ Siempre funciona (dev mode)

Gemini API falla?
├─ Sí: Log error + fallback local (alert_engine)
└─ No: responde con LLM

TTS API falla?
├─ Sí: Log error + sigue (no reproduce audio)
└─ No: síntesis + cola de audio

ElevenLabs offline?
├─ Sí: Cache hit (si existe) o bytes vacíos
└─ No: síntesis fresca
```

---

## Configuración (config.py)

```python
class Settings(BaseSettings):
    # APIs externas
    gemini_api_key: str = "dev"
    elevenlabs_api_key: str = "dev"
    elevenlabs_voice_id: str = "dev"
    mongodb_uri: str = "mongodb://localhost:27017"

    # Hardware
    pi_zero_stream_url: str = "http://stream-mock:8080/stream.mjpg"

    # YOLO
    yolo_model_path: str = "models/yolo11n.rknn"
    yolo_confidence_threshold: float = 0.4

    # Alertas
    alert_cooldown_seconds: int = 5  # default fallback

    # Audio / STT
    whisper_model_size: str = "tiny"
    audio_device: str = "default"

    # Entorno
    env: str = "development"

    @property
    def is_dev(self) -> bool:
        return self.env == "development"
```

---

## Dependencias Externas

### Librerías Python

| Paquete | Versión | Propósito |
|---------|---------|----------|
| `fastapi` | 0.115.0 | Framework HTTP |
| `uvicorn` | 0.30.6 | Servidor ASGI |
| `httpx` | 0.27.2 | Cliente HTTP async |
| `opencv-python-headless` | 4.10.0.84 | Codificación JPEG, post-procesado |
| `numpy` | ≥2.0 | Álgebra lineal |
| `onnxruntime` | ≥1.18.0 | Inferencia YOLO (CPU fallback) |
| `rknnlite` | Included en Orange Pi | Inferencia YOLO (NPU) |
| `elevenlabs` | 1.9.0 | TTS |
| `google-generativeai` | 0.8.3 | Gemini API |
| `faster-whisper` | 1.0.3 | STT (futuro) |
| `sounddevice` | Latest | Captura de micrófono (futuro) |
| `pydantic-settings` | 2.5.2 | Configuración |

### Hardware/Sistema

| Componente | Especificación |
|-----------|----------------|
| **Procesador** | ARM Cortex-A76/A55 (8 cores @ 2.4GHz) |
| **NPU** | Rockchip RK3588 (3× NEON + 1× NEON+) @ 800MHz |
| **RAM** | 16GB LPDDR5 |
| **Storage** | 32GB eMMC + SDCard |
| **Audio** | Bluetooth 5.3 + 3.5mm jack |
| **Sistema** | Linux (Debian/Ubuntu ARM) |

---

## Testing

### Cobertura (178 tests)

```
core/detector.py          26 tests ✅
core/alert_engine.py      18 tests ✅
core/audio_manager.py     14 tests ✅
core/traffic_light.py     7 tests ✅
core/orchestrator.py      14 tests ✅
core/voice_capture.py     6 tests ✅
core/stt.py              6 tests ✅
modes/proactive.py        16 tests ✅
modes/reactive.py         13 tests ✅
services/stream_reader.py 17 tests ✅
services/elevenlabs_client.py 15 tests ✅
services/gemini_client.py 18 tests ✅
─────────────────────────────────
TOTAL:                   178 tests ✅
```

### Ejecutar Tests

```bash
# Todos
pytest

# Por módulo
pytest tests/unit/test_detector.py -v

# Con cobertura
pytest --cov=core --cov=services --cov-report=html

# En Orange Pi (hardware real)
ssh orangepi@192.168.1.X "cd ~/lazarillo && pytest tests/"
```

---

## Próximos Pasos

### Fase 2 (Crítica para MVP)
- ✅ `core/audio_manager.py` — Cola de audio
- ✅ `services/elevenlabs_client.py` — TTS
- ✅ `modes/proactive.py` — Loop principal
- ✅ `core/traffic_light.py` — Clasificador semáforos

### Fase 3 (Alta prioridad)
- `core/voice_capture.py` — VAD + captura
- `core/stt.py` — Transcripción Whisper
- `modes/reactive.py` — Pregunta-respuesta
- `core/orchestrator.py` — Coordinación de modos

### Fase 4-6
- Memory mode + MongoDB
- Dashboard web
- Bluetooth pairing
- Deployment systemd

---

## Referencias

- **YOLO:** https://github.com/ultralytics/yolov8
- **RKNN Toolkit:** https://github.com/airockchip/rknn-toolkit2
- **rknn-model-zoo:** https://github.com/airockchip/rknn_model_zoo
- **Orange Pi 5 Max:** http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details_142.html
- **Gemini API:** https://ai.google.dev/
- **ElevenLabs:** https://elevenlabs.io/
