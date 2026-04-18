# Lazarillo — Arquitectura del Sistema

**Last Updated:** 2026-04-18

## Visión General

Lazarillo es un sistema distribuido que combina visión artificial en tiempo real con asistencia por IA para ciegos. La arquitectura está optimizada para baja latencia, tolerancia a fallos y operación offline (fallback a alertas locales sin Gemini).

## Componentes Principales

```
┌──────────────────────────────────────────────────────────────────┐
│ ENTRADA: Cámara (Raspberry Pi Zero W)                            │
└─────────────────────┬──────────────────────────────────────────┘
                      │ MJPEG stream (HTTP)
                      ↓
┌──────────────────────────────────────────────────────────────────┐
│ PROCESAMIENTO CENTRAL: Orange Pi 5 Max (RK3588 NPU)              │
│                                                                  │
│  ┌─────────────────────────┐      ┌──────────────────────┐     │
│  │ StreamReader            │      │ Audio Manager        │     │
│  │ • httpx async client    │      │ • Cola de prioridad  │     │
│  │ • Buffer MJPEG          │      │ • Control interrup.  │     │
│  │ • Reconexión auto       │      │ • Sync/async mixing  │     │
│  └──────────┬──────────────┘      └──────────┬───────────┘     │
│             │ 640×480 np.ndarray             │ Audio segments   │
│             ↓                                 ↓                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Pipeline de Detección (detector.py)                    │    │
│  │                                                        │    │
│  │  [Frame] → Redimensionar (640×640)                    │    │
│  │         → Normalizar RGB                              │    │
│  │         ↓                                              │    │
│  │  ┌──────────────────────────────────┐                 │    │
│  │  │ Backend Detector (elegir 1)       │                 │    │
│  │  │ • RKNNDetector → 30 FPS          │                 │    │
│  │  │ • ONNXDetector → 5 FPS (CPU)     │                 │    │
│  │  │ • MockDetector → ∞ FPS (dev)     │                 │    │
│  │  └────────┬─────────────────────────┘                 │    │
│  │           │ Raw output (84, 8400)                      │    │
│  │           ↓                                            │    │
│  │  Post-procesado: _postprocess()                        │    │
│  │  • Filtrar por threshold (0.4)                         │    │
│  │  • Escalar bboxes a resolución original                │    │
│  │  • NMS (IoU=0.45) para eliminar duplicados             │    │
│  │           ↓                                            │    │
│  │  list[Detection]                                       │    │
│  │  • class_id, class_name, confidence                    │    │
│  │  • bbox (x1, y1, x2, y2), direction (izq/frente/der)  │    │
│  │           ↓                                            │    │
│  └────────────┬──────────────────────────────────────────┘    │
│               │                                                 │
│  ┌────────────▼──────────────────────────────────────────┐    │
│  │ AlertEngine (alert_engine.py)                          │    │
│  │                                                        │    │
│  │  should_alert(detection) → bool                        │    │
│  │  • Cooldown por class: car=5s, person=8s, etc.        │    │
│  │  • Retorna True si cooldown expiró                    │    │
│  │                                                        │    │
│  │  direction_message(detection) → str                    │    │
│  │  • Traduce: "person" → "persona"                      │    │
│  │  • Agrega dirección: "persona al frente"              │    │
│  │           ↓                                            │    │
│  └────────────┬──────────────────────────────────────────┘    │
│               │ Alert message                                   │
│               ↓                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Modos de Operación                                     │    │
│  │                                                        │    │
│  │ • Proactive (TODO): YOLO → AlertEngine → TTS          │    │
│  │ • Reactive (TODO): Voice → Whisper → Gemini → TTS    │    │
│  │ • Memory (TODO): Frame buffer → MongoDB → Report      │    │
│  │           ↓                                            │    │
│  └────────────┬──────────────────────────────────────────┘    │
│               │                                                 │
└───────────────┼────────────────────────────────────────────────┘
                │
         ┌──────┴──────┬─────────────────┬───────────────┐
         │             │                 │               │
         ↓             ↓                 ↓               ↓
    ┌─────────┐  ┌──────────┐     ┌──────────┐    ┌──────────┐
    │ TTS     │  │ Gemini   │     │ MongoDB  │    │ Dashboard
    │ Eleven  │  │ LLM      │     │ Sessions │    │ Web
    │ Labs    │  │ (ctx)    │     │ Reports  │    │
    └─────────┘  └──────────┘     └──────────┘    └──────────┘
         │             │                 │               │
         └─────────────┴─────────────────┴───────────────┘
                        │
                   SALIDA: Audio + Datos
```

## Componentes Detallados

### 1. StreamReader (services/stream_reader.py)

**Responsabilidad:** Conectar con el stream MJPEG, extraer frames, manejar desconexiones.

**Arquitectura:**
```
HTTP stream (MJPEG) ──→ httpx.AsyncClient
                        ↓
                   Buffer circular (bytes)
                        ↓
                   JPEG markers (0xFFD8...0xFFD9)
                        ↓
                   cv2.imdecode() → np.ndarray
                        ↓
                   _latest_frame (protegido con lock)
```

**API:**
```python
reader = StreamReader("http://host:8080/stream.mjpg")
await reader.start()

frame = await reader.get_frame()  # np.ndarray | None
async for frame in reader.frames():  # Itera nuevos frames
    ...
```

**Características:**
- Reconexión automática con backoff exponencial (1s → 30s)
- Máx 10 intentos antes de fallar
- Thread-safe con asyncio.Lock
- Detecta múltiples boundary markers (--frame, --mjpegstream, --myboundary)

**Estado en Fase 1:** ✅ Completo con tests

---

### 2. Detector (core/detector.py)

**Responsabilidad:** Ejecutar YOLO y post-procesado, abstraer backends de hardware.

**Backends:**

| Backend | Hardware | Latencia | Status |
|---------|----------|----------|--------|
| MockDetector | N/A | <1ms | ✅ Producción |
| ONNXDetector | CPU (ARM/x86) | ~205ms | ✅ Fallback validado |
| RKNNDetector | NPU RK3588 | ~30ms | 🔄 Validando modelo |

**Factory Pattern:**
```python
detector = build_detector(
    dev_mode=True,                    # → MockDetector
    model_path="model.rknn",          # → RKNNDetector
    confidence_threshold=0.4
)

# Si RKNNDetector falla:
# 1. Intenta cargar model.onnx (fallback)
# 2. Si falla, usa MockDetector
```

**Postprocessing (YOLOv8n output format):**
```
YOLO output: (batch, 84, 8400)
            └─ 84 = 4 coords (cx,cy,w,h) + 80 class scores

1. Threshold: confidence >= 0.4
2. Scale: coords 640×640 → original resolution
3. NMS (Non-Maximum Suppression): eliminar overlaps (IoU=0.45)
4. Map: class_id → COCO label
```

**Detection dataclass:**
```python
@dataclass(frozen=True)  # Immutable
class Detection:
    class_id: int                      # 0=person, 2=car, etc
    class_name: str                    # COCO label
    confidence: float                  # 0.0-1.0
    bbox: tuple[int, int, int, int]   # x1,y1,x2,y2

    @property
    def is_priority(self) -> bool:
        """¿Es objeto prioritario para alertar?"""
        return class_id in PRIORITY_CLASSES

    @property
    def direction(self) -> str:
        """Dirección (izquierda/frente/derecha) basada en center_x."""
```

**COCO Classes soportadas:**

Prioritarias (alertar):
- person (0), bicycle (1), car (2), motorcycle (3), bus (5), truck (7)
- traffic_light (9), stop_sign (11), cat (15), dog (16)

Otras 70 clases mapeadas del COCO estándar.

**Estado en Fase 1:** ✅ Completo, ONNXDetector validado en Orange Pi

---

### 3. AlertEngine (core/alert_engine.py)

**Responsabilidad:** Throttle alertas por clase, generar mensajes en español.

**Cooldowns por clase:**
```python
ALERT_COOLDOWNS = {
    "car": 5.0,          # Auto cada 5 segundos máximo
    "person": 8.0,       # Persona cada 8 segundos
    "traffic_light": 3.0,# Semáforo cada 3 segundos
    "dog": 10.0,         # Perro cada 10 segundos
    ...
}
DEFAULT_COOLDOWN = 5.0   # Clases no mapeadas
```

**API:**
```python
engine = AlertEngine()

if engine.should_alert(detection):
    msg = engine.direction_message(detection)
    # "persona al frente" → TTS
    engine.mark_alerted(detection)

# Cooldown es por class_name, no por bbox/posición
# Dos personas en distinto lugar comparten cooldown
```

**Invariantes:**
- Timestamps con `time.monotonic()` (no afectado por ajustes de reloj)
- Immutable state: `mark_alerted` retorna nuevo dict, no muta
- Español: DIRECTION_LABELS y CLASS_LABELS_ES

**Estado en Fase 1:** ✅ Completo, 100% tests

---

### 4. FastAPI App (main.py)

**Responsabilidad:** Servir endpoints HTTP para monitoreo y debugging.

**Endpoints:**

| Método | Path | Descripción | Retorna |
|--------|------|-------------|---------|
| GET | `/health` | Status del sistema | `{"status": "ok", "stream": "connected", "env": "dev"}` |
| GET | `/detections` | Detecciones JSON | `{"count": 2, "detections": [...]}` |
| GET | `/preview` | Frame actual con bboxes | JPEG binario |
| GET | `/stream` | MJPEG vivo anotado | Multipart/MJPEG |

**Lifespan:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await stream_reader.start()
    yield
    # Shutdown
    await stream_reader.stop()
```

**Estado en Fase 1:** ✅ Completo

---

## Data Flow por Modo (Futuro)

### Modo Proactivo (Fase 2)

```
┌─────────────────────────────────────────────────┐
│ Loop: 30 FPS @ 640×640 (RKNN)                   │
├─────────────────────────────────────────────────┤
│ 1. StreamReader.get_frame()                     │
│ 2. detector.detect(frame) → list[Detection]     │
│ 3. Para cada detection (filtrar priority):      │
│    a. if engine.should_alert(det):              │
│       • msg = engine.direction_message(det)     │
│       • await tts.speak(msg)                    │
│       • engine.mark_alerted(det)                │
│    b. else: skip (cooldown activo)              │
│ 4. Log/dashboard update                         │
│ 5. sleep(1/30)                                  │
└─────────────────────────────────────────────────┘
```

**Latencia esperada:**
- Frame acquisition: 33ms
- YOLO inference: 30ms (RKNN)
- NMS post-procesado: 2ms
- TTS synthesize (cached): 100ms
- Audio playback: 200-500ms
- **Total: ~400ms** (aceptable)

---

### Modo Reactivo (Fase 3)

```
┌──────────────────────────────────────────────────┐
│ Pauso alertas automáticas mientras usuario habla │
├──────────────────────────────────────────────────┤
│ 1. Escuchar audio con VAD (webrtcvad)           │
│ 2. Transcribir con Whisper (faster-whisper)    │
│ 3. Pasar contexto + transcripción a Gemini     │
│ 4. Sintetizar respuesta con ElevenLabs TTS     │
│ 5. Reproducir audio Bluetooth                   │
│ 6. Reanudar modo proactivo                      │
│                                                  │
│ Timeout: 30 segundos (fallback a proactivo)    │
└──────────────────────────────────────────────────┘
```

**Gemini context:** Últimos 10 frames como descripción YOLO + detections

---

### Modo Memoria (Fase 4)

```
┌──────────────────────────────────────────────────┐
│ Grabar sesión + generar reporte                 │
├──────────────────────────────────────────────────┤
│ • Capturar frame cada 100ms (10 FPS)            │
│ • Guardar detections JSON en MongoDB            │
│ • Keyframes en JPEG (cada 30s o cambio grande)  │
│ • Al terminar:                                  │
│   - Gemini Pro analiza keyframes + detections  │
│   - Genera reporte Markdown                     │
│   - Guardar en MongoDB con full-text index      │
│   - Usuario puede consultar reportes            │
│                                                  │
│ Límites: 100 keyframes/sesión, 2h máximo       │
└──────────────────────────────────────────────────┘
```

---

## Hardware Setup

### Raspberry Pi Zero W
```
┌─────────────────────────────────────┐
│ Raspberry Pi Zero W (ARMv6)         │
├─────────────────────────────────────┤
│ • CPU: ARM1176 @1GHz (1 core)       │
│ • RAM: 512MB                        │
│ • WiFi: 802.11n (2.4GHz)            │
│ • Cámara: PiCamera v2 o v3          │
│ • Streaming: picamera2 → MJPEG      │
│ • Resolución: 640×480 @ 10-15 FPS   │
│ • Consumo: <2W                      │
└─────────────────────────────────────┘
```

**Script streaming (Fase 5):**
```python
# pi_zero/stream.py
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder

camera = Picamera2()
camera.configure(camera.create_video_configuration(main={"size": (640, 480)}))
camera.start_recording(MJPEGEncoder(), FileOutput("stream.mjpg"))
```

---

### Orange Pi 5 Max

```
┌──────────────────────────────────────────┐
│ Orange Pi 5 Max (RK3588)                 │
├──────────────────────────────────────────┤
│ • CPU: ARM Cortex-A76/A55 8-core        │
│ • RAM: 16GB LPDDR5                       │
│ • NPU: Rockchip RK3588 Cores:            │
│   - 3× NEON (2.4 GOPS INT8)              │
│   - 1× NEON+           (2.4 GOPS INT8)   │
│   - Utilización: 800MHz (default)        │
│ • GPU: Mali-G610 MP4                     │
│ • Storage: eMMC 32GB + SDCard            │
│ • WiFi: 802.11 a/b/g/n/ac               │
│ • Bluetooth: 5.3                         │
│ • Audio: 3.5mm jack + USB                │
│ • Consumo: 5-12W (idle-peak)             │
├──────────────────────────────────────────┤
│ RKNPU drivers:                           │
│ • RKNPU kernel module: 0.9.8             │
│ • RKNPU runtime: 2.3.0                   │
│ • NPU device: /dev/dri/renderD129        │
│ • Inference: rknnlite API                │
└──────────────────────────────────────────┘
```

**Benchmark YOLO (rknn-toolkit2):**
| Modelo | Formato | Latencia | FPS | Precisión |
|--------|---------|----------|-----|-----------|
| yolov8n | RKNN INT8 | ~30ms | 33 | -0.5-1% |
| yolov8n | ONNX FP32 | ~205ms | 5 | Baseline |
| yolov8n | MockDetector | <1ms | ∞ | Mocked |

---

### Audio Bluetooth

**Hardware:**
- Auriculares Bluetooth 5.0+ (latency <50ms recomendado)
- Built-in Bluetooth en Orange Pi

**Stack (Linux ARM):**
- BlueZ 5.x (kernel, dmesg)
- PipeWire (audio daemon, reemplaza PulseAudio)
- alsa-lib (driver audio)

**Flujo (Fase 5 setup):**
```bash
# 1. Parear auricular
bluetoothctl
> scan on
> pair <MAC>
> connect <MAC>

# 2. Verificar dispositivo
pactl list sinks

# 3. Python TTS → Bluetooth
import pydub
audio = pydub.AudioSegment.from_file("alert.mp3")
# (reproducir en sink Bluetooth)
```

---

## Optimización de Latencia

### Target: <2s desde detección a audio

| Componente | Latencia | Margen |
|-----------|----------|--------|
| Frame capture + MJPEG | 33ms | buffer 1-3 frames |
| YOLO inference (RKNN) | 30ms | parallelizable |
| Post-procesado NMS | 2ms | tight |
| AlertEngine.should_alert() | <1ms | negligible |
| ElevenLabs TTS (cached) | 0ms | local cache |
| ElevenLabs TTS (fresh) | 500-1000ms | red flag |
| Audio playback | 100-500ms | Bluetooth latency |
| **Total** | **~600ms** | ✅ Aceptable |

**Optimizaciones implementadas:**
1. Cache TTS de frases comunes (Fase 6.1)
2. Detector elegido por backend (RKNN vs ONNX)
3. AlertEngine sin locks (timestamps atómicos)
4. JPEG quality=70 en streaming (reducir ancho de banda)

---

## Tolerancia a Fallos

### Fallback Chain

```python
# detector.py build_detector()
try:
    return RKNNDetector(model_path)
except RuntimeError:
    # Fallback 1: Intentar ONNX
    if model_path.endswith(".rknn"):
        onnx_path = model_path.replace(".rknn", ".onnx")
        if os.path.exists(onnx_path):
            return ONNXDetector(onnx_path)
    
    # Fallback 2: Mock (siempre funciona)
    logger.warning("Usando MockDetector como fallback")
    return MockDetector()
```

### Reconexión Stream

```python
# services/stream_reader.py _capture_loop()
retries = 0
while running and retries < 10:
    try:
        await _read_stream()
        retries = 0  # reset en éxito
    except Exception:
        wait = min(2 ** retries, 30)  # backoff: 1s → 30s
        retries += 1
        await asyncio.sleep(wait)
```

### API Externas (Gemini, ElevenLabs)

- **Sin conexión:** AlertEngine genera mensajes locales sin Gemini
- **Rate limiting:** Queue con exponential backoff
- **Timeout:** 30s máximo, fallback a proactivo
- **Cache:** TTS cache LRU (20-30 frases comunes)

---

## Seguridad

### No Implementado en Fase 1-2

- [ ] Autenticación API endpoints (agregar simple bearer token)
- [ ] Rate limiting por IP
- [ ] Encrypt API keys en .env
- [ ] CORS restricción a localhost

### Plan Fase 5-6
- Usar systemd user service (sin root)
- SSH key-only access para Orange Pi
- Firewall: solo puertos necesarios (8000, 22)
- Audit logging de sesiones Memory mode

---

## Escalabilidad Futura

| Aspecto | Limitación Actual | Mejora Futuro |
|--------|------------------|---|
| Cámaras | 1 Pi Zero | Multi-cam 360°, cambiar dinámicamente |
| Modelos | 1 YOLO | Ensemble (YOLO + Pose + Tracking) |
| Bases de datos | SQLite local | MongoDB sharded + replicación |
| LLM context | Últimos 10 frames | Temporal graph (30+ frames) |
| Deployment | Single Orange Pi | Kubernetes ARM cluster |
| Reporte | Markdown static | Análisis temporal, trends |

---

## Diagrama de Dependencias

```
core/
├── detector.py ─→ core/__ (Detection)
└── alert_engine.py ─→ core/detector.py

services/
├── stream_reader.py ─→ (httpx, cv2, numpy)
├── elevenlabs_client.py (TODO)
└── gemini_client.py (TODO)

modes/
├── proactive.py (TODO) ─→ {stream_reader, detector, alert_engine, elevenlabs}
├── reactive.py (TODO) ─→ {stt, gemini, elevenlabs}
└── memory.py (TODO) ─→ {mongodb, detector, gemini}

main.py ─→ {stream_reader, detector, FastAPI}
```

---

## Testing Strategy

### Unit Tests (Fase 1-2: 32 tests)

```python
# core/detector.py
test_detection_properties()        # center_x, direction, is_priority
test_mock_detector()               # default + custom detections
test_postprocess_nms()             # threshold, scaling, NMS filtering

# core/alert_engine.py
test_cooldown_per_class()          # independent throttling
test_direction_message()           # Spanish translations
test_mark_alerted()                # timestamp updates

# services/stream_reader.py
test_extract_frames()              # JPEG boundary parsing
test_decode_jpeg()                 # cv2.imdecode error handling
test_frame_async_iteration()       # frames() generator
```

### Integration Tests (Fase 3+)

```python
# modes/reactive.py
async def test_voice_to_response():
    # Record WAV → Whisper → Gemini → TTS
    # Verify latency <3s

# modes/memory.py
async def test_session_recording():
    # 2 min recording → Gemini analysis → MongoDB
    # Verify Markdown output coherence
```

### Hardware Tests (Fase 5)

```bash
# Orange Pi
ssh orangepi@192.168.1.100
python tests/npu_validation.py models/rknn/yolov8n.rknn

# Salida esperada:
# ✅ NPU device: /dev/dri/renderD129
# ✅ Latencia media: 30ms → 33 FPS
# ✅ Output shape (84, 8400) compatible
```

---

## Próximas Fases

### Fase 2: Modo Proactivo (CRÍTICA para MVP)
- [ ] traffic_light.py: HSV classifier (rojo/verde)
- [ ] elevenlabs_client.py: TTS async + cache LRU
- [ ] audio_manager.py: Queue + interrupción
- [ ] modes/proactive.py: Loop principal
- [ ] Tests: 16 tests nuevos, 100% cobertura Fase 2

### Fase 3: Modo Reactivo (ALTA)
- [ ] voice_capture.py: VAD con webrtcvad
- [ ] stt.py: faster-whisper tiny español
- [ ] modes/reactive.py: Pipeline voz-respuesta
- [ ] orchestrator.py: Pauso proactivo durante pregunta

### Fases 4-6
Consultar docs/plan.md para timeline detallado.

---

## Referencias

- **YOLO:** https://github.com/ultralytics/yolov8
- **RKNN Toolkit:** https://github.com/airockchip/rknn-toolkit2
- **rknn-model-zoo:** https://github.com/airockchip/rknn_model_zoo
- **faster-whisper:** https://github.com/guillaumekln/faster-whisper
- **Gemini API:** https://ai.google.dev/
- **ElevenLabs:** https://elevenlabs.io/docs
