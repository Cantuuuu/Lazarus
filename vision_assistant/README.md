# vision_assistant — Cliente Python de Lazarus

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Plataforma-Windows%20%7C%20Orange%20Pi%205-orange)

Cliente en Python del sistema [Lazarus](../README.md). Captura video y audio del usuario, decide cuándo hay un momento relevante (de forma manual, automática por YOLO o por voz) y empaqueta el contexto para enviarlo al agente `lazarus_assistant` en n8n.

Para la descripción general del sistema completo (infraestructura, servicios externos, workflows de n8n), consulta el [README raíz](../README.md).

---

## Arquitectura

El sistema sigue una arquitectura de múltiples hilos con una cola de eventos centralizada:

```
┌──────────────────────────────────────────────────────────────┐
│                         main.py                              │
│                                                              │
│  CameraService (ring buffer de 10 segundos)                  │
│       │                                                      │
│       │     Triggers (hilos daemon independientes)           │
│       │                                                      │
│       │  ┌─────────────────┐                                 │
│       ├─►│  ManualTrigger  │ ◄── hotkey Ctrl+Shift+G         │
│       │  └────────┬────────┘                                 │
│       │           │ Event(MANUAL)                            │
│       │           ▼                                          │
│       │  ┌─────────────────┐                                 │
│       ├─►│   YoloTrigger   │ ◄── análisis a 10 fps           │
│       │  └────────┬────────┘                                 │
│       │           │ Event(YOLO)                              │
│       │           ▼                                          │
│       │  ┌─────────────────┐                                 │
│       └─►│  VoiceTrigger   │ ◄── "Lazarus" + comando         │
│          └────────┬────────┘                                 │
│                   │ Event(VOICE)                             │
│                   ▼                                          │
│            queue.Queue (maxsize=100)                         │
│                   │                                          │
│                   ▼                                          │
│          ┌──────────────────────┐                            │
│          │   EventDispatcher    │  (anti-spam, dedupe)       │
│          └──────────┬───────────┘                            │
│                     │                                        │
│           ┌─────────▼──────────┐                            │
│           │   n8n Webhook      │                             │
│           │  (Gemini Vision)   │                             │
│           └─────────┬──────────┘                            │
│                     │                                        │
│           ┌─────────▼──────────┐                            │
│           │  ElevenLabs TTS    │  [futuro]                   │
│           └────────────────────┘                            │
│                                                              │
│  Hardware dev  : Windows 11 + RTX 4050  (onnxruntime-gpu)   │
│  Hardware prod : Orange Pi 5 RK3588 NPU (RKNN)              │
└──────────────────────────────────────────────────────────────┘
```

Todos los triggers producen objetos `Event` con el mismo formato (tipo, frame, buffer de contexto y metadata) y los empujan a la misma `queue.Queue`. El `EventDispatcher` los consume, aplica anti-spam por ventana temporal y los reenvía al webhook de n8n.

---

## Triggers

### ManualTrigger

Escucha una hotkey global mediante `pynput.GlobalHotKeys`. Al detectar la combinación, captura el frame actual de la cámara más un snapshot del ring buffer y emite un `Event(type=MANUAL)`.

- **Hotkey por defecto:** `Ctrl + Shift + G`
- Configurable en `config.yaml` bajo la clave `manual_trigger.hotkey`
- Funciona aunque la ventana de la aplicación no tenga foco

### YoloTrigger

Corre YOLOv8n en formato ONNX en su propio hilo, analizando frames a la tasa configurada (`analysis_fps`). Implementa seis heurísticas para generar alertas útiles para personas con discapacidad visual:

| # | Heurística | Descripción |
|---|-----------|-------------|
| 1 | **Metadata espacial** | Cada detección incluye `zone` (left/center/right) y `proximity` (near/medium/far) calculados a partir del bounding box relativo al frame. |
| 2 | **Sistema de prioridades** | Las clases se clasifican como `urgent` (vehículos), `important` (personas, semáforos) o `informative` (mobiliario). Cada prioridad tiene su propio cooldown independiente. |
| 3 | **Detección de cambio de escena** | Compara histogramas HSV entre frames mediante distancia de Bhattacharyya. Dispara un evento `scene_change` si el entorno cambia drásticamente. |
| 4 | **Detección de cambio en cantidad** | Dispara un evento `count_change` si el número de objetos de una clase varía en más de `count_change_delta` unidades respecto al frame anterior. |
| 5 | **Resumen periódico** | Emite un evento `scene_summary` cada `summary_interval_seconds` segundos con todas las detecciones activas en escena. |
| 6 | **Context frames para Gemini** | Adjunta al evento N frames equidistantes del ring buffer para dar contexto temporal (movimiento, dirección de objetos) a la descripción de Gemini. |

**Clases monitorizadas:** `person`, `car`, `bicycle`, `motorcycle`, `bus`, `truck`, `traffic light`, `stop sign`, `bench`, `chair`, `door`

**Cooldowns por prioridad (valores por defecto):**
- `urgent`: 2 s — vehículos que pueden representar peligro inmediato
- `important`: 5 s — personas y señales de tráfico
- `informative`: 15 s — mobiliario y elementos estáticos

### VoiceTrigger

Pipeline de reconocimiento de voz en dos etapas:

```
Micrófono (16 kHz, mono)
    │
    ▼
openWakeWord  →  evalúa chunks de 80ms → score por wake word
    │ score ≥ threshold
    ▼
Grabación de N segundos de audio
    │
    ▼
faster-whisper  →  transcripción en español con VAD
    │
    ▼
parse_voice_command()  →  intent (snapshot / describe / start_recording / stop_recording)
    │
    ▼
Event(type=VOICE, metadata={command, transcript})
```

- **Wake word:** `alexa` (modelo openWakeWord por defecto; "Lazarus" como palabra objetivo en producción)
- **Umbral de activación:** 0.5 (configurable)
- **Duración de grabación:** 4 segundos tras el wake word
- **Modelo Whisper:** `base`, transcripción en español (`es`)
- **Portabilidad:** `sounddevice` funciona tanto en Windows como en Linux/Orange Pi; `openWakeWord` usa ONNX internamente

---

## Scripts

### `scripts/process_video.py`

Procesa un video pregrabado con el mismo pipeline YOLO y de audio que el modo en tiempo real. Útil para depurar el sistema, preparar demostraciones o procesar grabaciones de campo de forma asíncrona.

Reutiliza directamente `inference/yolo_onnx.py`, las heurísticas de `triggers/yolo_trigger.py` y `config.yaml`, por lo que el comportamiento es idéntico al modo en vivo.

#### Modo `voice` (por defecto)

Detecta la palabra de activación "Lazarus" en el audio del video usando `faster-whisper` con timestamps por palabra. Por cada activación encontrada:

1. Cruza el timestamp con las detecciones YOLO activas en ese fotograma.
2. Extrae hasta 5 frames de contexto del video alrededor de ese momento.
3. Hace un POST multipart al webhook de n8n con el comando de voz y las imágenes.

Maneja variantes fonéticas que Whisper puede producir al transcribir "Lazarus" en español (`lazaro`, `laseros`, `lasalud`, `laceros`, etc.) mediante coincidencia exacta contra un conjunto de variantes conocidas y fuzzy matching con `SequenceMatcher`. También detecta bigramas ("La salud" → "lasalud").

#### Modo `manual`

Transcribe el audio completo del video sin buscar wake word. Hace un único POST a n8n con:

- La transcripción completa como campo `input`.
- Hasta 5 frames representativos: primero los triggers YOLO más importantes, luego frames equidistantes del video para completar hasta cinco.

#### Anotaciones en el video de salida

Cada frame del video de salida (`processed.mp4`) incluye bounding boxes coloreados por prioridad y un HUD con estadísticas:

| Color | Significado |
|-------|-------------|
| Rojo | Urgente (vehículos) |
| Amarillo | Importante (personas, semáforos) |
| Verde | Informativo (mobiliario) |
| Magenta | Cambio de escena |
| Naranja | Cambio de cantidad |

### `scripts/export_yolo.py`

Descarga YOLOv8n mediante `ultralytics` y lo exporta al formato ONNX con tamaño de entrada fijo de 640×640. Solo se necesita ejecutar una vez para generar `models/yolov8n.onnx`.

```bash
python scripts/export_yolo.py
```

---

## Instalación

**Requisitos previos:**
- Python 3.13
- CUDA Toolkit compatible con onnxruntime-gpu (opcional, para aceleración GPU en desarrollo)
- `ffmpeg` en el PATH del sistema (necesario para extraer audio en `process_video.py`)

**Pasos:**

```bash
# 1. Clonar el repositorio y entrar al cliente
git clone <url-del-repositorio>
cd Lazarus/vision_assistant

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Generar el modelo ONNX (descarga ~6 MB y genera models/yolov8n.onnx)
python scripts/export_yolo.py
```

> **Nota:** Las carpetas `models/` y `output/` están en `.gitignore` y deben generarse localmente. El modelo ONNX no está incluido en el repositorio.

> **Orange Pi 5:** Reemplaza `onnxruntime-gpu` por `onnxruntime` en `requirements.txt` antes de instalar. La conversión a RKNN para la NPU está en el roadmap.

---

## Uso

### Modo en tiempo real

```bash
python main.py
```

Inicia todos los componentes en orden: `CameraService` → `EventDispatcher` → `ManualTrigger` → `YoloTrigger` → `VoiceTrigger`. Presiona `Ctrl+C` para un shutdown limpio que drena la cola de eventos antes de salir.

### Procesamiento de video — modo voz

Detecta la palabra "Lazarus" en el audio y hace un POST a n8n por cada activación:

```bash
python scripts/process_video.py \
  --video ruta/al/video.mp4 \
  --mode voice \
  --wake-word lazarus \
  --n8n-url https://tu-instancia.n8n.cloud/webhook/XXXX \
  --user-id lazarus_user_001
```

Para videos largos, usar YOLO cada 3 frames reduce significativamente el tiempo de procesamiento:

```bash
python scripts/process_video.py \
  --video ruta/al/video.mp4 \
  --mode voice \
  --yolo-every 3 \
  --n8n-url https://tu-instancia.n8n.cloud/webhook/XXXX
```

### Procesamiento de video — modo manual

Transcribe el audio completo y hace un único POST con la transcripción y frames representativos:

```bash
python scripts/process_video.py \
  --video ruta/al/video.mp4 \
  --mode manual \
  --yolo-every 3 \
  --n8n-url https://tu-instancia.n8n.cloud/webhook/XXXX
```

### Referencia de argumentos de `process_video.py`

| Argumento | Descripción | Valor por defecto |
|-----------|-------------|-------------------|
| `--video` | Ruta al video de entrada | *(requerido)* |
| `--config` | Ruta al archivo de configuración | `config.yaml` |
| `--mode` | `voice` o `manual` | `voice` |
| `--wake-word` | Palabra de activación | `lazarus` |
| `--whisper-model` | Modelo Whisper (`base`, `small`, `medium`) | Valor en `config.yaml` |
| `--yolo-every` | Ejecutar YOLO cada N frames | `1` |
| `--n8n-url` | URL del webhook n8n | *(opcional, sin POST si se omite)* |
| `--user-id` | ID de usuario enviado en el POST | `lazarus_user_001` |
| `--show` | Mostrar preview en tiempo real | `false` |
| `--output-fps` | FPS del video de salida | Igual que el original |

---

## Configuración

El archivo `config.yaml` controla todos los parámetros del sistema:

```yaml
camera:
  device_index: 0         # índice del dispositivo de cámara
  width: 1280
  height: 720
  fps: 30
  buffer_seconds: 10      # segundos de video almacenados en el ring buffer

manual_trigger:
  hotkey: "<ctrl>+<shift>+g"   # formato pynput

yolo_trigger:
  enabled: true
  model_path: "models/yolov8n.onnx"
  analysis_fps: 10              # frames analizados por segundo (hilo dedicado)
  confidence_threshold: 0.6     # score mínimo para aceptar una detección
  cooldown_seconds: 5.0         # cooldown base (reemplazado por priority_cooldowns)
  stability_frames: 3           # mínimo de apariciones requeridas en la ventana
  stability_window: 5           # tamaño de la ventana de estabilidad (en frames)
  class_whitelist: [...]        # solo estas clases generan eventos
  priority_cooldowns:           # cooldowns diferenciados por nivel de prioridad
    urgent: 2
    important: 5
    informative: 15
  class_priority: {...}         # asignación clase → nivel de prioridad
  scene_change_enabled: true
  scene_change_threshold: 0.45  # distancia Bhattacharyya (0=idéntico, 1=opuesto)
  scene_change_cooldown: 10
  count_change_enabled: true
  count_change_delta: 2         # delta mínimo de objetos para disparar evento
  count_change_cooldown: 10
  summary_enabled: true
  summary_interval_seconds: 30
  context_frames_count: 5       # frames del ring buffer adjuntos al evento

voice_trigger:
  enabled: true
  wake_word_model: "alexa"      # nombre del modelo openWakeWord
  wake_word_threshold: 0.5      # score mínimo para activar
  recording_seconds: 4          # segundos grabados tras detectar el wake word
  whisper_model: "base"
  whisper_language: "es"
  sample_rate: 16000

dispatcher:
  dedupe_window_seconds: 1.0    # eventos del mismo tipo en esta ventana se descartan
  output_dir: "output"

logging:
  level: "INFO"
  file: "logs/app.log"
```

---

## Estructura de salida

Cada ejecución de `process_video.py` genera una carpeta única con timestamp:

```
output/run_YYYYMMDD_HHMMSS/
│
├── processed.mp4                        Video completo con bboxes anotados y HUD
│
├── frame_NNNNNN_<clase>.jpg             Frame en el momento del trigger YOLO
├── frame_NNNNNN_<clase>.json            Metadata: clase, prioridad, detecciones, timestamp
├── frame_NNNNNN_<clase>_ctx_0.jpg       Frames de contexto temporal (hasta 5)
├── frame_NNNNNN_<clase>_ctx_N.jpg
│   ...
│
├── voice_NNNN_ctx_0.jpg                 Frames de contexto de cada activación de voz
├── voice_NNNN_ctx_N.jpg
│   ...
│
├── voice_requests.json                  [modo voice] Todas las activaciones detectadas
├── manual_request.json                  [modo manual] Transcripción completa + frames
│
└── summary.json                         Resumen completo de la ejecución
```

**Ejemplo de entrada en `voice_requests.json`:**

```json
{
  "request_index": 0,
  "timestamp_s": 12.34,
  "wake_word_detected": "lazaro",
  "command": "describe lo que ves",
  "user_id": "lazarus_user_001",
  "yolo_detections_at_trigger": [
    {
      "class_name": "person",
      "confidence": 0.87,
      "bbox": [240, 80, 560, 620],
      "zone": "center",
      "proximity": "near"
    }
  ],
  "yolo_context_text": "person (center/near)",
  "context_frame_paths": ["voice_0000_ctx_0.jpg", "voice_0000_ctx_1.jpg"]
}
```

En el modo en tiempo real, el `EventDispatcher` guarda cada evento procesado en `output/` con el formato `{timestamp}_{TIPO}.jpg` y `{timestamp}_{TIPO}.json`.

---

## Portabilidad: ONNX → RKNN

El módulo `inference/yolo_onnx.py` abstrae completamente el backend de inferencia. El código que llama a `YoloOnnx.detect()` no necesita cambiar al portar entre plataformas:

| Entorno | Backend | Dependencia |
|---------|---------|-------------|
| Windows 11 + RTX 4050 (desarrollo) | `CUDAExecutionProvider` → fallback `CPUExecutionProvider` | `onnxruntime-gpu` |
| Orange Pi 5 RK3588 (producción, CPU) | `CPUExecutionProvider` | `onnxruntime` |
| Orange Pi 5 RK3588 NPU (futuro) | RKNN custom provider | `rknn-toolkit2` |

La selección del provider es automática: si CUDA está disponible se usa; de lo contrario, se cae silenciosamente a CPU sin ningún cambio en el código que llama al módulo.

---

## Estructura del proyecto

```
vision_assistant/
├── main.py                     # Punto de entrada del sistema en tiempo real
├── config.yaml                 # Configuración central de todos los componentes
├── requirements.txt
│
├── core/
│   ├── events.py               # Dataclass Event y enums EventType, VoiceCommand
│   ├── dispatcher.py           # Consumidor de la cola con anti-spam y handler
│   └── config_loader.py        # Carga y validación de config.yaml
│
├── services/
│   └── camera.py               # CameraService: captura continua + ring buffer
│
├── triggers/
│   ├── base.py                 # BaseTrigger: _emit_event() compartido
│   ├── manual_trigger.py       # Hotkey global con pynput
│   ├── yolo_trigger.py         # Detección automática con YOLOv8 ONNX
│   └── voice_trigger.py        # Wake word (openWakeWord) + transcripción (Whisper)
│
├── inference/
│   └── yolo_onnx.py            # Wrapper ONNX: preprocess → inferencia → NMS
│
├── scripts/
│   ├── process_video.py        # Procesamiento offline de video con audio
│   └── export_yolo.py          # Exporta YOLOv8n a ONNX
│
└── utils/
    ├── logger.py               # Configuración de loguru
    └── audio.py                # normalize_audio, parse_voice_command
```

---

## Dependencias principales

| Paquete | Rol |
|---------|-----|
| `opencv-python` | Captura de cámara, procesamiento de imagen, escritura de video anotado |
| `onnxruntime-gpu` | Inferencia ONNX con CUDA (reemplazar por `onnxruntime` en Orange Pi) |
| `ultralytics` | Solo para exportar YOLOv8 a ONNX; no se usa en runtime |
| `pynput` | Hotkey global para `ManualTrigger` |
| `openwakeword` | Detección de wake word en tiempo real (ONNX internamente) |
| `faster-whisper` | Transcripción de voz eficiente en CPU con VAD integrado |
| `sounddevice` | Captura de audio del micrófono |
| `pyyaml` | Carga de `config.yaml` |
| `loguru` | Logging estructurado con rotación de archivos |

---

## Roadmap del cliente

- [x] Sistema de triggers (manual, YOLO, voz)
- [x] Procesamiento offline de video con transcripción de audio
- [x] Heurísticas avanzadas de YOLO: metadata espacial, prioridades, cambio de escena, cambio de cantidad, resumen periódico, context frames
- [ ] Integración n8n en tiempo real dentro de `main.py` (actualmente el dispatcher solo guarda a disco)
- [ ] Reproducción local del audio devuelto por el agente (ElevenLabs)
- [ ] Exportación RKNN para NPU del Orange Pi 5
- [ ] Wake word personalizado "Lazarus" con Porcupine
