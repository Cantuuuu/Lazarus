# Lazarillo — Asistente de Visión Edge para Personas Ciegas

Sistema wearable de visión artificial que detecta objetos en tiempo real y describe el entorno por voz. Corre completamente en local sobre una **Orange Pi 5 Max** (NPU RK3588) conectada a una **Raspberry Pi Zero W** como cámara inalámbrica y auriculares Bluetooth.

---

## Cómo funciona

```
┌─────────────────┐   MJPEG/WiFi   ┌────────────────────────────────────┐   Audio   ┌───────────────┐
│ Raspberry Pi    │ ─────────────→ │ Orange Pi 5 Max  (RK3588 NPU)      │ ────────→ │ Auriculares   │
│ Zero W          │                │                                    │           │ Bluetooth     │
│ • Cámara        │                │ • YOLO v8n → NPU (30ms / 33 FPS)  │           │               │
│ • Stream MJPEG  │                │ • Motor de alertas (cooldown/clase) │           │               │
└─────────────────┘                │ • TTS ElevenLabs (cache LRU)       │           └───────────────┘
                                   │ • STT Whisper local (español)      │
                                   │ • Gemini (descripción de escena)   │
                                   │ • Orquestador proactivo/reactivo   │
                                   └────────────────────────────────────┘
```

El sistema alterna entre dos modos de operación coordinados por el `Orchestrator`:

| Modo | Activación | Qué hace |
|------|-----------|----------|
| **Proactivo** | Al iniciar (o botón) | Analiza frames a 5 FPS, genera alertas de voz automáticas con cooldown por clase |
| **Reactivo** | Pregunta del usuario vía `/ask` | Pausa el proactivo, captura frame, consulta Gemini con la imagen y responde por voz |

---

## Implementado

### Visión y detección
- **StreamReader** — cliente async MJPEG con reconexión automática (backoff exponencial 1s→30s)
- **Detector** — YOLOv8n con tres backends y fallback automático:
  - `RKNNDetector` → NPU RK3588, **~30ms** por frame (33 FPS)
  - `ONNXDetector` → CPU ARM, ~205ms por frame (5 FPS)
  - `MockDetector` → desarrollo local sin hardware
- **TrafficLight** — clasificador HSV de semáforos (rojo/verde/amarillo) integrado en el modo proactivo

### Alertas y audio
- **AlertEngine** — cooldowns independientes por clase (persona: 8s, auto: 5s, semáforo: 3s), mensajes de dirección en español
- **AudioManager** — cola de prioridad (`asyncio.PriorityQueue`), control de interrupciones, soporte ffplay/aplay/mpg123
- **ElevenLabsClient** — TTS async con cache LRU para frases repetidas (latencia: 0ms en cache, ~600ms en cold)

### Modos de operación
- **ProactiveMode** — loop a 5 FPS: frame → YOLO → AlertEngine → TTS → audio
- **ReactiveMode** — pregunta → frame actual → Gemini Vision → TTS → audio (prioridad máxima en cola)
- **Orchestrator** — coordina modos con mutex, pausa proactivo durante preguntas, reanuda automáticamente

### Voz y lenguaje
- **WhisperSTT** — transcripción local con `openai-whisper` (modelo tiny, español, sin internet)
- **GeminiClient** — descripción de escena con Gemini Vision, rate limiting y fallback sin conexión
- **VoiceCapture** — captura de audio con VAD

### API HTTP (FastAPI)
| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/health` | Estado del sistema, modo activo, conexión stream |
| `GET` | `/detections` | Detecciones JSON del frame actual |
| `GET` | `/preview` | JPEG con bounding boxes anotados |
| `GET` | `/stream` | MJPEG en vivo con detecciones anotadas |
| `POST` | `/trigger/button` | Alterna proactivo ↔ pausado (simula botón físico) |
| `POST` | `/mode/{mode}` | Cambia modo directamente: `proactive` \| `paused` |
| `POST` | `/ask` | Responde pregunta sobre la escena actual |

---

## Pipeline NPU: ONNX → RKNN

```
YOLOv8n PyTorch ──→ Export ONNX (12MB) ──→ Docker rknn-toolkit2 ──→ RKNN (8MB) ──→ Orange Pi NPU
                                                ↓
                                         convert.py
                                    (rknn.config + build)
                                         2–5 min
```

```bash
# Convertir modelo en x86 (Mac/Linux)
docker compose run --rm rknn-convert

# El archivo .rknn queda en docker/rknn-convert/
# Copiarlo a la Orange Pi:
scp models/rknn/yolov8n.rknn orangepi@<ip>:~/Proyectos/Lazarillo/models/rknn/

# Validar en hardware
python tests/npu_validation.py models/rknn/yolov8n.rknn
```

**Benchmarks en Orange Pi 5 Max:**

| Backend | Latencia | FPS | Consumo |
|---------|----------|-----|---------|
| RKNN (NPU) | ~30ms | 33 | 5–12W |
| ONNX (CPU) | ~205ms | 5 | 8–15W |
| Mock (dev) | <1ms | ∞ | — |

---

## Quick Start

### Requisitos

- Python 3.10+
- Docker + Docker Compose
- Orange Pi 5 Max con Ubuntu ARM64 (para NPU)

### Desarrollo local (sin hardware)

```bash
git clone https://github.com/Cantuuuu/Lazarus.git
cd Lazarus/Lazarillo_Edge_Computing

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Copia y edita variables de entorno
cp .env.example .env

# Levantar con mock stream y MockDetector
docker compose up --build

# Verificar
curl http://localhost:8000/health
curl http://localhost:8000/detections
```

### Deploy en Orange Pi

```bash
# En la Orange Pi
git clone https://github.com/Cantuuuu/Lazarus.git
cd Lazarus/Lazarillo_Edge_Computing

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configurar entorno
cp .env.example .env
# Editar .env con tus API keys y ruta del modelo RKNN

# Ejecutar
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Configuración

```bash
# .env
GEMINI_API_KEY=<tu-key>
ELEVENLABS_API_KEY=<tu-key>
ELEVENLABS_VOICE_ID=<id>

PI_ZERO_STREAM_URL=http://<ip-pi-zero>:8080/stream.mjpg

YOLO_MODEL_PATH=models/rknn/yolov8n.rknn   # o .onnx para CPU
YOLO_CONFIDENCE_THRESHOLD=0.4

WHISPER_MODEL_SIZE=tiny
AUDIO_DEVICE=default

ENV=production   # development usa MockDetector sin hardware
```

---

## Tests

```bash
# Unitarios (no requieren hardware)
pytest tests/unit/ -v

# Con cobertura
pytest --cov=core --cov=services --cov-report=term-missing

# Validación NPU (requiere Orange Pi con modelo RKNN)
python tests/npu_validation.py models/rknn/yolov8n.rknn
```

---

## Estructura

```
Lazarillo_Edge_Computing/
├── core/
│   ├── detector.py          # YOLOv8n (RKNN / ONNX / Mock)
│   ├── alert_engine.py      # Cooldowns y mensajes en español
│   ├── audio_manager.py     # Cola de audio con prioridad
│   ├── orchestrator.py      # Coordinación proactivo/reactivo
│   ├── traffic_light.py     # Clasificador semáforos HSV
│   ├── voice_capture.py     # Captura de audio con VAD
│   └── stt.py               # Whisper local (español)
├── modes/
│   ├── proactive.py         # Loop YOLO → alertas automáticas
│   └── reactive.py          # Pregunta → Gemini → respuesta
├── services/
│   ├── stream_reader.py     # Cliente MJPEG async
│   ├── elevenlabs_client.py # TTS con cache LRU
│   └── gemini_client.py     # Descripción de escena
├── docker/
│   └── rknn-convert/        # Conversión ONNX → RKNN (x86)
├── tests/
│   ├── unit/                # Tests unitarios (sin hardware)
│   ├── fixtures/            # Mock stream y frames de prueba
│   └── npu_validation.py   # Benchmark y validación en Orange Pi
├── scripts/
│   ├── convert_model.py     # Conversión YOLO a RKNN
│   ├── setup_bluetooth.sh   # Configurar audio Bluetooth
│   └── test_audio.py        # Prueba de audio en hardware
├── docs/
│   ├── ARCHITECTURE.md      # Componentes y flujo de datos
│   ├── NPU_PIPELINE.md      # Guía completa ONNX → RKNN
│   ├── NPU_SETUP.md         # Setup Orange Pi, drivers, benchmarks
│   └── CODEMAPS/            # Mapas técnicos por módulo
├── main.py                  # FastAPI app (7 endpoints)
├── config.py                # Pydantic Settings
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Hardware

| Componente | Modelo | Rol |
|------------|--------|-----|
| Procesador edge | Orange Pi 5 Max (RK3588) | Inferencia NPU, lógica, audio |
| Cámara | Raspberry Pi Zero W + PiCamera v2 | Stream MJPEG via WiFi |
| Audio | Auriculares Bluetooth 5.0 | Salida de alertas y respuestas |

---

## Contacto

Gabriel Gudiño Lara — gabriels114@gmail.com
