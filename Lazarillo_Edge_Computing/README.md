# Lazarillo

Asistente de visión artificial wearable para personas ciegas. Detecta objetos en tiempo real usando una Raspberry Pi Zero W (streaming de cámara) + Orange Pi 5 Max (procesamiento NPU) + auriculares Bluetooth.

**Status:** MVP en desarrollo (Fase 1-2 completas)

## Sistema en 60 segundos

```
┌─────────────────┐         ┌──────────────────────┐         ┌─────────┐
│ Raspberry Pi    │         │ Orange Pi 5 Max      │         │ Audio   │
│ Zero W          │         │ (RK3588 NPU)         │         │ Output  │
│                 │         │                      │         │         │
│ • Cámara        │ MJPEG   │ • StreamReader       │ Alert   │ • TTS   │
│ • Streaming     ├────────→│ • Detector (YOLO)    ├────────→│ • Bluetooth
└─────────────────┘         │ • AlertEngine        │         └─────────┘
                            │ • FastAPI            │
                            │ • Modos (proactivo/  │
                            │   reactivo/memoria)  │
                            └──────────────────────┘
                                      ↓
                            ┌──────────────────────┐
                            │ Google Cloud         │
                            │ • Gemini LLM         │
                            │ • ElevenLabs TTS     │
                            │ • MongoDB session    │
                            └──────────────────────┘
```

## Características (completadas)

✅ **Infraestructura de desarrollo**
- Mock stream MJPEG (servidor Flask con video en loop)
- Ambiente Docker con docker-compose
- Configuración con Pydantic Settings

✅ **Detección de objetos**
- StreamReader async con reconexión automática
- Detector YOLO con 3 backends:
  - MockDetector (dev local sin hardware)
  - ONNXDetector (CPU con onnxruntime)
  - RKNNDetector (NPU Orange Pi RK3588)
- Post-procesado YOLO (NMS, thresholding, mapeo COCO)
- Fallback chain: RKNN → ONNX → Mock

✅ **Motor de alertas**
- Cooldowns independientes por clase (5-10s)
- Mensajes de dirección en español (izquierda/frente/derecha)
- Per-class throttling para evitar spam

✅ **Testing**
- 32 tests unitarios (100% cobertura en Fase 1-2)
- 15 tests pasan en Orange Pi hardware
- Fixtures de mock stream
- pytest con asyncio_mode=auto

## En progreso

🔄 **Modelo YOLO**
- Conversión yolov8n.onnx → .rknn via Docker (rknn-toolkit2 x86)
- Validación NPU en hardware (RKNPU 0.9.8, runtime 2.3.0)
- Inference CPU: ~205ms en Orange Pi

## Próximos pasos

❌ **Fase 2: Modo Proactivo** (alertas automáticas)
- Clasificador de semáforos HSV
- Cliente ElevenLabs TTS
- AudioManager con cola de prioridad
- Loop principal: frame → detect → alert → speak

❌ **Fase 3: Modo Reactivo** (preguntas por voz)
- VoiceCapture con VAD (webrtcvad)
- STT con faster-whisper (español)
- Pipeline: audio → transcription → Gemini → TTS

❌ **Fase 4: Modo Memoria** (grabación y reportes)
- MongoDB para sesiones
- Keyframes y reporte con Gemini Pro

❌ **Fase 5: Hardware real**
- Pi Zero W script streaming
- Conversión YOLO a RKNN (en progreso)
- BlueZ + PipeWire para Bluetooth
- Deploy con systemd

❌ **Fase 6: Pulido para demo**
- Cache TTS
- Dashboard web
- Fallbacks en puntos de falla

## Quick Start

### Requisitos

- Python 3.10+
- Docker + Docker Compose
- Modelo YOLO (descarga abajo)

### Instalación

```bash
# Clonar proyecto
git clone <repo>
cd Lazarillo

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # o: .venv\Scripts\activate en Windows

# Instalar dependencias
pip install -r requirements.txt

# Descargar modelo YOLO (opcional — MockDetector en dev)
mkdir -p models/rknn
# Descarga desde: https://github.com/airockchip/rknn_model_zoo
# wget -O models/rknn/yolov8n.onnx <url>
```

### Ejecutar con Docker

```bash
# Build + start todos los servicios
docker compose up --build

# En otra terminal, probar endpoints
curl http://localhost:8000/health
curl http://localhost:8000/detections
```

### Ejecutar localmente (sin Docker)

```bash
# Terminal 1: Mock stream
python tests/fixtures/mock_stream.py

# Terminal 2: Lazarillo
python -m uvicorn main:app --reload

# Terminal 3: Tests
pytest tests/unit/ -v
```

## Endpoints HTTP

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/health` | Status del sistema + stream |
| `GET` | `/detections` | Detecciones JSON del frame actual |
| `GET` | `/preview` | Frame actual con bboxes anotados |
| `GET` | `/stream` | MJPEG vivo con detecciones anotadas |

Ejemplo:
```bash
curl -s http://localhost:8000/detections | jq
# {
#   "count": 2,
#   "detections": [
#     {"class": "person", "confidence": 0.92, "direction": "frente", ...},
#     {"class": "car", "confidence": 0.87, "direction": "derecha", ...}
#   ]
# }
```

## Configuración

Variables de entorno en `.env`:

```bash
# APIs externas
GEMINI_API_KEY=<tu-key>
ELEVENLABS_API_KEY=<tu-key>
ELEVENLABS_VOICE_ID=<id>
MONGODB_URI=mongodb://localhost:27017

# Hardware
PI_ZERO_STREAM_URL=http://stream-mock:8080/stream.mjpg

# YOLO
YOLO_MODEL_PATH=models/rknn/yolov8n.rknn
YOLO_CONFIDENCE_THRESHOLD=0.4

# Audio
WHISPER_MODEL_SIZE=tiny
AUDIO_DEVICE=default

# Entorno
ENV=development
```

## Edge Computing

Lazarillo corre **completamente offline** en Orange Pi 5 Max (RK3588 NPU):

```
Inferencia YOLO:  30ms @ 640×640 (33 FPS)
Latencia total:   ~600ms (detección → TTS → audio)
Hardware:         RK3588 NPU + 8 cores ARM + 16GB RAM
Consumo:          5-12W normal, 2-3W (NPU solo)
Fallbacks:        CPU ONNX (5 FPS), Mock (desarrollo)
```

**Pipeline ONNX → RKNN:**
1. Modelo: YOLOv8n (12MB ONNX → 4MB RKNN)
2. Conversión: Docker + rknn-toolkit2 (2-5 min)
3. Runtime: rknnlite en Orange Pi
4. Validación: tests/npu_validation.py

Consulta **[NPU_PIPELINE.md](docs/NPU_PIPELINE.md)** para procedimiento completo.

## Arquitectura

Documentación detallada en:
- **[CODEMAPS/edge_computing.md](docs/CODEMAPS/edge_computing.md)** — Mapa técnico completo (streamreader, detector, modes, services)
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — Componentes, data flow, hardware
- **[NPU_PIPELINE.md](docs/NPU_PIPELINE.md)** — Conversión ONNX → RKNN, validación NPU
- **[DEVELOPMENT.md](docs/DEVELOPMENT.md)** — Setup dev, tests, Docker, SSH
- **[NPU_SETUP.md](docs/NPU_SETUP.md)** — Troubleshooting hardware, benchmarks

## Testing

```bash
# Todos los tests
pytest

# Tests unitarios solamente
pytest tests/unit/ -v

# Con cobertura
pytest --cov=core --cov=services --cov-report=html

# En Orange Pi (hardware real)
ssh orangepi@<ip>
cd ~/lazarillo
pytest tests/npu_validation.py models/rknn/yolov8n.rknn
```

## Estructura del código

```
Lazarillo/
├── core/                    # Lógica principal
│   ├── detector.py         # YOLO wrapper (MockDetector, ONNXDetector, RKNNDetector)
│   ├── alert_engine.py     # Cooldowns y mensajes de alerta
│   └── __init__.py
├── services/               # Integraciones externas
│   ├── stream_reader.py    # Lee MJPEG con reconexión automática
│   └── __init__.py
├── modes/                  # Modos de operación
│   ├── proactive.py        # Alertas automáticas (próximo)
│   ├── reactive.py         # Preguntas por voz (próximo)
│   ├── memory.py           # Grabación de sesiones (próximo)
│   └── __init__.py
├── tests/
│   ├── fixtures/           # Mock stream, test data
│   ├── unit/               # Tests unitarios
│   └── npu_validation.py   # Validación hardware Orange Pi
├── docker/
│   └── rknn-convert/       # Docker para conversión YOLO→RKNN
├── scripts/
│   ├── convert_model.py    # Conversión YOLO (en x86)
│   └── deploy.sh           # Deploy a hardware (próximo)
├── models/                 # Modelos YOLO (git-ignored)
│   ├── rknn/
│   │   ├── yolov8n.onnx
│   │   └── yolov8n.rknn
│   └── ...
├── main.py                 # FastAPI app
├── config.py               # Pydantic Settings
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
└── README.md
```

## Decisiones técnicas

1. **ONNXDetector sobre cv2.dnn**: opencv-python-headless 4.10.0 en ARM no soporta DNN. onnxruntime es más portable.

2. **Fallback chain (RKNN→ONNX→Mock)**: Permite desarrollo en Mac/Linux, fallback a CPU si NPU no disponible, siempre funciona.

3. **YOLOv8n (nano)**: 12MB ONNX, ~4MB RKNN esperado. Rápido en CPU/NPU, buena precisión.

4. **numpy>=2.0**: RKNN toolkit2 para *conversión* requiere numpy<=1.26.4, pero el *runtime* funciona con 2.x. Incompatibilidad documentada.

5. **Alertas con cooldown por clase**: Evita spam. Semáforos: 3s, persona: 8s, auto: 5s.

6. **Pydantic Settings**: Config centralizada, variables de entorno, defaults sensatos.

## Flujo típico (Fase 1)

```python
# main.py inicia StreamReader + Detector
stream_reader = StreamReader(url)
detector = build_detector(dev_mode=True)  # MockDetector en dev

# Endpoint /detections
frame = await stream_reader.get_frame()
detections = detector.detect(frame)  # list[Detection]

# Cada Detection tiene:
# - class_id, class_name, confidence
# - bbox (x1, y1, x2, y2)
# - properties: is_priority, center_x, direction
```

## Próximo: Fase 2

```python
# Mode proactivo (en modes/proactive.py — próximo)
from core.alert_engine import AlertEngine
from services.elevenlabs_client import ElevenLabsClient

engine = AlertEngine()
tts = ElevenLabsClient(api_key)

async for frame in stream_reader.frames():
    detections = detector.detect(frame)
    for det in detections:
        if engine.should_alert(det):
            msg = engine.direction_message(det)  # "persona al frente"
            await tts.synthesize_and_play(msg)
            engine.mark_alerted(det)
```

## Troubleshooting

### Mock stream no conecta
```bash
# Revisar que stream-mock esté corriendo
docker logs <container-id>
curl http://localhost:8080/stream.mjpg
```

### Tests fallan en Orange Pi
```bash
# Verificar NPU disponible
ls /dev/dri/renderD*
python tests/npu_validation.py models/rknn/yolov8n.rknn
```

### Latencia alta
- Reducir resolución frame (640→416)
- Usar MockDetector + cache TTS
- Benchmark local: `tests/npu_validation.py` muestra FPS teóricos

## Licencia

MIT

## Contacto

Gabriel Gudiño Lara — gabriels114@gmail.com
