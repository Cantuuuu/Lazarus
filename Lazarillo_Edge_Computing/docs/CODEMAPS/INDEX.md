# Codemaps Index — Lazarillo

**Last Updated:** 2026-04-18

Documentación técnica generada del codebase. Cada mapa cubre un área específica del sistema con enfoque en arquitectura, módulos, dependencias y data flow.

---

## 📋 Mapas Disponibles

### 1. Edge Computing System
**Archivo:** [`edge_computing.md`](edge_computing.md)

Mapa completo del sistema de visión en tiempo real. Cubre:
- Input layer (StreamReader)
- Inference engine (Detector con backends RKNN/ONNX/Mock)
- Decision layer (AlertEngine)
- Orchestration (Orchestrator + Modos)
- Output layer (AudioManager)
- Servicios externos (ElevenLabs, Gemini)

**Cuando consultar:** Entender arquitectura de datos, dónde agregar features, cómo fluyen requests.

---

### 2. NPU Pipeline (ONNX → RKNN → Inference)
**Archivo:** [`../NPU_PIPELINE.md`](../NPU_PIPELINE.md)

Procedimiento completo de conversión de modelos YOLO. Cubre:
- Descargar ONNX desde model zoo
- Convertir ONNX → RKNN con Docker
- Copiar a Orange Pi
- Validar con npu_validation.py
- Troubleshooting y benchmarks

**Cuando consultar:** Necesitas convertir un modelo, validar hardware, optimizar latencia.

---

### 3. Architecture (Componentes y Data Flow)
**Archivo:** [`../ARCHITECTURE.md`](../ARCHITECTURE.md)

Documentación de arquitectura general. Cubre:
- Diagrama de componentes (entrada → procesamiento → salida)
- Descripción detallada de cada módulo
- Data flow por modo de operación
- Optimización de latencia
- Tolerancia a fallos y fallback chain
- Testing strategy

**Cuando consultar:** Entender decisiones de diseño, revisar componentes antes de cambios.

---

## 🏗️ Estructura del Codebase

```
core/
├── detector.py          # Wrapper YOLO (MockDetector, ONNXDetector, RKNNDetector)
├── alert_engine.py      # Cooldowns y mensajes en español
├── audio_manager.py     # Cola de audio con prioridad
├── orchestrator.py      # Coordinador de modos
├── traffic_light.py     # Clasificador HSV semáforos
├── voice_capture.py     # Captura VAD + audio
├── stt.py              # Transcripción Whisper
└── __init__.py

services/
├── stream_reader.py     # Lector MJPEG async
├── elevenlabs_client.py # TTS con cache LRU
├── gemini_client.py     # Visión + LLM
└── __init__.py

modes/
├── proactive.py         # Loop YOLO → alertas → TTS
├── reactive.py          # Pregunta → Gemini → respuesta
├── memory.py            # (Futuro) Grabación de sesiones
└── __init__.py

main.py                  # FastAPI app + lifespan
config.py               # Pydantic Settings
```

---

## 🔍 Módulos Clave por Responsabilidad

### Input Layer
- **StreamReader** (`services/stream_reader.py`)
  - Conecta con stream MJPEG
  - Maneja desconexiones automáticas
  - Thread-safe con asyncio.Lock

### Inference Layer
- **Detector** (`core/detector.py`)
  - Factory pattern con fallback (RKNN → ONNX → Mock)
  - Post-procesado YOLO (NMS, thresholding, scaling)
  - Detection dataclass (inmutable)

### Decision Layer
- **AlertEngine** (`core/alert_engine.py`)
  - Cooldowns por clase (person=8s, car=5s, etc.)
  - Mensajes en español
  - Immutable state management

### Orchestration Layer
- **Orchestrator** (`core/orchestrator.py`)
  - Coordina ProactiveMode ↔ ReactiveMode
  - Mutex para evitar solapamiento
  - Manejo de button trigger

- **ProactiveMode** (`modes/proactive.py`)
  - Loop principal @ 5 FPS
  - Detección → Alert → TTS → Audio

- **ReactiveMode** (`modes/reactive.py`)
  - Pregunta → Gemini → respuesta
  - Pausa modo proactivo durante consulta

### Output Layer
- **AudioManager** (`core/audio_manager.py`)
  - Cola asyncio.PriorityQueue
  - Reproducción con ffplay/aplay/mpg123
  - Control de interrupción

### External Services
- **ElevenLabsClient** (`services/elevenlabs_client.py`)
  - TTS con cache LRU (50 entradas)
  - Dev mode con bytes vacíos

- **GeminiClient** (`services/gemini_client.py`)
  - Visión + LLM
  - Rate limiting (10 rpm)
  - Fallback local

---

## 📊 Datos Principales

### Detection
```python
@dataclass(frozen=True)
class Detection:
    class_id: int                          # 0=person, 2=car
    class_name: str                        # COCO label
    confidence: float                      # 0.0-1.0
    bbox: tuple[int, int, int, int]       # x1, y1, x2, y2

    @property
    def is_priority(self) -> bool:         # ¿Alerta?
    
    @property
    def center_x(self) -> int:             # Para dirección
    
    @property
    def direction(self) -> str:            # izq/frente/der
```

### COCO Classes Prioritarias
```
person (0), bicycle (1), car (2), motorcycle (3), bus (5), truck (7),
traffic_light (9), stop_sign (11), cat (15), dog (16)
```

### Cooldowns por Clase
```
car: 5.0s
person: 8.0s
traffic_light: 3.0s
dog: 10.0s
default: 5.0s
```

---

## 🔄 Data Flow Principales

### Modo Proactivo
```
StreamReader → Detector.detect() → AlertEngine.should_alert()
    ↓
    └─→ (si True): direction_message() → TTS.generate() 
                   → AudioManager.speak(priority) → ffplay/aplay
```

### Modo Reactivo
```
POST /ask → Orchestrator.handle_question()
    ├─ Pausa ProactiveMode
    ├─ StreamReader.get_frame()
    ├─ Gemini.describe_scene(frame, question)
    ├─ TTS.generate(answer)
    ├─ AudioManager.speak(answer, priority=1)
    └─ Reanuda ProactiveMode
```

---

## 🧪 Testing

### Coverage (178 tests)
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
────────────────────────────────
TOTAL:                   178 tests ✅
```

### Ejecutar Tests
```bash
pytest                                    # Todos
pytest tests/unit/test_detector.py -v    # Un módulo
pytest --cov=core --cov-report=html      # Con cobertura
```

---

## 📈 Benchmarks

### Orange Pi 5 Max RK3588 @ 800MHz

| Componente | Latencia | FPS |
|-----------|----------|-----|
| YOLO (RKNN) | 30ms | 33 |
| YOLO (ONNX CPU) | 205ms | 5 |
| MockDetector | <1ms | ∞ |
| **Total (cached TTS)** | **~165ms** | — |
| **Total (fresh TTS)** | **~600ms** | — |

---

## 🚀 Próximas Fases

### Fase 2 (Completada)
- ✅ ProactiveMode con loop YOLO
- ✅ AudioManager con cola de prioridad
- ✅ ElevenLabsClient TTS

### Fase 3 (En progreso)
- VoiceCapture con VAD
- Whisper STT
- ReactiveMode preguntas-respuestas
- Orchestrator coordinación

### Fase 4-6
- Memory mode + MongoDB
- Dashboard web
- Bluetooth pairing
- Deploy systemd

---

## 🔗 Enlaces Relacionados

- **README.md** — Descripción general del proyecto
- **ARCHITECTURE.md** — Arquitectura detallada
- **NPU_PIPELINE.md** — Conversión YOLO y validación
- **NPU_SETUP.md** — Hardware, benchmarks, troubleshooting
- **DEVELOPMENT.md** — Setup dev, Docker, tests
- **ESTADO_PROYECTO.md** — Estado actual y tareas pendientes

---

## 🎯 Guía Rápida: ¿A dónde ir?

| Pregunta | Consulta |
|----------|----------|
| ¿Cómo fluyen los datos en el sistema? | edge_computing.md → "Pipeline de Datos" |
| ¿Qué backends de YOLO hay? | edge_computing.md → "Módulo Detector" |
| ¿Cómo convertir un modelo ONNX a RKNN? | NPU_PIPELINE.md → "Fase 1-2" |
| ¿Cuál es la latencia esperada? | NPU_PIPELINE.md → "Benchmarks" |
| ¿Cómo agregegar una clase prioritaria? | edge_computing.md → "COCO Classes" |
| ¿Dónde ajustar el cooldown de alertas? | edge_computing.md → "AlertEngine" |
| ¿Cómo cambiar el modelo YOLO? | NPU_PIPELINE.md → "Troubleshooting" |
| ¿Qué hace el Orchestrator? | edge_computing.md → "Modos de Operación" |

---

**Generado:** 2026-04-18  
**Versión:** 1.0  
**Estado:** Completo
