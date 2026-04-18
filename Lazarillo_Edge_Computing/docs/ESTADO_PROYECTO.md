# Estado del Proyecto Lazarillo — 2026-04-18

## Resumen
Asistente IA portátil para personas ciegas. Arquitectura: Orange Pi 5 Max (RK3588) + FastAPI + YOLOv8n + Gemini + ElevenLabs TTS.

---

## Lo que está listo ✅

### Modelo YOLO → RKNN
- `yolov8n.rknn` (7.3MB) generado con `docker/rknn-convert/` (arm64 nativo, sin cuantización)
- Ubicación local: `docker/rknn-convert/yolov8n.rknn` y `models/yolov8n.rknn`
- **Pendiente**: copiar al Orange Pi cuando esté encendido:
  ```bash
  scp models/yolov8n.rknn orangepi@10.0.0.174:/home/orangepi/lazarillo/models/
  ```

### Código implementado (178 tests, todos passing)

| Módulo | Descripción | Tests |
|--------|-------------|-------|
| `core/detector.py` | YOLOv8, Mock/ONNX/RKNN con fallback automático | 26 |
| `core/alert_engine.py` | Cooldowns por clase, mensajes en español | 18 |
| `core/audio_manager.py` | Cola de prioridad asyncio, ffplay/aplay | 14 |
| `core/traffic_light.py` | Clasificador HSV rojo/verde | 7 |
| `core/orchestrator.py` | Orquestador de modos, pausa proactivo durante reactivo | 14 |
| `core/voice_capture.py` | VAD por energía RMS, sounddevice | 6 |
| `core/stt.py` | Whisper local (tiny), transcripción en español | 6 |
| `modes/proactive.py` | Loop YOLO→alertas→TTS | 16 |
| `modes/reactive.py` | Pregunta→Gemini→respuesta | 13 |
| `services/stream_reader.py` | Lector MJPEG async con reintentos | 17 |
| `services/elevenlabs_client.py` | TTS con cache LRU (50 entradas) | 15 |
| `services/gemini_client.py` | Gemini 1.5 Flash, rpm_limit=10 | 18 |

### API FastAPI (`main.py`)
| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/health` | GET | Estado del sistema y modo actual |
| `/preview` | GET | Frame JPEG con bounding boxes |
| `/detections` | GET | Detecciones JSON del frame actual |
| `/stream` | GET | MJPEG con detecciones en tiempo real |
| `/trigger/button` | POST | Alterna proactivo/pausado (botón físico) |
| `/mode/{mode}` | POST | Cambia modo: proactive \| paused |
| `/ask` | POST | Responde pregunta sobre la escena |

### Infraestructura
- `docker/rknn-convert/` — conversión ONNX→RKNN en Docker arm64
- `docker-compose.yml` — entorno de desarrollo completo
- `pytest.ini` + `conftest.py` — suite de tests configurada
- `config.py` — pydantic-settings con `.env`

---

## Pendiente 🔧

### Prioridad alta (siguiente sesión)
1. **Encender Orange Pi** y copiar el modelo RKNN:
   ```bash
   scp models/yolov8n.rknn orangepi@10.0.0.174:/home/orangepi/lazarillo/models/
   ```
2. **Validar NPU en producción** — correr `tests/npu_validation.py` en la Pi:
   ```bash
   ssh orangepi@10.0.0.174 "cd /home/orangepi/lazarillo && python3 tests/npu_validation.py"
   ```
3. **Sincronizar código nuevo** al Orange Pi:
   ```bash
   rsync -av --exclude='.git' --exclude='__pycache__' \
     /Users/gabriels/Proyectos/Lazarillo/ \
     orangepi@10.0.0.174:/home/orangepi/lazarillo/
   ```
4. **Instalar dependencias en Pi** (dentro del venv):
   ```bash
   pip install sounddevice openai-whisper
   ```

### Prioridad media
5. `modes/memory.py` + `services/mongodb_client.py` — memoria semántica de lugares
6. Dashboard web (`static/dashboard.html`) — monitor en tiempo real
7. Pre-caché de frases TTS (`data/cached_phrases.json`) — frases comunes pre-generadas
8. `core/voice_capture.py` — integrar en el loop principal (captura mic → STT → reactivo)

### Prioridad baja
9. Bluetooth: firmware `hci0` no carga en kernel 5.x del Orange Pi. Script listo en `scripts/setup_bluetooth.sh`. Solución: **USB headset** para el demo, o actualizar a kernel 6.1.
10. Raspberry Pi Zero W como cámara — ya configurado el stream MJPEG.

---

## Arquitectura actual

```
[Pi Zero W: cámara] ──MJPEG──► [Orange Pi 5 Max: FastAPI]
                                    │
                    ┌───────────────┤
                    │               │
                [ProactiveMode]  [ReactiveMode]
                YOLO→NPU         Gemini 1.5 Flash
                AlertEngine      (Google Cloud)
                    │               │
                    └───────────────┤
                                    ▼
                            [AudioManager]
                            ElevenLabs TTS
                            ffplay/aplay
                                    ▼
                            [Auriculares USB]
```

---

## Comandos clave

```bash
# Desarrollo local
docker-compose up

# Tests
python -m pytest tests/unit/ -v

# Regenerar RKNN (si se necesita)
cd docker/rknn-convert
docker build --platform linux/arm64 -t rknn-convert:latest .
docker run --rm -v $(pwd):/workspace rknn-convert:latest python3 /workspace/convert.py

# En Orange Pi — arrancar servidor
cd /home/orangepi/lazarillo && uvicorn main:app --host 0.0.0.0 --port 8000
```
