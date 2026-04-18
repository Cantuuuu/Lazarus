# Plan de Implementación: Lazarillo

## Resumen

Lazarillo es un asistente wearable de visión artificial para personas ciegas, compuesto por una Raspberry Pi Zero W (streaming de cámara), una Orange Pi 5 Max (procesamiento central con NPU), y auriculares Bluetooth. El sistema opera en tres modos: proactivo (alertas automáticas por YOLO), reactivo (preguntas por voz respondidas con Gemini), y memoria (grabación de eventos con reportes). Este plan prioriza un MVP funcional para demo de hackathon.

---

## Orden recomendado para hackathon

```
Prioridad CRÍTICA — demo mínima viable (~9 horas):
  Fase 0: Infraestructura dev         ~2h
  Fase 1: Stream + Detector YOLO      ~3h
  Fase 2: Modo Proactivo              ~4h

Prioridad ALTA — demo completa (~7 horas más):
  Fase 3: Modo Reactivo               ~4h
  Fase 6.1: Cache TTS                 ~1h
  Fase 6.2: Dashboard                 ~2h

Prioridad MEDIA — diferenciación (~4 horas más):
  Fase 4: Modo Memoria                ~4h

Prioridad BAJA — solo si hay tiempo (~6 horas más):
  Fase 5: Integración hardware        ~4h
  Fase 6.3-6.4: Pulido demo           ~2h
```

---

## Fase 0: Infraestructura de desarrollo

**Objetivo:** Poder iterar rápidamente en Docker sin hardware real.

| # | Tarea | Archivo | Riesgo |
|---|-------|---------|--------|
| 0.1 | Mock del stream MJPEG (servidor Flask con video en loop) | `tests/fixtures/mock_stream.py` | Bajo |
| 0.2 | Imágenes de prueba con objetos COCO | `tests/fixtures/sample_frame.jpg` | Bajo |
| 0.3 | Extender config.py con campos YOLO, audio, dev_mode | `config.py` | Bajo |
| 0.4 | Crear `__init__.py` en todos los paquetes | `core/`, `modes/`, `services/`, `tests/` | Bajo |
| 0.5 | Agregar `pydantic-settings` a requirements | `requirements.txt` | Bajo |

**Criterios de aceptación:**
- [ ] `docker compose --profile dev up` levanta lazarillo + stream-mock
- [ ] Mock stream sirve frames MJPEG accesibles desde el contenedor
- [ ] `pytest` corre sin errores

---

## Fase 1: Lector de stream + Detector YOLO

**Objetivo:** Leer frames del stream y ejecutar detección de objetos.

| # | Tarea | Archivo | Riesgo |
|---|-------|---------|--------|
| 1.1 | Clase `StreamReader` async con reconexión automática | `services/stream_reader.py` | Medio |
| 1.2 | Tests del stream reader | `tests/test_stream_reader.py` | Bajo |
| 1.3 | Wrapper `Detector` con `RKNNDetector` + `MockDetector` | `core/detector.py` | Alto |
| 1.4 | Post-procesado YOLO: NMS, thresholding, mapeo COCO | `core/yolo_postprocess.py` | Medio |
| 1.5 | Tests del detector y post-procesado | `tests/test_detector.py` | Bajo |

**Criterios de aceptación:**
- [ ] StreamReader lee frames del mock stream sin errores
- [ ] Detector retorna lista de `Detection` con MockDetector
- [ ] NMS filtra correctamente bounding boxes solapados
- [ ] Cobertura >80% en estos módulos

---

## Fase 2: Modo Proactivo

**Objetivo:** Sistema que detecta objetos y genera alertas de voz automáticamente.

| # | Tarea | Archivo | Riesgo |
|---|-------|---------|--------|
| 2.1 | `AlertEngine` con cooldowns por clase y lógica de dirección | `core/alert_engine.py` | Bajo |
| 2.2 | Clasificador de semáforos HSV (rojo/verde) | `core/traffic_light.py` | Medio |
| 2.3 | Cliente ElevenLabs async con cache LRU y mock dev | `services/elevenlabs_client.py` | Medio |
| 2.4 | `AudioManager` con cola de prioridad y control de interrupción | `core/audio_manager.py` | Medio |
| 2.5 | Cliente Gemini async (Flash + Pro) con rate limiting | `services/gemini_client.py` | Medio |
| 2.6 | Loop proactivo: frame → detect → alert → speak | `modes/proactive.py` | Medio |
| 2.7 | Tests: alert engine, HSV, loop proactivo mockeado | `tests/test_alert_engine.py`, `tests/test_traffic_light.py`, `tests/test_proactive.py` | Bajo |

**Cooldowns por clase:**
```python
ALERT_COOLDOWNS = {
    "car": 5, "person": 8, "traffic_light": 3,
    "dog": 10, "stairs": 3, "bicycle": 5
}
```

**Criterios de aceptación:**
- [ ] Alert engine respeta cooldowns
- [ ] Clasificador HSV distingue semáforo rojo de verde
- [ ] Loop procesa ≥5 fps con MockDetector
- [ ] Demo en Docker: stream → detecciones → alertas de voz

---

## Fase 3: Modo Reactivo

**Objetivo:** El usuario pregunta algo y recibe respuesta contextual.

| # | Tarea | Archivo | Riesgo |
|---|-------|---------|--------|
| 3.1 | `VoiceCapture` con VAD (webrtcvad) y mock dev | `core/voice_capture.py` | Alto |
| 3.2 | `SpeechToText` con faster-whisper tiny, idioma español | `core/stt.py` | Medio |
| 3.3 | Pipeline reactivo: audio → Whisper → Gemini → TTS | `modes/reactive.py` | Bajo |
| 3.4 | `Orchestrator`: gestión de modos, pausa proactivo durante pregunta | `core/orchestrator.py` | Alto |
| 3.5 | Tests: STT con audio pregrabado, pipeline reactivo mockeado | `tests/test_stt.py`, `tests/test_reactive.py`, `tests/test_orchestrator.py` | Bajo |

**Criterios de aceptación:**
- [ ] VAD detecta inicio/fin de habla en audio de prueba
- [ ] Whisper transcribe "¿Qué hay frente a mí?" correctamente
- [ ] Pipeline end-to-end <3 segundos en dev
- [ ] Orquestador pausa alertas durante pregunta y las reanuda

---

## Fase 4: Modo Memoria

**Objetivo:** Grabar sesiones y generar reportes consultables.

| # | Tarea | Archivo | Riesgo |
|---|-------|---------|--------|
| 4.1 | `MongoDBClient` async: sesiones, reportes, búsqueda | `services/mongodb_client.py` | Bajo |
| 4.2 | `MemoryMode`: grabación continua + keyframes + Gemini Pro → reporte | `modes/memory.py` | Alto |
| 4.3 | Tests: MongoDB mockeado, flujo de grabación | `tests/test_mongodb_client.py`, `tests/test_memory.py` | Bajo |

**Criterios de aceptación:**
- [ ] Sesión de 2 minutos genera reporte Markdown coherente
- [ ] Keyframes capturados cada 30s y ante cambios de detección
- [ ] Reporte guardado en MongoDB con metadata correcta
- [ ] Consulta por texto retorna reportes relevantes

---

## Fase 5: Integración hardware

**Objetivo:** Pasar de Docker en Mac a hardware real funcionando.

| # | Tarea | Archivo | Riesgo |
|---|-------|---------|--------|
| 5.1 | Configurar SSH, WiFi, hostname en Pi Zero W y Orange Pi | — (config de sistema) | Medio |
| 5.2 | Script de streaming Pi Zero W con picamera2 | `pi_zero/stream.py` | Bajo |
| 5.3 | Convertir modelo YOLO a RKNN INT8 | `scripts/convert_model.py` | Alto |
| 5.4 | Configurar BlueZ + PipeWire para auriculares BT | `scripts/setup_bluetooth.sh` | Alto |
| 5.5 | Script de deploy con rsync + systemd service | `scripts/deploy.sh` | Medio |
| 5.6 | Prueba end-to-end en hardware | — (manual) | Alto |

**Criterios de aceptación:**
- [ ] Pi Zero W transmite MJPEG estable a 10fps
- [ ] Orange Pi ejecuta YOLO en NPU a ≥15fps
- [ ] Audio Bluetooth funciona (entrada y salida)
- [ ] Sistema arranca autónomamente al encender

---

## Fase 6: Pulido para demo

**Objetivo:** Demo robusta e impresionante.

| # | Tarea | Archivo | Riesgo |
|---|-------|---------|--------|
| 6.1 | Pre-cachear 20-30 frases TTS comunes al arranque | `data/cached_phrases.json` | Bajo |
| 6.2 | Dashboard web: frame con bboxes, detecciones, estado del modo | `static/dashboard.html` | Bajo |
| 6.3 | Fallbacks en todos los puntos de falla (TTS local, alertas sin Gemini) | todos los módulos | Medio |
| 6.4 | Script de arranque de demo con verificación de conectividad | `scripts/demo.sh` | Bajo |

**Criterios de aceptación:**
- [ ] Demo de 5 minutos sin crashes
- [ ] Latencia de alerta <1 segundo con frases cacheadas
- [ ] Dashboard muestra detecciones en tiempo real
- [ ] Sistema se recupera automáticamente si pierde el stream

---

## Riesgos y mitigaciones

| Riesgo | Probabilidad | Mitigación |
|--------|-------------|------------|
| Bluetooth audio inestable en Linux ARM | Alta | Fallback a altavoz USB o jack 3.5mm; script de reinicio BlueZ |
| Conversión YOLO→RKNN falla o da mala precisión | Media | Fallback a modelo ONNX en CPU; revisar rknn-model-zoo |
| Latencia total >3 segundos | Media | Cache TTS, usar solo Gemini Flash en proactivo, reducir resolución |
| APIs externas fallan en demo | Media | pyttsx3 como TTS local; alertas solo con clase YOLO |
| Stream MJPEG se desconecta | Media | Reconexión automática con backoff; video pregrabado como fallback |
| RAM agotada en Modo Memoria | Baja | Comprimir keyframes a JPEG, límite de 100 keyframes por sesión |

---

## Criterios de éxito globales

- [ ] Sistema detecta objetos y genera alertas de voz en <2 segundos
- [ ] Usuario puede hacer preguntas por voz y recibir respuestas contextuales
- [ ] Demo de 5 minutos sin interrupciones
- [ ] Dashboard muestra detecciones en tiempo real para los jueces
- [ ] Código en módulos pequeños con tests (>80% cobertura en `core/`)
