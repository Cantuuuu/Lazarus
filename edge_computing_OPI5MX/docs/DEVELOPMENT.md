# Lazarillo — Guía de Desarrollo

**Last Updated:** 2026-04-18

## Setup Local (Mac/Linux)

### Requisitos previos

```bash
# Python 3.10+ (verificar)
python --version

# pip actualizado
python -m pip install --upgrade pip

# Git
git --version
```

### 1. Clonar repositorio

```bash
git clone https://github.com/tu-usuario/Lazarillo.git
cd Lazarillo
```

### 2. Crear entorno virtual

```bash
# Crear venv
python -m venv .venv

# Activar (macOS/Linux)
source .venv/bin/activate

# Activar (Windows)
.venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
# Desarrollo (incluye pytest, etc)
pip install -r requirements.txt

# Opcional: dev-only (linting, etc — agregado después)
# pip install black flake8 mypy
```

### 4. Verificar instalación

```bash
# Debería salir sin errores
python -c "import fastapi, cv2, onnxruntime; print('✅ Deps OK')"

# Verificar pytest
pytest --version
```

---

## Desarrollo con Docker

### Levantar todo (recomendado)

```bash
# Build + start todos los servicios
docker compose up --build

# Logs
docker compose logs -f lazarillo
docker compose logs -f stream-mock

# Parar
docker compose down
```

### Estructura de servicios

| Servicio | Puerto | Función |
|----------|--------|---------|
| `stream-mock` | 8080 | MJPEG server (simula Pi Zero) |
| `lazarillo` | 8000 | FastAPI principal |

### Desarrollo con hot-reload

```yaml
# docker-compose.yml tiene volumes: .:/app
# → Los cambios en Python se recargan automáticamente
# (Dockerfile tiene --reload activado)
```

### Ejecutar tests en Docker

```bash
# Terminal 1: docker compose up
docker compose up

# Terminal 2: tests
docker compose exec lazarillo pytest tests/unit/ -v

# Con cobertura
docker compose exec lazarillo pytest --cov=core --cov=services
```

---

## Desarrollo local sin Docker

### Opción 1: Mock stream + servidor FastAPI

```bash
# Terminal 1: Mock stream MJPEG
python tests/fixtures/mock_stream.py
# → http://localhost:8080/stream.mjpg

# Terminal 2: Lazarillo
python -m uvicorn main:app --reload
# → http://localhost:8000

# Terminal 3: Tests
pytest tests/unit/ -v
```

### Opción 2: Tests solamente (sin servidor)

```bash
pytest tests/unit/ -v --tb=short

# Con cobertura
pytest --cov=core --cov=services --cov-report=html
# → Abre htmlcov/index.html
```

---

## Estructura de Testing

### Ejecutar tests

```bash
# Todos
pytest

# Unitarios solamente
pytest tests/unit/ -v

# Con markers
pytest -m asyncio                    # Solo async tests
pytest -m "not npu_hardware"        # Excluir hardware tests

# Específico
pytest tests/unit/test_detector.py::TestDetection::test_center_x

# Watch mode (con pytest-watch)
ptw tests/unit/
```

### Coverage

```bash
# Generar reporte
pytest --cov=core --cov=services --cov-report=term-missing

# HTML
pytest --cov=core --cov=services --cov-report=html
open htmlcov/index.html

# Excluir líneas
# ... código ... # pragma: no cover
```

### Fixtures

```python
# tests/fixtures/ contiene:
# - mock_stream.py        (MJPEG server)
# - sample_frame.jpg      (frame de prueba)
```

### Asyncio en pytest

```bash
# pytest.ini tiene asyncio_mode = auto
# → Detecta automáticamente tests async

@pytest.mark.asyncio
async def test_stream_reader():
    sr = StreamReader("http://host/stream")
    frame = await sr.get_frame()
```

---

## Flujo de Trabajo TDD

### Fase 1 (Detector + Stream): YA COMPLETA ✅

Ejemplo de cómo se hizo:

```bash
# 1. RED: tests/unit/test_detector.py define casos
#    Ejecutar: pytest tests/unit/test_detector.py
#    Resultado: FAIL (functions don't exist)

# 2. GREEN: core/detector.py implementa mínimamente
#    Ejecutar: pytest tests/unit/test_detector.py
#    Resultado: PASS

# 3. IMPROVE: Refactor, optimización, comentarios
#    pytest --cov para verificar cobertura >80%
```

### Fase 2 (Proactivo): PRÓXIMO

**Tareas:**

1. **traffic_light.py** (HSV classifier)
   ```python
   # RED: tests/unit/test_traffic_light.py
   def test_classify_red_light():
       frame = load_test_image("red_light.jpg")
       color = classify_traffic_light(frame)
       assert color == "red"
   
   # GREEN: core/traffic_light.py
   def classify_traffic_light(frame: np.ndarray) -> str:
       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       # Buscar rojo/verde/amarillo en rangos HSV
       ...
   ```

2. **elevenlabs_client.py** (TTS async)
   ```python
   # RED: tests/test_elevenlabs_client.py
   @pytest.mark.asyncio
   async def test_synthesize_cached():
       client = ElevenLabsClient(api_key="dev")
       audio1 = await client.speak("Hola")
       audio2 = await client.speak("Hola")
       assert audio1 is audio2  # cached
   
   # GREEN: services/elevenlabs_client.py
   class ElevenLabsClient:
       def __init__(self, api_key: str):
           self._cache = {}  # LRU cache
       
       async def speak(self, text: str) -> bytes:
           if text in self._cache:
               return self._cache[text]
           # API call...
   ```

3. **modes/proactive.py** (Loop principal)
   ```python
   # RED: tests/test_proactive.py
   @pytest.mark.asyncio
   async def test_loop_processes_detections():
       # Mock stream + detector + alert engine
       # Verificar que genera alertas correctamente
   
   # GREEN: modes/proactive.py
   async def run_proactive(stream, detector, engine, tts):
       async for frame in stream.frames():
           detections = detector.detect(frame)
           for det in detections:
               if engine.should_alert(det):
                   msg = engine.direction_message(det)
                   await tts.speak(msg)
                   engine.mark_alerted(det)
   ```

### Checklist de implementación

- [ ] Tests RED fallan
- [ ] Implementar mínimamente
- [ ] Tests GREEN pasan
- [ ] Coverage >80%
- [ ] Refactor/documenta código
- [ ] Docstrings y type hints
- [ ] Tests edge cases
- [ ] Merge a main

---

## Descargar Modelos YOLO

### Opción 1: ONNX (recomendado para dev)

```bash
# Crear directorio
mkdir -p models/rknn

# Descargar yolov8n ONNX desde model zoo
cd models/rknn
wget https://github.com/airockchip/rknn_model_zoo/raw/main/models/cv/object_detection/yolo/yolov8_original_float_model/yolov8n.onnx

# Verificar (debería ser ~12MB)
ls -lh yolov8n.onnx
```

### Opción 2: RKNN (solo si tienes Orange Pi)

```bash
# Si ya tienes yolov8n.onnx, convertir:
python scripts/convert_model.py \
    --input models/rknn/yolov8n.onnx \
    --output models/rknn/yolov8n.rknn

# (Requiere rknn-toolkit2 instalado en x86)
```

### Testing sin modelo real

```python
# core/detector.py build_detector() con dev_mode=True
# → Usa MockDetector, no necesita modelo

config.py:
    env = "development"  # → dev_mode=True
    yolo_model_path = "models/rknn/yolov8n.rknn"
    # Si no existe, fallback a Mock
```

---

## SSH a Orange Pi (Fase 5)

### Configuración inicial (one-time)

```bash
# 1. Descubrir IP (en la red WiFi)
ping orangepi.local
# o: arp-scan -l (si instalaste arp-scan)

# 2. SSH con default credentials
ssh orangepi@<IP>
# password: orangepi (cambiar después!)

# 3. Setup Python
apt update && apt install python3.10 python3-pip python3-venv

# 4. Clonar Lazarillo
git clone https://github.com/tu-usuario/Lazarillo.git
cd Lazarillo

# 5. Setup venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 6. Cambiar contraseña SSH
passwd
```

### Desarrollo remoto (después setup)

```bash
# Desde tu Mac/Linux:

# 1. SSH sin contraseña (agregar tu public key)
ssh-copy-id orangepi@192.168.1.X
ssh orangepi@192.168.1.X

# 2. Correr tests en hardware
ssh orangepi@192.168.1.X "cd Lazarillo && python tests/npu_validation.py models/rknn/yolov8n.rknn"

# 3. Sincronizar código (rsync)
rsync -avz ./core/ ./services/ orangepi@192.168.1.X:Lazarillo/

# 4. Usar como remote en VS Code
# Extensions → Remote - SSH
# Select Host → orangepi@192.168.1.X
# (abre VS Code remoto)
```

### Troubleshooting SSH

```bash
# Revisar que Orange Pi está online
nmap -p 22 192.168.1.1/24

# Permisos SSH
ssh orangepi@<IP> "ls -la .ssh"
# ~/.ssh debe existir con 700 permisos

# Test de conectividad
ssh -vvv orangepi@<IP> "echo OK"
# (muestra debug si falla)
```

---

## Docker para Conversión YOLO

### Convertir ONNX → RKNN en tu Mac

```bash
# docker/rknn-convert/Dockerfile contiene:
# FROM python:3.10-slim
# RUN pip install rknn-toolkit2==2.3.2

# 1. Build imagen Docker
docker build -t rknn-convert docker/rknn-convert

# 2. Correr conversión
docker run --rm -v $(pwd)/models/rknn:/workspace rknn-convert \
    python docker/rknn-convert/convert.py

# Outputs:
# → models/rknn/yolov8n.rknn (~4MB RKNN INT8)
# → Tarda 2-5 minutos
```

### Troubleshooting

```bash
# Si docker run da error de permisos
docker run --rm --user $(id -u):$(id -g) -v ... rknn-convert ...

# Si RKNN build falla
# → Ver logs: ... --volume para debug
# → Possible: onnx model corrupted o versión incompatible
```

---

## Git Workflow

### Branching

```bash
# 1. Feature branch
git checkout -b feature/modo-proactivo

# 2. Develop (write tests first, RED)
# 3. Implement (GREEN)
# 4. Commit
git add core/alert_engine.py tests/unit/test_alert_engine.py
git commit -m "feat: alerting system with per-class cooldowns"

# 5. Push
git push origin feature/modo-proactivo

# 6. Pull Request (GitHub)
# → Revisar tests, coverage, code review
# → Merge a main

# 7. Cleanup
git branch -d feature/modo-proactivo
```

### Commit messages

```
<type>: <descripción corta>

<body opcional con detalles>

Types:
  feat: nueva característica
  fix: corrección de bug
  refactor: cambio sin funcionalidad
  docs: documentación
  test: tests nuevos o fixes
  chore: config, deps, etc
```

### Ejemplo

```
feat: add proactive mode with YOLO detection loop

- Implement modes/proactive.py with async frame processing
- Add ElevenLabs TTS client with LRU cache
- Add AlertEngine cooldown-per-class throttling
- 16 unit tests with 98% coverage
- Latency: <500ms frame-to-alert on RKNN

Fixes: #42
```

---

## Debugging

### Logs

```python
# main.py tiene logging básico
import logging
logger = logging.getLogger(__name__)
logger.info(f"Stream conectado: {settings.pi_zero_stream_url}")

# En Docker
docker compose logs -f lazarillo

# En local
# → logs en stdout (--reload muestra debug)
```

### Breakpoints (pdb)

```python
# En cualquier línea
import pdb; pdb.set_trace()

# En pytest
pytest --pdb tests/unit/test_detector.py::TestDetection
# → Pausa en failures

# Async debugging
import asyncio
asyncio.get_event_loop().set_debug(True)
```

### Performance profiling

```python
# core/detector.py
import time
t0 = time.monotonic()
detections = detector.detect(frame)
elapsed = (time.monotonic() - t0) * 1000
print(f"Inference: {elapsed:.1f}ms")

# Full stack
python -m cProfile -s cumtime main.py
```

### Mock stream debug

```bash
# Verificar que stream-mock sirve frames
curl -v http://localhost:8080/stream.mjpg | hexdump -C | head -20
# Debería ver: FFD8 FFE0 (JPEG header)
```

---

## Endpoints para Testing

### Health check

```bash
curl http://localhost:8000/health
# {"status": "ok", "stream": "connected", "env": "development"}
```

### Detections

```bash
curl -s http://localhost:8000/detections | jq
# {
#   "count": 2,
#   "detections": [
#     {"class": "person", "confidence": 0.92, "direction": "frente"},
#     ...
#   ]
# }
```

### Preview frame

```bash
curl -s http://localhost:8000/preview > frame.jpg
open frame.jpg

# Muestra frame actual con bboxes dibujados
```

### Live stream

```bash
# En vlc o ffmpeg
ffplay http://localhost:8000/stream

# O grabar 5 segundos
ffmpeg -i http://localhost:8000/stream -t 5 output.mp4
```

---

## Environment Variables (.env)

### Desarrollo

```bash
# .env (git-ignored)
GEMINI_API_KEY=dev
ELEVENLABS_API_KEY=dev
ELEVENLABS_VOICE_ID=dev
MONGODB_URI=mongodb://localhost:27017

PI_ZERO_STREAM_URL=http://stream-mock:8080/stream.mjpg
YOLO_MODEL_PATH=models/rknn/yolov8n.rknn
YOLO_CONFIDENCE_THRESHOLD=0.4

WHISPER_MODEL_SIZE=tiny
AUDIO_DEVICE=default

ENV=development
```

### Producción (Orange Pi)

```bash
# Reales (nunca hardcodear)
GEMINI_API_KEY=<real-key>
ELEVENLABS_API_KEY=<real-key>
ELEVENLABS_VOICE_ID=<real-id>

PI_ZERO_STREAM_URL=http://192.168.1.X:8080/stream.mjpg
YOLO_MODEL_PATH=/home/orangepi/lazarillo/models/rknn/yolov8n.rknn

ENV=production
```

---

## Checklist Pre-Commit

Antes de hacer `git push`:

- [ ] Tests pasan: `pytest`
- [ ] Coverage >80%: `pytest --cov=core`
- [ ] Sin hardcoded keys: `grep -r "api_key\|password" core/ services/`
- [ ] Type hints: `mypy core/ services/` (opcional)
- [ ] Docstrings: Todas las functions públicas
- [ ] Código formateado: `black . --check` (opcional)

---

## CI/CD (Futuro — Fase 6)

### GitHub Actions (plan)

```yaml
# .github/workflows/test.yml
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -r requirements.txt
      - run: pytest --cov=core --cov-report=xml
      - uses: codecov/codecov-action@v3
```

### Deploy (plan)

```bash
# scripts/deploy.sh (Fase 5)
#!/bin/bash
HOST=orangepi@192.168.1.X
rsync -av . $HOST:~/lazarillo/
ssh $HOST "cd ~/lazarillo && systemctl restart lazarillo"
```

---

## Recursos

### Documentación oficial

- FastAPI: https://fastapi.tiangolo.com/
- pytest: https://docs.pytest.org/
- YOLO: https://docs.ultralytics.com/
- RKNN: https://github.com/airockchip/rknn-toolkit2/wiki
- Orange Pi 5 Max: http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details_142.html

### Libros/Tutoriales

- Async Python: https://realpython.com/async-io-python/
- OpenCV: https://docs.opencv.org/4.x/

### Comunidades

- Orange Pi Forum: http://www.orangepi.org/orangepibbsen/forum.php
- RKNN Issues: https://github.com/airockchip/rknn-toolkit2/issues
- Ultralytics Issues: https://github.com/ultralytics/yolov8/issues
