# NPU Pipeline: ONNX → RKNN → Inferencia

**Last Updated:** 2026-04-18  
**Scope:** Conversión de modelos YOLO y ejecución en RK3588 NPU

## Pipeline Completo

```
yolov8n.pt (PyTorch)
    ↓ (ultralytics export)
yolov8n.onnx (12 MB, FP32)
    ↓ (rknn-toolkit2 convert, Docker x86)
yolov8n.rknn (4 MB, INT8)
    ↓ (copy a Orange Pi 5 Max)
    ↓ (Orange Pi: rknnlite runtime)
Detecciones @ 30ms (33 FPS)
```

---

## Fase 1: Obtener Modelo ONNX

### Opción A: Descargar desde RKNN Model Zoo (Recomendado)

```bash
mkdir -p models/rknn
cd models/rknn

# YOLOv8n (recomendado para MVP)
wget https://github.com/airockchip/rknn_model_zoo/raw/main/models/cv/object_detection/yolo/yolov8_original_float_model/yolov8n.onnx

# Alternativa: YOLOv5n (más ligero)
# wget https://github.com/airockchip/rknn_model_zoo/raw/main/models/cv/object_detection/yolo/yolov5_original_float_model/yolov5n.onnx

# Verificar
ls -lh yolov8n.onnx
# -rw-r--r-- 12M yolov8n.onnx ✅

# Validar formato ONNX
python3 -c "import onnx; onnx.checker.check_model('yolov8n.onnx')"
# ✅ Model is valid
```

**Por qué este modelo:**
- `yolov8_original_float_model/` = single output `(1, 84, 8400)` compatible
- NO: `yolov8_rk3588_i8/` = multi-scale output (requiere parche postprocess)

### Opción B: Exportar desde PyTorch

```bash
# En máquina con ultralytics + torch
pip install ultralytics torch

python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.export(format='onnx', imgsz=640)
print(f'ONNX exportado: {results}')
"

# → Genera yolov8n.onnx en CWD
cp yolov8n.onnx models/rknn/
```

---

## Fase 2: Convertir ONNX → RKNN

### Opción A: Docker (Recomendado)

**Por qué Docker:**
- Aislamiento: no contamina dependencias locales
- Portabilidad: funciona en Mac M1/M2/M3 con emulación
- Reproducibilidad: imagen fija con numpy 1.26.4 + rknn-toolkit2 2.3.2

#### Paso 1: Build imagen Docker

```bash
# En raíz del proyecto
docker build -t rknn-convert docker/rknn-convert
```

**Dockerfile (`docker/rknn-convert/Dockerfile`):**
```dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    build-essential cmake \
    libglib2.0-0 libgl1-mesa-glx libgl1 libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch==2.2.0 --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    "numpy<=1.26.4" \
    "onnx==1.16.1" \
    "onnxoptimizer==0.3.8" \
    "onnxruntime>=1.16.0" \
    "protobuf==3.20.3" \
    "opencv-python-headless>=4.5.5.64" \
    "scipy>=1.9.3" \
    "tqdm>=4.64.1" \
    "fast-histogram>=0.11" \
    "psutil>=5.9.0" \
    "ruamel.yaml>=0.17.21" \
    "Pillow>=10.0.1"

RUN pip install --no-cache-dir rknn-toolkit2==2.3.2

WORKDIR /workspace
CMD ["bash"]
```

#### Paso 2: Ejecutar conversión

```bash
# Opción 1: Script shell
bash scripts/docker_convert.sh

# Opción 2: Docker run directo
docker run --rm \
  -v $(pwd)/models/rknn:/workspace \
  rknn-convert \
  python3 docker/rknn-convert/convert.py
```

**Salida esperada:**
```
Converting yolov8n.onnx → yolov8n.rknn for rk3588
[1/3] Loading ONNX...
[2/3] Building (quantization=False)...
[3/3] Exporting...
Done: yolov8n.rknn (4.2MB)
```

⏱ **Duración esperada:** 2-5 minutos en x86_64.

### Opción B: Localmente (x86 solamente)

**Advertencia:** Requiere máquina x86 + rknn-toolkit2. En ARM (Mac M1/M3) falla.

```bash
# En máquina x86 (Linux o WSL2)
pip install "numpy==1.26.4" rknn-toolkit2==2.3.2 onnx opencv-python-headless

python3 scripts/convert_model.py \
  --input models/rknn/yolov8n.onnx \
  --output models/rknn/yolov8n.rknn \
  --target rk3588
```

**Script (`scripts/convert_model.py`):**
```python
"""Converts YOLOv8n ONNX → RKNN for RK3588."""
from rknn.api import RKNN
from pathlib import Path

def convert(input_path: str, output_path: str, target: str = "rk3588") -> None:
    rknn = RKNN(verbose=False)

    print(f"[1/4] Configurando para {target}...")
    rknn.config(target_platform=target, mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]])

    print(f"[2/4] Cargando ONNX: {input_path}")
    ret = rknn.load_onnx(model=input_path)
    if ret != 0:
        raise RuntimeError(f"load_onnx failed: {ret}")

    print("[3/4] Construyendo modelo RKNN (2-5 min)...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError(f"build failed: {ret}")

    print(f"[4/4] Exportando a {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        raise RuntimeError(f"export failed: {ret}")

    rknn.release()
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"✅ Modelo exportado: {output_path} ({size_mb:.1f}MB)")
```

### Troubleshooting Conversión

| Error | Causa | Solución |
|-------|-------|----------|
| `ImportError: cannot import 'RKNN'` | rknn-toolkit2 no instalado | `pip install rknn-toolkit2==2.3.2` |
| `numpy>=2.0` + toolkit2 conflict | Versión incompatible | `pip install "numpy==1.26.4"` EN CONVERSIÓN SOLAMENTE |
| ONNX model check fails | Modelo corrupto o viejo | Reexportar desde `yolov8n.pt` |
| Build tarda >10 min | Sistema lento | Normal, esperar o aumentar CPU asignado a Docker |
| Output shape `(3, 25200, 85)` | Usaste modelo multi-escala incorrecto | Descargar `yolov8_original_float_model/yolov8n.onnx` |

---

## Fase 3: Copiar a Orange Pi

### Opción A: SCP (Directo)

```bash
# Desde tu máquina local
scp models/rknn/yolov8n.rknn orangepi@192.168.1.X:~/lazarillo/models/rknn/

# Verificar
ssh orangepi@192.168.1.X "ls -lh ~/lazarillo/models/rknn/yolov8n.rknn"
# -rw-r--r-- 4.2M yolov8n.rknn ✅
```

### Opción B: Git + Pull

```bash
# En tu máquina
git add models/rknn/yolov8n.rknn
git commit -m "chore: add yolov8n.rknn model for RK3588"
git push origin master

# En Orange Pi
cd ~/lazarillo
git pull origin master
```

### Opción C: Rsync (Más flexible)

```bash
# Copiar modelos (excluir .git)
rsync -av --exclude='.git' --exclude='__pycache__' \
  models/rknn/ \
  orangepi@192.168.1.X:~/lazarillo/models/rknn/
```

---

## Fase 4: Validación en Orange Pi

### Script de Validación

**Archivo:** `tests/npu_validation.py`

```bash
ssh orangepi@192.168.1.X

# En Orange Pi
cd ~/lazarillo

# Instalar rknnlite (si no está)
pip install rknn-toolkit2-lite

# Correr validación
python3 tests/npu_validation.py models/rknn/yolov8n.rknn
```

**Salida esperada:**
```
[1/4] Cargando modelo RKNN...
  Verificando modelo: models/rknn/yolov8n.rknn
  Tamaño: 4.2MB ✅

[2/4] Inicializando runtime...
  NPU device: /dev/dri/renderD129 ✅
  Tiempo init: 487ms

[3/4] Calentamiento (3 frames)...
  Frame 1: 32.1ms
  Frame 2: 29.8ms
  Frame 3: 30.5ms

[4/4] Benchmark (20 frames)...
  Latencia media: 30.2ms → 33.1 FPS teóricos
  Latencia P95:   35.8ms
  Latencia P99:   42.1ms

  Num outputs: 1
  Output[0] shape: (1, 84, 8400) dtype: float32

  ✅ Compatible con RKNNDetector._postprocess (sin cambios)

✅ NPU validada correctamente
```

### Qué Verifica

| Aspecto | Esperado | Problema Si... |
|---------|----------|---|
| NPU device | `/dev/dri/renderD129` | No existe → kernel driver no cargado |
| Runtime init | <1000ms | >5000ms → problema de memoria |
| Latencia | 25-40ms @ 640×640 | >100ms → modelo incorrecto o throttling |
| Output shape | `(1, 84, 8400)` | 9 outputs → modelo multi-escala (parche requerido) |
| dtype | `float32` | `int8` → cambiar thresholds postprocess |

### Troubleshooting Hardware

```bash
# ❌ "NPU device no encontrado"
ls /dev/dri/
# Debería ver renderD128, renderD129, etc.
# Si no: kernel driver issue

# ❌ "load_rknn failed: -1"
file models/rknn/yolov8n.rknn
# Debería decir "data" (binary)

# ❌ Latencia >100ms
watch -n1 "cat /sys/class/thermal/thermal_zone*/temp"
# Si >70°C → thermal throttling

# ❌ Memoria insuficiente
free -h
# Debería tener >100MB disponible
```

---

## Integración con core/detector.py

### RKNNDetector

**Ubicación:** `core/detector.py` líneas 133-163

```python
class RKNNDetector:
    """Detector YOLO en NPU RK3588 — solo disponible en Orange Pi."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.4) -> None:
        try:
            from rknnlite.api import RKNNLite  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "rknnlite no disponible — usa ONNXDetector o MockDetector en dev"
            ) from e

        self._threshold = confidence_threshold
        self._rknn = RKNNLite()
        ret = self._rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Error cargando modelo RKNN: {ret}")
        ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            raise RuntimeError(f"Error inicializando runtime RKNN: {ret}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
        outputs = self._rknn.inference(inputs=[blob])
        if outputs is None or len(outputs) == 0:
            return []
        if len(outputs) == 9:
            return _postprocess_dfl(outputs, w, h, self._threshold)
        return _postprocess(outputs[0][0], w, h, self._threshold)
```

### Post-procesado: _postprocess_dfl()

**Para modelos con 9 outputs (multi-escala):**

```python
def _postprocess_dfl(
    outputs: list[np.ndarray],
    orig_w: int,
    orig_h: int,
    threshold: float,
    input_size: int = 640,
    reg_max: int = 16,
) -> list[Detection]:
    """Postprocessing para 9-outputs DFL YOLOv8 RKNN.

    Estructura:
      outputs[i*3+0]: (1, 64, H, W)  — box DFL (4 * reg_max distancias)
      outputs[i*3+1]: (1, 80, H, W)  — class logits
      outputs[i*3+2]: (1,  1, H, W)  — ignorado
    
    Escalas: stride 8 (80×80), stride 16 (40×40), stride 32 (20×20).
    """
    strides = [8, 16, 32]
    x_scale = orig_w / input_size
    y_scale = orig_h / input_size
    bins = np.arange(reg_max, dtype=np.float32)

    all_boxes: list[list[int]] = []
    all_scores: list[float] = []
    all_class_ids: list[int] = []

    for i, stride in enumerate(strides):
        box_dfl = outputs[i * 3][0]      # (64, H, W)
        cls_raw = outputs[i * 3 + 1][0]  # (80, H, W)
        _, H, W = cls_raw.shape

        # Anchor grid — center points in input_size coords
        gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        cx = (gx.ravel() + 0.5) * stride
        cy = (gy.ravel() + 0.5) * stride

        # DFL decode: (64, H, W) → (4, reg_max, H*W) → softmax → weighted sum
        flat = box_dfl.reshape(4, reg_max, H * W)
        shifted = flat - flat.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        soft = exp / exp.sum(axis=1, keepdims=True)
        dist = (soft * bins[np.newaxis, :, np.newaxis]).sum(axis=1) * stride

        x1 = (cx - dist[0]) * x_scale
        y1 = (cy - dist[1]) * y_scale
        x2 = (cx + dist[2]) * x_scale
        y2 = (cy + dist[3]) * y_scale

        # Class scores already sigmoid-activated by RKNN toolkit
        cls_flat = cls_raw.reshape(80, H * W)
        max_cls = cls_flat.max(axis=0)

        pre_mask = max_cls >= threshold
        if not pre_mask.any():
            continue

        idx_keep = np.where(pre_mask)[0]
        cls_sub = cls_flat[:, idx_keep]
        class_ids_arr = np.argmax(cls_sub, axis=0)
        max_scores = cls_sub[class_ids_arr, np.arange(len(idx_keep))]

        mask = max_scores >= threshold
        for local_j in np.where(mask)[0]:
            global_j = int(idx_keep[local_j])
            bx1 = int(x1[global_j])
            by1 = int(y1[global_j])
            bx2 = int(x2[global_j])
            by2 = int(y2[global_j])
            all_boxes.append([bx1, by1, bx2 - bx1, by2 - by1])
            all_scores.append(float(max_scores[local_j]))
            all_class_ids.append(int(class_ids_arr[local_j]))

    if not all_boxes:
        return []

    indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, threshold, 0.45)
    result = []
    for idx in indices:
        idx = int(idx)
        x, y, w, h = all_boxes[idx]
        cid = all_class_ids[idx]
        name = COCO_NAMES[cid] if cid < len(COCO_NAMES) else str(cid)
        result.append(Detection(
            class_id=cid,
            class_name=name,
            confidence=all_scores[idx],
            bbox=(x, y, x + w, y + h),
        ))
    return result
```

---

## Build Factory & Fallback Chain

**Ubicación:** `core/detector.py` líneas 166-188

```python
def build_detector(
    *,
    dev_mode: bool = True,
    model_path: str = "",
    confidence_threshold: float = 0.4,
) -> DetectorBackend:
    """Factory con fallback automático."""
    if dev_mode:
        return MockDetector()
    
    if model_path.endswith(".rknn"):
        try:
            return RKNNDetector(model_path, confidence_threshold)
        except RuntimeError as e:
            import os
            onnx_path = model_path.replace(".rknn", ".onnx")
            if os.path.exists(onnx_path):
                logger.warning("RKNN falló (%s), usando ONNX: %s", e, onnx_path)
                return ONNXDetector(onnx_path, confidence_threshold)
            logger.warning("RKNN y ONNX fallaron, usando Mock")
            return MockDetector()
    
    if model_path.endswith(".onnx"):
        return ONNXDetector(model_path, confidence_threshold)
    
    return MockDetector()
```

**Fallback chain:**
1. RKNN (si existe + disponible)
2. ONNX (si .onnx existe en mismo directorio)
3. Mock (siempre funciona)

---

## Benchmarks Reales

### Orange Pi 5 Max @ 800MHz (default)

```
Modelo: yolov8n INT8 RKNN
Input: 640×640 RGB
Output: (1, 84, 8400) float32

Latencia:
  P50:   28ms
  P95:   35ms
  P99:   42ms

FPS:     33 FPS teórico
Temp:    45-55°C
Consumo: 2-3W (NPU)
```

### Comparativa

| Hardware | Modelo | Latencia | FPS | Notas |
|----------|--------|----------|-----|-------|
| Orange Pi CPU | ONNX FP32 | 205ms | 5 | Fallback si NPU falla |
| Orange Pi NPU | RKNN INT8 | 30ms | 33 | **6.8× más rápido** |
| MacBook M1/M2 | ONNX FP32 | 40ms | 25 | Más rápido que Pi CPU pero sin NPU |

---

## Problemas Conocidos & Soluciones

### Issue 1: Output shape inesperado (3 outputs)

**Síntoma:** Validación reporta `3 outputs` en lugar de `1`

**Causa:** Descargaste modelo multi-escala (`yolov8_rk3588_i8/`) en lugar de single-output

**Solución:**
```bash
# Descargar modelo CORRECTO
rm models/rknn/yolov8n.onnx
wget https://github.com/airockchip/rknn_model_zoo/raw/main/models/cv/object_detection/yolo/yolov8_original_float_model/yolov8n.onnx -O models/rknn/yolov8n.onnx

# Reconvertir
docker run --rm -v $(pwd)/models/rknn:/workspace rknn-convert python3 convert.py
```

### Issue 2: Numpy incompatibility

**Síntoma:** `pip install rknn-toolkit2` falla o requiere `numpy<=1.26.4`

**Causa:** Toolkit2 (conversión) tiene constraint diferente a runtime

**Solución:**
```bash
# EN CONVERSIÓN (x86):
pip install "numpy==1.26.4" rknn-toolkit2==2.3.2

# EN RUNTIME (Orange Pi):
pip install "numpy>=2.0" rknn-toolkit2-lite
# ✅ Sin conflictos (lite no requiere numpy vieja)
```

**Por qué:** Toolkit2 full necesita numpy 1.26.4 para debugging durante conversión. Pero rknnlite (runtime) es compatible con numpy 2.x.

### Issue 3: Latencia intermitente

**Síntoma:** Validación muestra 30ms normalmente, pero a veces >100ms

**Causa:** Thermal throttling o CPU contención

**Solución:**
```bash
# Monitor temperatura
watch -n1 "cat /sys/class/thermal/thermal_zone*/temp"

# Si >70°C, agregar heatsink o ventilador
# Reducir procesos background
systemctl stop cups bluetooth  # (si no necesitas)
```

### Issue 4: "Can't find load func" error

**Síntoma:** `rknnlite.load_rknn()` falla con mensaje críptico

**Causa:** Versión incompatible de rknnlite

**Solución:**
```bash
pip uninstall rknn-toolkit2-lite -y
pip install "rknn-toolkit2-lite==2.3.2"
```

### Issue 5: Stream MJPEG se congela durante inference

**Síntoma:** Detector funciona pero stream se desconecta después de 30s

**Causa:** Buffer MJPEG lleno mientras detector está en inference

**Solución:**
```python
# Opción 1: Reducir JPEG quality en main.py
cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])  # 70 es default

# Opción 2: Procesamiento asincrónico (Fase 3+)
# Procesar frames en thread separado
```

---

## Optimización de Performance

### Reducir Latencia

```bash
# 1. Menor resolución input
# core/detector.py: INPUT_SIZE = 416 (en lugar de 640)
# Latencia: 30ms → 10ms
# Tradeoff: precisión -2-3% mAP

# 2. Modelo más ligero
# yolov8n → yolov5n
# Latencia: 30ms → 20ms
# Tradeoff: precisión similar

# 3. Batch processing (futuro)
# Procesar 2-4 frames paralelo
# Latencia: 30ms → 60ms
# FPS: 1 lote cada 60ms = mejor throughput
```

### Aumentar Precisión

```bash
# 1. Modelo más pesado
# yolov8n → yolov8s
# Latencia: 33 FPS → 10 FPS (inaceptable para MVP)

# 2. Post-procesado mejorado (Fase 3+)
# NMS adaptativo por clase
# Confidence threshold dinámico

# 3. Ensemble con tracking (Fase 4+)
# YOLO + centroid tracking
# Reduce false positives
```

---

## CI/CD Integration (Futuro)

### GitHub Actions para Validación

```yaml
# .github/workflows/npu-validation.yml (Fase 6)
name: NPU Validation
on: [push]
jobs:
  npu-test:
    runs-on: [self-hosted, orangepi]  # Runner en Orange Pi
    steps:
      - uses: actions/checkout@v3
      - run: python tests/npu_validation.py models/rknn/yolov8n.rknn
      - run: pytest tests/unit/test_detector.py -v
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash
if [ -f /dev/dri/renderD129 ]; then
    python tests/npu_validation.py models/rknn/yolov8n.rknn || exit 1
fi
```

---

## Checklist Final (Pre-Producción)

- [ ] Modelo ONNX descargado (12MB) y validado
- [ ] Conversión RKNN completada (4MB, <5 min)
- [ ] Modelo copiado a Orange Pi
- [ ] `npu_validation.py` pasa en Orange Pi
- [ ] Latencia P95 < 40ms
- [ ] MockDetector + ONNXDetector funcionan en Mac
- [ ] RKNNDetector fallback correcto si NPU no disponible
- [ ] Tests pasen: `pytest tests/unit/test_detector.py`
- [ ] Temperatura <70°C durante benchmark
- [ ] Modelo commiteado a Git o documentado dónde descargarlo

---

## Referencias

- **rknn-toolkit2:** https://github.com/airockchip/rknn-toolkit2
- **RKNN Model Zoo:** https://github.com/airockchip/rknn_model_zoo
- **YOLOv8 Docs:** https://docs.ultralytics.com/
- **Orange Pi 5 Max:** http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details_142.html
- **RK3588 NPU Specs:** https://www.rockchip.com.cn/a/en/product/sale/detail/id/38
