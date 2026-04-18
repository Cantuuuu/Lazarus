# Formato de salida: yolov8n.rknn en NPU RK3588

## Modelo y hardware

| Parámetro       | Valor                              |
|-----------------|------------------------------------|
| Modelo          | yolov8n.rknn (7.3 MB)              |
| Toolkit         | RKNN-Toolkit2 2.3.2, sin cuantizar |
| SoC             | RK3588 — Orange Pi 5 Max           |
| Driver NPU      | RKNPU v2, driver 0.9.8             |
| Core activo     | NPU_CORE_0                         |
| Latencia media  | 59.2 ms → 16.9 FPS teóricos        |
| Latencia p95    | 62.3 ms                            |

## Input

```
shape : (1, 3, 640, 640)  — NCHW, float32
rango : [0, 1] normalizados
orden : RGB
```

## Outputs — 9 tensores DFL multi-escala

```
idx  shape             descripción
---  ----------------  ------------------------------------------
 0   (1, 64, 80, 80)   box DFL stride 8   (64 = 4 × reg_max=16)
 1   (1, 80, 80, 80)   class logits stride 8  (80 clases COCO)
 2   (1,  1, 80, 80)   ignorado en postprocess
 3   (1, 64, 40, 40)   box DFL stride 16
 4   (1, 80, 40, 40)   class logits stride 16
 5   (1,  1, 40, 40)   ignorado
 6   (1, 64, 20, 20)   box DFL stride 32
 7   (1, 80, 20, 20)   class logits stride 32
 8   (1,  1, 20, 20)   ignorado
```

Escalas activas: stride 8 → 80×80 (6400 anclas), stride 16 → 40×40 (1600), stride 32 → 20×20 (400).
Total: 8400 anclas candidatas.

## Pipeline de decodificación

```
outputs[0,3,6]  outputs[1,4,7]
(box DFL)       (class logits)
     │                │
     ▼                ▼
  DFL decode       sigmoid
     │                │
  dist ltrb      P_class [0,1]
     │                │
     └────────┬────────┘
              ▼
         filtro score
       (conf × class ≥ thr)
              │
              ▼
      dist → xyxy (píxeles)
              │
              ▼
           NMS (IoU 0.45)
              │
              ▼
        detecciones finales
```

## Decodificación DFL — pseudocódigo

```python
# Para cada escala s ∈ {(outputs[0], stride=8, H=80, W=80),
#                       (outputs[3], stride=16, H=40, W=40),
#                       (outputs[6], stride=32, H=20, W=20)}:

reg_max = 16
proj = arange(0, reg_max)          # [0, 1, ..., 15]

box_raw = outputs[s]               # (1, 64, H, W)
box_raw = box_raw.squeeze(0)       # (64, H, W)
box_raw = reshape(box_raw, (4, reg_max, H*W))  # (4, 16, N)
box_raw = softmax(box_raw, axis=1) # distribucion sobre los 16 bins
dist_ltrb = einsum('j, kjn->kn', proj, box_raw)  # (4, N) distancias en unidades stride

# Convertir distancias a coordenadas xyxy absolutas
cx, cy = meshgrid de centros para (H, W) con stride dado
x1 = (cx - dist_ltrb[0]) * stride
y1 = (cy - dist_ltrb[1]) * stride
x2 = (cx + dist_ltrb[2]) * stride
y2 = (cy + dist_ltrb[3]) * stride

# Scores de clase
cls_raw = outputs[s+1]             # (1, 80, H, W)
scores  = sigmoid(cls_raw)         # (80, H, W)
```

## Implementación

`core/detector.py` → `_postprocess_dfl()`

Umbrales de referencia: confianza 0.25, NMS IoU 0.45.
