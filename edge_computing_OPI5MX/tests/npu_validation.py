"""Validación rápida de NPU en Orange Pi RK3588.

Uso:
    source .venv/bin/activate
    python tests/npu_validation.py models/rknn/yolov8n.rknn
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np


def main(model_path: str) -> None:
    print("=== VALIDACIÓN NPU RK3588 (rknnlite) ===\n")

    # RKNPU 0.9.x expone la NPU como DRM device, no como /dev/rknn0
    npu_candidates = ["/dev/dri/renderD129", "/dev/dri/renderD128", "/dev/rknn0"]
    npu_dev = next((p for p in npu_candidates if os.path.exists(p)), None)
    if npu_dev is None:
        print("⚠️  NPU device no encontrado. Probados:", npu_candidates)
        print("   Ejecuta: ls /dev/dri/ && dmesg | grep -i rknpu")
        sys.exit(1)
    print(f"   NPU device: {npu_dev}")

    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print("❌ rknnlite no instalado. Ejecuta: pip install rknn-toolkit2-lite")
        sys.exit(1)

    rknn = RKNNLite()

    print("[1/4] Cargando modelo...")
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f"❌ load_rknn falló: {ret}")
        sys.exit(1)

    print("[2/4] Inicializando runtime NPU (core 0)...")
    t0 = time.time()
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    init_ms = (time.time() - t0) * 1000
    if ret != 0:
        print(f"❌ init_runtime falló: {ret}")
        rknn.release()
        sys.exit(1)
    print(f"   ⏱  Init: {init_ms:.0f} ms")

    # Probar ambos formatos: HWC (uint8) y NCHW (float32 normalizado)
    img_hwc = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    img_nchw = (img_hwc.transpose(2, 0, 1).astype(np.float32) / 255.0)[np.newaxis]

    # Detectar qué formato acepta el modelo
    test_out = rknn.inference(inputs=[img_hwc])
    if test_out is None:
        print("   HWC uint8 → None, probando NCHW float32...")
        test_out = rknn.inference(inputs=[img_nchw])
        if test_out is None:
            print("❌ inference() devuelve None en ambos formatos.")
            print("   Verifica que el modelo fue convertido correctamente.")
            rknn.release()
            sys.exit(1)
        img = img_nchw
        print("   Formato aceptado: NCHW float32 (1,3,640,640)")
    else:
        img = img_hwc
        print("   Formato aceptado: HWC uint8 (640,640,3)")

    print("[3/4] Calentamiento (3 frames descartados)...")
    for _ in range(3):
        rknn.inference(inputs=[img])

    print("[4/4] Benchmark (20 iteraciones)...")
    times: list[float] = []
    for _ in range(20):
        t0 = time.time()
        outputs = rknn.inference(inputs=[img])
        times.append((time.time() - t0) * 1000)

    avg = sum(times) / len(times)
    p95 = sorted(times)[int(len(times) * 0.95)]

    print(f"\n--- Resultados ---")
    print(f"  Latencia media : {avg:.1f} ms  →  {1000/avg:.1f} FPS teóricos")
    print(f"  Latencia p95   : {p95:.1f} ms")
    print(f"  Num outputs    : {len(outputs)}")
    for i, o in enumerate(outputs):
        print(f"  Output[{i}] shape: {o.shape}  dtype: {o.dtype}")

    rknn.release()

    _check_postprocess_compat(outputs)
    print("\n✅ NPU validada correctamente")


def _check_postprocess_compat(outputs: list[np.ndarray]) -> None:
    """Verifica si los outputs son compatibles con _postprocess en core/detector.py."""
    print("\n--- Compatibilidad con RKNNDetector._postprocess ---")

    if len(outputs) == 1:
        o = outputs[0]
        # Esperado: (1, 84, 8400) o (84, 8400)
        shape = o.shape
        if shape[-2:] == (84, 8400) or (len(shape) == 3 and shape[1] == 84):
            print("  ✅ Output único (84, 8400) — compatible sin cambios")
        else:
            print(f"  ⚠️  Output único shape inesperado: {shape}")
            print("     → Necesita parche en _postprocess. Reporta este shape.")
    elif len(outputs) == 3:
        print("  ⚠️  3 outputs (formato multi-escala del model zoo)")
        print("     → _postprocess actual NO es compatible.")
        print("     → Ejecuta con --report para ver shapes detallados y generar parche.")
        for i, o in enumerate(outputs):
            print(f"     Output[{i}]: {o.shape}")
    else:
        print(f"  ⚠️  {len(outputs)} outputs — formato desconocido. Reporta los shapes.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "models/rknn/yolov8n.rknn"
    if not os.path.exists(path):
        print(f"❌ Modelo no encontrado: {path}")
        print("   Descarga: wget -O models/rknn/yolov8n.rknn <url-del-model-zoo>")
        sys.exit(1)
    main(path)
