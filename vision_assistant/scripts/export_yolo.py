"""
scripts/export_yolo.py — Descarga YOLOv8n y exporta a ONNX.

Ejecutar UNA SOLA VEZ antes de arrancar el sistema:
    python scripts/export_yolo.py

Genera: models/yolov8n.onnx

Notas de portabilidad:
    - Este script usa `ultralytics` (solo en dev/export).
    - En producción (Orange Pi 5) el runtime usa onnxruntime directamente,
      sin depender de ultralytics. Ver inference/yolo_onnx.py.
    - Para convertir el .onnx a RKNN en el Pi, usar rknn-toolkit2:
        from rknn.api import RKNN
        rknn = RKNN()
        rknn.load_onnx(model='models/yolov8n.onnx')
        rknn.build(do_quantization=False)
        rknn.export_rknn('models/yolov8n.rknn')
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Añade el directorio raíz al path para poder importar config_loader
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga YOLOv8n y exporta a ONNX para uso con onnxruntime."
    )
    parser.add_argument(
        "--model",
        default="yolov8n",
        help="Variante de YOLOv8 a descargar (default: yolov8n). "
             "Opciones: yolov8n, yolov8s, yolov8m",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamaño de imagen de entrada del modelo ONNX (default: 640).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "models"),
        help="Directorio donde se guardará el .onnx (default: models/).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="Versión de opset ONNX (default: 17). onnxruntime>=1.16 soporta 17.",
    )
    return parser.parse_args()


def export(model_name: str, imgsz: int, output_dir: Path, opset: int) -> Path:
    """Descarga el modelo PyTorch y lo exporta a ONNX.

    Args:
        model_name: Nombre del modelo ultralytics (ej. "yolov8n").
        imgsz: Resolución cuadrada de entrada.
        output_dir: Carpeta destino para el .onnx.
        opset: Opset ONNX a usar en la exportación.

    Returns:
        Path al archivo .onnx generado.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ERROR: 'ultralytics' no está instalado.\n"
            "Instálalo con:  pip install ultralytics"
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{model_name}.onnx"

    if onnx_path.exists():
        print(f"[INFO] Ya existe: {onnx_path}")
        print("[INFO] Borra el archivo si quieres re-exportar.")
        return onnx_path

    print(f"[INFO] Descargando {model_name}.pt desde Ultralytics Hub...")
    model = YOLO(f"{model_name}.pt")  # descarga automática si no existe

    print(f"[INFO] Exportando a ONNX: imgsz={imgsz}, opset={opset}...")
    # DECISION: simplify=True aplica optimizaciones ONNX básicas (fold constants,
    # eliminar nodos redundantes) sin romper compatibilidad con onnxruntime.
    # dynamic=False fija el batch size a 1, necesario para onnxruntime en el Pi.
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=True,
        dynamic=False,
    )

    # ultralytics guarda el .onnx junto al .pt; lo movemos a models/
    exported_path = Path(exported)
    if exported_path.resolve() != onnx_path.resolve():
        import shutil
        shutil.move(str(exported_path), str(onnx_path))
        print(f"[INFO] Movido a {onnx_path}")

    print(f"[OK] Exportación completada: {onnx_path}")
    print(f"     Tamaño: {onnx_path.stat().st_size / 1_048_576:.1f} MB")
    return onnx_path


def verify(onnx_path: Path, imgsz: int) -> None:
    """Carga el modelo con onnxruntime y hace una inferencia de prueba."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnxruntime no instalado, omitiendo verificación.")
        return

    print("[INFO] Verificando con onnxruntime...")
    providers = []
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
        print("[INFO] Usando CUDAExecutionProvider para verificación")
    providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    outputs = sess.run(None, {input_name: dummy})

    print(f"[OK] Inferencia de prueba exitosa.")
    print(f"     Input:  {input_name} {dummy.shape}")
    print(f"     Output: {[o.shape for o in outputs]}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    print("=" * 55)
    print("  YOLOv8 -> ONNX Export Script")
    print("=" * 55)
    print(f"  Modelo  : {args.model}")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  opset   : {args.opset}")
    print(f"  Destino : {output_dir}")
    print("=" * 55)

    onnx_path = export(args.model, args.imgsz, output_dir, args.opset)
    verify(onnx_path, args.imgsz)

    print()
    print("Siguiente paso:")
    print("  Asegúrate de que config.yaml apunta a este archivo:")
    print(f"    yolo_trigger:")
    print(f"      model_path: \"{onnx_path}\"")


if __name__ == "__main__":
    main()
