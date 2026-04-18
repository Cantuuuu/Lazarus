"""Convierte YOLOv8 ONNX → RKNN para RK3588 (Orange Pi 5 Max).

Uso (en x86 con rknn-toolkit2):
    pip install rknn-toolkit2
    python scripts/convert_model.py --input models/rknn/yolov8n.onnx --output models/rknn/yolov8n.rknn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def convert(input_path: str, output_path: str, target: str = "rk3588") -> None:
    try:
        from rknn.api import RKNN
    except ImportError:
        print("❌ rknn-toolkit2 no instalado. Ejecuta en x86:")
        print("   pip install rknn-toolkit2")
        sys.exit(1)

    rknn = RKNN(verbose=False)

    print(f"[1/4] Configurando para {target}...")
    ret = rknn.config(
        target_platform=target,
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
    )
    if ret != 0:
        print(f"❌ config falló: {ret}")
        sys.exit(1)

    print(f"[2/4] Cargando ONNX: {input_path}")
    ret = rknn.load_onnx(model=input_path)
    if ret != 0:
        print(f"❌ load_onnx falló: {ret}")
        sys.exit(1)

    print("[3/4] Construyendo modelo RKNN (puede tardar 2-5 min)...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"❌ build falló: {ret}")
        sys.exit(1)

    print(f"[4/4] Exportando a {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"❌ export falló: {ret}")
        sys.exit(1)

    rknn.release()
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"✅ Modelo exportado: {output_path} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="models/rknn/yolov8n.onnx")
    parser.add_argument("--output", default="models/rknn/yolov8n.rknn")
    parser.add_argument("--target", default="rk3588")
    args = parser.parse_args()
    convert(args.input, args.output, args.target)
