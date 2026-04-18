#!/usr/bin/env python3
"""Converts YOLOv8n ONNX → RKNN for RK3588."""
from rknn.api import RKNN

ONNX_MODEL = "yolov8n.onnx"
RKNN_MODEL = "yolov8n.rknn"
TARGET = "rk3588"

print(f"Converting {ONNX_MODEL} → {RKNN_MODEL} for {TARGET}")

rknn = RKNN(verbose=False)

# Standard ultralytics ONNX export expects float32 [0,1] input directly —
# no mean/std normalization needed inside the RKNN runtime.
rknn.config(
    target_platform=TARGET,
)

print("[1/3] Loading ONNX...")
ret = rknn.load_onnx(model=ONNX_MODEL)
if ret != 0:
    print(f"load_onnx failed: {ret}")
    exit(1)

print("[2/3] Building (quantization=False)...")
ret = rknn.build(do_quantization=False)
if ret != 0:
    print(f"build failed: {ret}")
    exit(1)

print("[3/3] Exporting...")
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print(f"export failed: {ret}")
    exit(1)

rknn.release()

import os
size = os.path.getsize(RKNN_MODEL) / 1024 / 1024
print(f"Done: {RKNN_MODEL} ({size:.1f}MB)")
