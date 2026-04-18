#!/bin/bash
# Convierte yolov8n.onnx → yolov8n.rknn usando rknn-toolkit2 en Docker (x86_64)
# Uso: bash scripts/docker_convert.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INPUT="$PROJECT_ROOT/models/rknn/yolov8n.onnx"
OUTPUT="$PROJECT_ROOT/models/rknn/yolov8n.rknn"

if [ ! -f "$INPUT" ]; then
    echo "❌ No se encuentra: $INPUT"
    exit 1
fi

echo "🐳 Corriendo conversión RKNN en Docker x86_64..."
docker run --rm --platform linux/amd64 \
    -v "$PROJECT_ROOT/models/rknn:/models" \
    -v "$SCRIPT_DIR:/scripts" \
    python:3.10-slim \
    bash -c "
        pip install --quiet rknn-toolkit2 2>&1 | tail -3
        python /scripts/convert_model.py \
            --input /models/yolov8n.onnx \
            --output /models/yolov8n.rknn \
            --target rk3588
    "

echo "✅ Conversión completada: $OUTPUT"
ls -lh "$OUTPUT"
