import asyncio
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from config import settings
from core.alert_engine import AlertEngine
from core.audio_manager import AudioManager
from core.detector import build_detector, Detection
from core.orchestrator import Orchestrator, Mode
from services.elevenlabs_client import ElevenLabsClient
from services.gemini_client import GeminiClient
from services.stream_reader import StreamReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

stream_reader = StreamReader(settings.pi_zero_stream_url)
detector = build_detector(
    dev_mode=settings.is_dev,
    model_path=settings.yolo_model_path,
    confidence_threshold=settings.yolo_confidence_threshold,
)
alert_engine = AlertEngine()
tts = ElevenLabsClient(
    api_key=settings.elevenlabs_api_key,
    voice_id=settings.elevenlabs_voice_id,
    dev_mode=settings.is_dev,
)
audio = AudioManager()
gemini = GeminiClient(
    api_key=settings.gemini_api_key,
    dev_mode=settings.is_dev,
)
orchestrator = Orchestrator(
    stream_reader, detector, alert_engine, tts, audio, gemini
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await stream_reader.start()
    await audio.start()
    await orchestrator.start()
    logger.info("Lazarillo iniciado — stream: %s", settings.pi_zero_stream_url)
    yield
    await orchestrator.stop()
    await audio.stop()
    await stream_reader.stop()


app = FastAPI(title="Lazarillo", version="0.2.0", lifespan=lifespan)


@app.get("/health")
async def health():
    frame = await stream_reader.get_frame()
    return {
        "status": "ok",
        "stream": "connected" if frame is not None else "waiting",
        "mode": orchestrator.mode,
        "env": settings.env,
    }


@app.get("/preview")
async def preview():
    """Frame actual con bounding boxes dibujados."""
    frame = await stream_reader.get_frame()
    if frame is None:
        return JSONResponse({"error": "sin frame aún"}, status_code=503)

    detections = detector.detect(frame)
    annotated = _draw_detections(frame, detections)
    _, jpeg = cv2.imencode(".jpg", annotated)
    return StreamingResponse(iter([jpeg.tobytes()]), media_type="image/jpeg")


@app.get("/detections")
async def detections():
    """Detecciones del frame actual en JSON."""
    frame = await stream_reader.get_frame()
    if frame is None:
        return JSONResponse({"error": "sin frame aún"}, status_code=503)

    dets = detector.detect(frame)
    return {
        "count": len(dets),
        "detections": [
            {
                "class": d.class_name,
                "confidence": round(d.confidence, 3),
                "direction": d.direction,
                "bbox": d.bbox,
                "priority": d.is_priority,
            }
            for d in dets
        ],
    }


@app.get("/stream")
async def live_stream():
    """Stream MJPEG con detecciones anotadas en tiempo real."""
    async def generate():
        async for frame in stream_reader.frames():
            dets = detector.detect(frame)
            annotated = _draw_detections(frame, dets)
            _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg.tobytes() + b"\r\n"
            )

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/trigger/button")
async def trigger_button():
    """Simula la pulsación del botón físico (alterna proactivo/pausado)."""
    await orchestrator.trigger_button()
    return {"mode": orchestrator.mode}


@app.post("/mode/{mode}")
async def set_mode(mode: str):
    """Cambia el modo directamente: proactive | paused."""
    try:
        target = Mode(mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Modo inválido: {mode!r}")
    await orchestrator.set_mode(target)
    return {"mode": orchestrator.mode}


@app.post("/ask")
async def ask(body: dict):
    """Responde una pregunta sobre la escena actual usando Gemini."""
    question = body.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Campo 'question' requerido")
    answer = await orchestrator.handle_question(question)
    return {"answer": answer}


def _draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        color = (0, 255, 0) if d.is_priority else (200, 200, 200)
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.0%} [{d.direction}]"
        cv2.putText(out, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out
