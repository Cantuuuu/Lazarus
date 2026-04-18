import cv2, time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse

app = FastAPI()

def gen_frames():
    cap = cv2.VideoCapture(0)  # Usa tu webcam USB
    if not cap.isOpened():
        print("⚠️ No se pudo abrir /dev/video0")
        return
    print("✅ Webcam activa")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 🎯 Mock de detección para que veas algo YA
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 3)
        cv2.putText(frame, "Lazarillo: Persona (98%)", (w//4, h//4 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Stream MJPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    cap.release()

@app.get("/")
def index():
    return HTMLResponse("""
    <div style="text-align:center; font-family:system-ui; background:#111; color:#fff; padding:20px">
        <h2>🦯 Lazarillo — Demo Visual</h2>
        <img src="/video_feed" width="640" style="border-radius:8px; box-shadow:0 4px 12px rgba(0,255,0,0.3)">
        <p><i>Stream local + Detección Mock | NPU: Standby</i></p>
    </div>
    """)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    print("🌐 Servidor corriendo en http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
