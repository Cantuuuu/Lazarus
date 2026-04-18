from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # APIs externas
    gemini_api_key: str = "dev"
    elevenlabs_api_key: str = "dev"
    elevenlabs_voice_id: str = "dev"
    mongodb_uri: str = "mongodb://localhost:27017"

    # Hardware
    pi_zero_stream_url: str = "http://stream-mock:8080/stream.mjpg"

    # YOLO
    yolo_model_path: str = "models/yolo11n.rknn"
    yolo_confidence_threshold: float = 0.4

    # Alertas
    alert_cooldown_seconds: int = 5

    # Audio / STT
    whisper_model_size: str = "tiny"
    audio_device: str = "default"

    # Entorno
    env: str = "development"

    @property
    def is_dev(self) -> bool:
        return self.env == "development"


settings = Settings()
