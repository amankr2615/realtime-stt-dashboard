import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Database: Defaults to SQLite for local development; override with Postgres in prod
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./transcriptions.db")

    # Faster-Whisper Model Settings
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "tiny")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    WHISPER_CPU_THREADS: int = int(os.getenv("WHISPER_CPU_THREADS", "2"))
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")

    # Audio Pipeline Settings
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    BYTES_PER_SAMPLE: int = 2  # 16-bit PCM = 2 bytes
    BATCH_DURATION_MS: int = 100
    BATCH_SIZE_BYTES: int = int(SAMPLE_RATE * (BATCH_DURATION_MS / 1000.0) * BYTES_PER_SAMPLE)  # 3200 bytes

    # Streaming Window Settings
    INFERENCE_INTERVAL_SEC: float = 0.8
    MAX_BUFFER_SECONDS: float = 5.0
    CONFIRMATION_OVERLAP_SECONDS: float = 0.4
    MIN_AUDIO_DURATION_SEC: float = 0.5

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

settings = Settings()
