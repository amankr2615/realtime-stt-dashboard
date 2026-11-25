from sqlalchemy import Column, Integer, String, Text, Float, JSON, DateTime
from sqlalchemy.sql import func
from backend.database import Base

class Transcription(Base):
    __tablename__ = "transcriptions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(255), nullable=False, default="Untitled Transcription")
    text = Column(Text, nullable=False)
    language = Column(String(50), default="Hinglish")
    duration_seconds = Column(Float, default=0.0)
    metrics = Column(JSON, nullable=True)  # { "avg_latency_ms": 150, "rtf": 0.32, "switch_count": 4 }
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
