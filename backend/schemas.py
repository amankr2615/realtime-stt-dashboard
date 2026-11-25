from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

class TranscriptionBase(BaseModel):
    title: Optional[str] = Field(default="Untitled Transcription", max_length=255)
    text: str
    language: Optional[str] = "Hinglish"
    duration_seconds: Optional[float] = 0.0
    metrics: Optional[Dict[str, Any]] = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Transcription text cannot be empty or whitespace")
        return cleaned

class TranscriptionCreate(TranscriptionBase):
    pass

class TranscriptionUpdate(BaseModel):
    title: Optional[str] = None
    text: Optional[str] = None
    language: Optional[str] = None

    @field_validator("text")
    @classmethod
    def update_text_must_not_be_blank(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Transcription text cannot be empty or whitespace")
        return cleaned

class TranscriptionResponse(TranscriptionBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {
        "from_attributes": True
    }

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "hi-IN-MadhurNeural"  # High quality Hindi/Hinglish neural voice
    rate: Optional[str] = "+0%"
