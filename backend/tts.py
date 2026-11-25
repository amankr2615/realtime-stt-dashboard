import io
import logging
import edge_tts
from typing import Optional

logger = logging.getLogger(__name__)

# Default neural voices
DEFAULT_VOICE = "hi-IN-MadhurNeural"  # Clear, natural Hindi/Hinglish voice

async def synthesize_speech(text: str, voice: Optional[str] = None, rate: str = "+0%") -> bytes:
    """
    Synthesizes speech using open-source edge-tts.
    Returns raw MP3 audio bytes.
    """
    if not text.strip():
        return b""

    selected_voice = voice or DEFAULT_VOICE
    communicate = edge_tts.Communicate(text=text, voice=selected_voice, rate=rate)
    
    mp3_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buffer.write(chunk["data"])
            
    mp3_buffer.seek(0)
    return mp3_buffer.read()
