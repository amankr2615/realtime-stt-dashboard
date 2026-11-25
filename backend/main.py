import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import get_db, init_db
from backend.models import Transcription
from backend.schemas import TranscriptionCreate, TranscriptionUpdate, TranscriptionResponse, TTSRequest
from backend.audio import BoundedAudioBuffer, calculate_rms_energy
from backend.transcriber import WhisperEngine, StreamingTranscriber
from backend.tts import synthesize_speech

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")

# Limit concurrent Whisper inferences globally to bound CPU/RAM on small containers
GLOBAL_INFERENCE_SEMAPHORE = asyncio.Semaphore(1)
WS_SEND_TIMEOUT_SEC = 2.0


def get_session_id(x_session_id: Optional[str] = Header(default="default", alias="X-Session-ID")) -> str:
    """Extracts and normalizes anonymous client session ownership ID."""
    cleaned = (x_session_id or "default").strip()
    return cleaned[:64] if cleaned else "default"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Database and Pre-warm Model
    logger.info("Initializing database...")
    init_db()
    logger.info("Pre-warming Faster-Whisper model in background...")
    # Load model synchronously once on startup
    await asyncio.to_thread(WhisperEngine.get_instance)
    logger.info("Application ready to accept connections.")
    yield
    logger.info("Application shutdown.")

app = FastAPI(
    title="Real-Time Hinglish STT & Dashboard",
    description="Multilingual Speech-to-Text streaming service with PostgreSQL CRUD and TTS",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# REST API: CRUD Operations for Transcriptions
# -------------------------------------------------------------

@app.get("/api/transcriptions", response_model=List[TranscriptionResponse])
def get_transcriptions(
    search: Optional[str] = Query(None, description="Search by title or text"),
    limit: int = Query(50, ge=1, le=200, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    session_id: str = Depends(get_session_id),
    db: Session = Depends(get_db)
):
    query = db.query(Transcription).filter(Transcription.session_id == session_id)
    if search:
        search_filter = f"%{search}%"
        query = query.filter(
            (Transcription.title.ilike(search_filter)) | (Transcription.text.ilike(search_filter))
        )
    return query.order_by(Transcription.created_at.desc()).offset(offset).limit(limit).all()

@app.get("/api/transcriptions/{transcription_id}", response_model=TranscriptionResponse)
def get_transcription(
    transcription_id: int,
    session_id: str = Depends(get_session_id),
    db: Session = Depends(get_db)
):
    item = db.query(Transcription).filter(
        Transcription.id == transcription_id,
        Transcription.session_id == session_id
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Transcription not found")
    return item

@app.post("/api/transcriptions", response_model=TranscriptionResponse, status_code=status.HTTP_201_CREATED)
def create_transcription(
    data: TranscriptionCreate,
    session_id: str = Depends(get_session_id),
    db: Session = Depends(get_db)
):
    item = Transcription(
        session_id=session_id,
        title=data.title or "Untitled Transcription",
        text=data.text.strip(),
        language=data.language or "Hinglish",
        duration_seconds=data.duration_seconds or 0.0,
        metrics=data.metrics or {}
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    logger.info("Saved transcription id=%d session=%s (%d chars)", item.id, session_id, len(item.text))
    return item

@app.put("/api/transcriptions/{transcription_id}", response_model=TranscriptionResponse)
def update_transcription(
    transcription_id: int,
    data: TranscriptionUpdate,
    session_id: str = Depends(get_session_id),
    db: Session = Depends(get_db)
):
    item = db.query(Transcription).filter(
        Transcription.id == transcription_id,
        Transcription.session_id == session_id
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Transcription not found")
    
    if data.title is not None:
        item.title = data.title.strip()
    if data.text is not None:
        item.text = data.text.strip()
    if data.language is not None:
        item.language = data.language.strip()

    db.commit()
    db.refresh(item)
    logger.info("Updated transcription id=%d", item.id)
    return item

@app.delete("/api/transcriptions/{transcription_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_transcription(
    transcription_id: int,
    session_id: str = Depends(get_session_id),
    db: Session = Depends(get_db)
):
    item = db.query(Transcription).filter(
        Transcription.id == transcription_id,
        Transcription.session_id == session_id
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Transcription not found")
    db.delete(item)
    db.commit()
    logger.info("Deleted transcription id=%d", transcription_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

# -------------------------------------------------------------
# REST API: Open-Source Text-to-Speech (TTS)
# -------------------------------------------------------------

@app.post("/api/tts")
async def generate_tts(payload: TTSRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        mp3_bytes = await synthesize_speech(payload.text, voice=payload.voice, rate=payload.rate)
        return Response(content=mp3_bytes, media_type="audio/mpeg")
    except Exception as e:
        logger.error("TTS generation error: %s", e)
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")

# -------------------------------------------------------------
# WebSocket: Real-Time Audio Streaming & Transcription
# -------------------------------------------------------------

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected.")

    audio_buffer = BoundedAudioBuffer(sample_rate=settings.SAMPLE_RATE, max_seconds=settings.MAX_BUFFER_SECONDS)
    streaming_state = StreamingTranscriber()
    whisper_engine = WhisperEngine.get_instance()

    is_running = True
    active_inference_lock = asyncio.Lock()
    language_mode = "hi"  # Default: Hinglish with Devanagari Hindi + English loanwords
    last_reported_bytes = 0

    async def safe_send_json(payload: dict) -> bool:
        """Sends JSON over WebSocket with timeout to prevent backpressure hangs."""
        try:
            await asyncio.wait_for(websocket.send_json(payload), timeout=WS_SEND_TIMEOUT_SEC)
            return True
        except (asyncio.TimeoutError, WebSocketDisconnect, RuntimeError):
            return False

    async def inference_worker():
        """
        Fires every INFERENCE_INTERVAL_SEC.
        Skips inference when:
          • Buffer is too short (< MIN_AUDIO_DURATION_SEC)
          • Energy VAD reports silence (prevents hallucination on background noise)
          • A previous inference pass is still running (backpressure guard)
        """
        nonlocal last_reported_bytes
        while is_running:
            await asyncio.sleep(settings.INFERENCE_INTERVAL_SEC)

            current_sec = audio_buffer.duration_seconds()
            if current_sec < settings.MIN_AUDIO_DURATION_SEC:
                continue

            # ── Energy VAD gate ─────────────────────────────────────────────
            if not audio_buffer.has_enough_speech():
                logger.debug("VAD: silence detected, skipping inference pass")
                continue

            if active_inference_lock.locked():
                continue

            async with active_inference_lock:
                audio_float = audio_buffer.get_audio_float32()
                if len(audio_float) == 0:
                    continue

                total_bytes = audio_buffer.total_bytes_received
                new_bytes = max(0, total_bytes - last_reported_bytes)
                last_reported_bytes = total_bytes
                new_audio_sec = new_bytes / audio_buffer.bytes_per_second

                async with GLOBAL_INFERENCE_SEMAPHORE:
                    hypothesis, infer_time = await asyncio.to_thread(
                        whisper_engine.transcribe,
                        audio_float,
                        language_mode=language_mode,
                    )

                if not hypothesis.strip():
                    streaming_state.record_empty_inference(infer_time, new_audio_sec)
                    continue

                result = streaming_state.process_hypothesis(
                    hypothesis_text=hypothesis,
                    audio_duration_sec=current_sec,
                    inference_time_sec=infer_time,
                    new_audio_sec=new_audio_sec,
                )

                # Advance audio window after confirmed words
                if result["words_committed_count"] > 0:
                    trim_duration = max(0.0, current_sec - 1.0)
                    if trim_duration > 0.3:
                        audio_buffer.trim_front(trim_duration, keep_overlap_sec=0.2)

                sent = await safe_send_json({
                    "type": "partial",
                    "committed":   result["committed_text"],
                    "provisional": result["provisional_text"],
                    "full_text":   result["full_text"],
                    "metrics":     result["metrics"],
                })
                if not sent:
                    break

    worker_task = asyncio.create_task(inference_worker())

    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                audio_buffer.append(message["bytes"])
            elif "text" in message and message["text"]:
                try:
                    data   = json.loads(message["text"])
                    action = data.get("action")
                    if action == "finalize":
                        # Transcribe any remaining audio in the buffer before finalizing
                        rem_sec = audio_buffer.duration_seconds()
                        if rem_sec >= 0.3 and audio_buffer.has_enough_speech():
                            async with active_inference_lock:
                                audio_float = audio_buffer.get_audio_float32()
                                if len(audio_float) > 0:
                                    total_bytes = audio_buffer.total_bytes_received
                                    new_bytes = max(0, total_bytes - last_reported_bytes)
                                    last_reported_bytes = total_bytes
                                    new_audio_sec = new_bytes / audio_buffer.bytes_per_second
                                    async with GLOBAL_INFERENCE_SEMAPHORE:
                                        hyp, infer_t = await asyncio.to_thread(
                                            whisper_engine.transcribe,
                                            audio_float,
                                            language_mode=language_mode,
                                        )
                                    if hyp.strip():
                                        streaming_state.process_hypothesis(
                                            hyp, rem_sec, infer_t, new_audio_sec=new_audio_sec
                                        )
                                    else:
                                        streaming_state.record_empty_inference(infer_t, new_audio_sec)
                        final_res = streaming_state.finalize()
                        await safe_send_json({
                            "type": "final",
                            "text":    final_res["final_text"],
                            "metrics": final_res["metrics"],
                        })
                        audio_buffer.clear()
                        last_reported_bytes = 0
                    elif action == "set_language":
                        language_mode = data.get("language", "hi")
                        logger.info("Language mode → %s", language_mode)
                    elif action == "reset":
                        audio_buffer.clear()
                        streaming_state.reset()
                        last_reported_bytes = 0
                except json.JSONDecodeError:
                    pass
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected gracefully.")
    except Exception as e:
        logger.warning("WebSocket error: %s", e)
    finally:
        is_running = False
        worker_task.cancel()
        logger.info("WebSocket session cleaned up.")

# -------------------------------------------------------------
# Static Files & Frontend Serving
# -------------------------------------------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")
