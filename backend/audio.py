import math
import struct
import threading
import numpy as np
from typing import Optional
from backend.config import settings

# ── Silence / noise detection thresholds ──────────────────────────────────────
# RMS of raw int16 samples.  Typical ambient noise on a laptop mic ≈ 100‑300.
# Quiet room speech starts at ~1 000.  Comfortable speech ≈ 3 000‑10 000.
SILENCE_RMS_THRESHOLD = 500      # below this → treat as silence, skip inference
MIN_SPEECH_RATIO      = 0.15     # at least 15 % of the buffer must be "voiced"

def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw 16‑bit signed PCM mono bytes → float32 in [‑1.0, 1.0]."""
    if not pcm_bytes:
        return np.array([], dtype=np.float32)
    int16_arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    return int16_arr.astype(np.float32) / 32768.0


def calculate_rms_energy(pcm_bytes: bytes) -> float:
    """RMS energy of a raw PCM16 chunk (higher → louder)."""
    if len(pcm_bytes) < 2:
        return 0.0
    int16_arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    if len(int16_arr) == 0:
        return 0.0
    return math.sqrt(float(np.mean(int16_arr.astype(np.float64) ** 2)))


def has_speech(audio_float: np.ndarray, threshold: float = SILENCE_RMS_THRESHOLD / 32768.0) -> bool:
    """
    Fast energy‑based VAD.
    Returns True when the audio array contains enough voiced energy to be worth
    sending to Whisper.  Works on float32 samples in [‑1.0, 1.0].
    """
    if len(audio_float) == 0:
        return False

    # Split into 20 ms frames and compute per‑frame RMS
    frame_size = int(settings.SAMPLE_RATE * 0.02)   # 320 samples @ 16 kHz
    if frame_size == 0:
        return False

    n_frames = len(audio_float) // frame_size
    if n_frames == 0:
        return False

    frames = audio_float[: n_frames * frame_size].reshape(n_frames, frame_size)
    rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=1))
    voiced_ratio  = float(np.mean(rms_per_frame > threshold))

    return voiced_ratio >= MIN_SPEECH_RATIO


class BoundedAudioBuffer:
    """
    Thread‑safe, bounded rolling audio buffer for 16 kHz mono PCM16 streaming.

    Key design choices
    ──────────────────
    • Fixed upper‑bound (max_seconds) prevents unbounded memory growth.
    • trim_front() removes confirmed audio while keeping an acoustic‑context
      overlap so the next Whisper pass has proper phoneme context.
    • has_enough_speech() wraps the energy VAD so the WebSocket handler can
      gate inference without importing audio utilities separately.
    """

    def __init__(self, sample_rate: int = 16_000, max_seconds: float = 6.0):
        self.sample_rate     = sample_rate
        self.bytes_per_second = sample_rate * 2        # 16‑bit mono = 2 bytes/sample
        self.max_bytes       = int(max_seconds * self.bytes_per_second)
        self.buffer          = bytearray()
        self.lock            = threading.Lock()
        self.total_bytes_received: int = 0

    # ── Write ──────────────────────────────────────────────────────────────────

    def append(self, pcm_chunk: bytes) -> None:
        """Append a PCM16 chunk; silently discard the oldest bytes when full."""
        with self.lock:
            self.buffer.extend(pcm_chunk)
            self.total_bytes_received += len(pcm_chunk)
            if len(self.buffer) > self.max_bytes:
                overflow = len(self.buffer) - self.max_bytes
                del self.buffer[:overflow]

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_audio_float32(self, max_duration_sec: Optional[float] = None) -> np.ndarray:
        """Return the buffer (or its last *max_duration_sec* seconds) as float32."""
        with self.lock:
            if not self.buffer:
                return np.array([], dtype=np.float32)
            if max_duration_sec is not None:
                requested = int(max_duration_sec * self.bytes_per_second)
                raw = bytes(self.buffer[-requested:])
            else:
                raw = bytes(self.buffer)
        return pcm16_to_float32(raw)

    def has_enough_speech(self) -> bool:
        """True when the buffer contains voiced audio worth transcribing."""
        audio = self.get_audio_float32(max_duration_sec=settings.MAX_BUFFER_SECONDS)
        return has_speech(audio)

    # ── Housekeeping ───────────────────────────────────────────────────────────

    def duration_seconds(self) -> float:
        with self.lock:
            return len(self.buffer) / self.bytes_per_second

    def trim_front(self, seconds_to_trim: float, keep_overlap_sec: float = 0.3) -> None:
        """
        Advance the audio window after a commit.
        Keeps `keep_overlap_sec` seconds of tail for phoneme context.
        """
        with self.lock:
            trim_bytes    = int(seconds_to_trim   * self.bytes_per_second)
            overlap_bytes = int(keep_overlap_sec  * self.bytes_per_second)
            effective     = max(0, trim_bytes - overlap_bytes)
            if effective > 0 and len(self.buffer) >= effective:
                del self.buffer[:effective]

    def clear(self) -> None:
        with self.lock:
            self.buffer.clear()
            self.total_bytes_received = 0
