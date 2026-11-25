import math
import struct
import threading
import numpy as np
from typing import Optional
from backend.config import settings

def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Converts raw 16-bit signed PCM mono audio bytes to float32 array in [-1.0, 1.0]."""
    if not pcm_bytes:
        return np.array([], dtype=np.float32)
    # 16-bit signed integer format
    int16_arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    return int16_arr.astype(np.float32) / 32768.0

def calculate_rms_energy(pcm_bytes: bytes) -> float:
    """Calculates RMS energy (loudness) of raw PCM16 bytes for silence detection."""
    if len(pcm_bytes) < 2:
        return 0.0
    int16_arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    if len(int16_arr) == 0:
        return 0.0
    sum_squares = np.sum(int16_arr.astype(np.float64) ** 2)
    mean_square = sum_squares / len(int16_arr)
    return math.sqrt(mean_square)

class BoundedAudioBuffer:
    """
    Thread-safe bounded audio buffer for streaming 16kHz mono PCM16 audio.
    Maintains a rolling window to prevent memory leaks and bounded CPU usage.
    """
    def __init__(self, sample_rate: int = 16000, max_seconds: float = 6.0):
        self.sample_rate = sample_rate
        self.bytes_per_second = sample_rate * 2  # 16-bit mono = 2 bytes per sample
        self.max_bytes = int(max_seconds * self.bytes_per_second)
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.total_bytes_received = 0

    def append(self, pcm_chunk: bytes) -> None:
        """Appends raw PCM chunk. Discards oldest audio if buffer exceeds max_bytes."""
        with self.lock:
            self.buffer.extend(pcm_chunk)
            self.total_bytes_received += len(pcm_chunk)
            # Bound the buffer to avoid unbounded memory accumulation
            if len(self.buffer) > self.max_bytes:
                overflow = len(self.buffer) - self.max_bytes
                del self.buffer[:overflow]

    def get_audio_float32(self, max_duration_sec: Optional[float] = None) -> np.ndarray:
        """Returns the audio buffer as a normalized float32 numpy array."""
        with self.lock:
            if not self.buffer:
                return np.array([], dtype=np.float32)
            
            if max_duration_sec is not None:
                requested_bytes = int(max_duration_sec * self.bytes_per_second)
                audio_bytes = bytes(self.buffer[-requested_bytes:])
            else:
                audio_bytes = bytes(self.buffer)

        return pcm16_to_float32(audio_bytes)

    def duration_seconds(self) -> float:
        """Current duration of audio in the buffer in seconds."""
        with self.lock:
            return len(self.buffer) / self.bytes_per_second

    def trim_front(self, seconds_to_trim: float, keep_overlap_sec: float = 0.3) -> None:
        """
        Trims audio from the front after text commitment,
        keeping a small overlap for acoustic context.
        """
        with self.lock:
            bytes_to_trim = int(seconds_to_trim * self.bytes_per_second)
            overlap_bytes = int(keep_overlap_sec * self.bytes_per_second)
            effective_trim = max(0, bytes_to_trim - overlap_bytes)
            
            if effective_trim > 0 and len(self.buffer) >= effective_trim:
                del self.buffer[:effective_trim]

    def clear(self) -> None:
        """Clears the buffer completely."""
        with self.lock:
            self.buffer.clear()
            self.total_bytes_received = 0
