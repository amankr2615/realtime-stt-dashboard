import re
import time
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from faster_whisper import WhisperModel
from backend.config import settings

logger = logging.getLogger(__name__)

def count_language_switches(text: str) -> int:
    """
    Counts script/language switches in Hinglish text (transitions between
    Devanagari script [U+0900 to U+097F] and Latin script [a-zA-Z]).
    """
    if not text:
        return 0
    words = text.strip().split()
    if len(words) < 2:
        return 0
    
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    latin_pattern = re.compile(r'[a-zA-Z]')
    
    current_script = None
    switches = 0
    
    for word in words:
        has_dev = bool(devanagari_pattern.search(word))
        has_lat = bool(latin_pattern.search(word))
        
        if has_dev and not has_lat:
            script = "devanagari"
        elif has_lat and not has_dev:
            script = "latin"
        else:
            continue
            
        if current_script is not None and script != current_script:
            switches += 1
        current_script = script
        
    return switches

class StreamingTranscriber:
    """
    Manages local agreement state for a single streaming session.
    Distinguishes between committed (finalized) and provisional (draft) text.
    """
    def __init__(self, agreement_count: int = 2):
        self.agreement_count = agreement_count
        self.committed_words: List[str] = []
        self.last_hypothesis_words: List[str] = []
        self.session_start_time = time.perf_counter()
        self.first_partial_time: Optional[float] = None
        self.inference_latencies: List[float] = []
        self.total_audio_processed_sec = 0.0

    def process_hypothesis(self, hypothesis_text: str, audio_duration_sec: float, inference_time_sec: float) -> Dict[str, Any]:
        """
        Applies local agreement: words matching across consecutive hypotheses are committed.
        Returns newly committed words and the remaining provisional tail.
        """
        now = time.perf_counter()
        if self.first_partial_time is None and hypothesis_text.strip():
            self.first_partial_time = (now - self.session_start_time) * 1000.0  # ms

        self.inference_latencies.append(inference_time_sec * 1000.0)
        self.total_audio_processed_sec += audio_duration_sec
        rtf = (inference_time_sec / audio_duration_sec) if audio_duration_sec > 0 else 0.0

        current_words = hypothesis_text.strip().split()
        
        # Determine common prefix between last hypothesis and current hypothesis
        common_len = 0
        min_len = min(len(self.last_hypothesis_words), len(current_words))
        for i in range(min_len):
            # Normalize for punctuation and case in agreement
            w1 = re.sub(r'[^\w\s]', '', self.last_hypothesis_words[i].lower())
            w2 = re.sub(r'[^\w\s]', '', current_words[i].lower())
            if w1 == w2 and w1 != "":
                common_len += 1
            else:
                break

        # Commit words confirmed by agreement
        newly_committed: List[str] = []
        if common_len >= 1:
            # Commit the agreed words
            newly_committed = current_words[:common_len]
            self.committed_words.extend(newly_committed)
            provisional_words = current_words[common_len:]
            self.last_hypothesis_words = provisional_words
        else:
            provisional_words = current_words
            self.last_hypothesis_words = current_words

        committed_text = " ".join(self.committed_words).strip()
        provisional_text = " ".join(provisional_words).strip()
        full_text = f"{committed_text} {provisional_text}".strip()
        switches = count_language_switches(full_text)

        p50_latency = float(np.median(self.inference_latencies)) if self.inference_latencies else 0.0
        p95_latency = float(np.percentile(self.inference_latencies, 95)) if self.inference_latencies else 0.0

        return {
            "committed_text": committed_text,
            "provisional_text": provisional_text,
            "newly_committed": " ".join(newly_committed),
            "full_text": full_text,
            "words_committed_count": len(newly_committed),
            "metrics": {
                "inference_ms": round(inference_time_sec * 1000.0, 1),
                "rtf": round(rtf, 3),
                "p50_ms": round(p50_latency, 1),
                "p95_ms": round(p95_latency, 1),
                "ttfp_ms": round(self.first_partial_time, 1) if self.first_partial_time else None,
                "switches": switches
            }
        }

    def finalize(self) -> Dict[str, Any]:
        """Flushes remaining provisional words into committed text on stream end."""
        if self.last_hypothesis_words:
            self.committed_words.extend(self.last_hypothesis_words)
            self.last_hypothesis_words = []

        final_text = " ".join(self.committed_words).strip()
        p50_latency = float(np.median(self.inference_latencies)) if self.inference_latencies else 0.0
        p95_latency = float(np.percentile(self.inference_latencies, 95)) if self.inference_latencies else 0.0

        return {
            "final_text": final_text,
            "metrics": {
                "p50_ms": round(p50_latency, 1),
                "p95_ms": round(p95_latency, 1),
                "ttfp_ms": round(self.first_partial_time, 1) if self.first_partial_time else None,
                "switches": count_language_switches(final_text),
                "total_audio_sec": round(self.total_audio_processed_sec, 2)
            }
        }


class WhisperEngine:
    """
    Singleton wrapper around faster-whisper.
    Ensures model is loaded only once and shared across requests.
    """
    _instance: Optional['WhisperEngine'] = None

    def __init__(self):
        logger.info(
            "Loading Faster-Whisper model='%s' compute_type='%s' device='%s'...",
            settings.WHISPER_MODEL, settings.WHISPER_COMPUTE_TYPE, settings.WHISPER_DEVICE
        )
        self.model = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            cpu_threads=settings.WHISPER_CPU_THREADS,
            download_root=None
        )
        logger.info("Faster-Whisper model loaded successfully.")

    @classmethod
    def get_instance(cls) -> 'WhisperEngine':
        if cls._instance is None:
            cls._instance = WhisperEngine()
        return cls._instance

    def transcribe(self, audio: np.ndarray, prompt: str = "") -> Tuple[str, float]:
        """
        Synchronously transcribes float32 16kHz audio.
        Designed to be called inside asyncio.to_thread to prevent blocking the event loop.
        """
        if len(audio) < int(settings.SAMPLE_RATE * 0.2):  # less than 200ms
            return "", 0.0

        t0 = time.perf_counter()
        
        # Beam size = 1 for fastest real-time inference
        # condition_on_previous_text=False prevents hallucination loops in streaming
        segments, info = self.model.transcribe(
            audio,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=prompt if prompt else "Hindi and English Hinglish speech conversation.",
            without_timestamps=True
        )

        texts = [segment.text for segment in segments]
        hypothesis = " ".join(texts).strip()
        elapsed = time.perf_counter() - t0
        
        return hypothesis, elapsed
