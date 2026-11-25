import re
import time
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from faster_whisper import WhisperModel
from backend.config import settings

logger = logging.getLogger(__name__)

# Bilingual priming prompt that anchors Whisper's decoder to both Devanagari Hindi and English loanwords
DEFAULT_HINGLISH_PROMPT = (
    "नमस्ते, यह एक Hindi और English Hinglish बातचीत है। "
    "आज शाम चार बजे meeting है, please be on time. "
    "Features deploy ho chuke hain aur code review complete hai."
)

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

def strip_overlap_prefix(committed_words: List[str], new_words: List[str], max_overlap: int = 5) -> List[str]:
    """
    Removes duplicate overlapping prefix from new hypothesis if it repeats the tail of committed words.
    E.g. committed: [... "meeting", "hai"], new: ["meeting", "hai", "please", "come"] -> ["please", "come"]
    """
    if not committed_words or not new_words:
        return new_words

    norm_committed = [re.sub(r'[^\w\s]', '', w.lower()) for w in committed_words]
    norm_new = [re.sub(r'[^\w\s]', '', w.lower()) for w in new_words]

    # Check overlaps from max_overlap down to 1
    overlap_limit = min(len(norm_committed), len(norm_new), max_overlap)
    for k in range(overlap_limit, 0, -1):
        if norm_committed[-k:] == norm_new[:k]:
            return new_words[k:]

    return new_words

class StreamingTranscriber:
    """
    Manages streaming agreement state with duplicate suppression.
    """
    def __init__(self, agreement_count: int = 2):
        self.agreement_count = agreement_count
        self.committed_words: List[str] = []
        self.last_hypothesis_words: List[str] = []
        self.session_start_time = time.perf_counter()
        self.first_partial_time: Optional[float] = None
        self.inference_latencies: List[float] = []
        self.total_audio_processed_sec = 0.0

    def process_hypothesis(
        self,
        hypothesis_text: str,
        audio_duration_sec: float,
        inference_time_sec: float
    ) -> Dict[str, Any]:
        now = time.perf_counter()
        if self.first_partial_time is None and hypothesis_text.strip():
            self.first_partial_time = (now - self.session_start_time) * 1000.0

        self.inference_latencies.append(inference_time_sec * 1000.0)
        self.total_audio_processed_sec += audio_duration_sec
        rtf = (inference_time_sec / audio_duration_sec) if audio_duration_sec > 0 else 0.0

        raw_words = hypothesis_text.strip().split()
        
        # 1. Strip any overlap with previously committed words
        stripped_words = strip_overlap_prefix(self.committed_words, raw_words)

        # 2. Check agreement between last provisional hypothesis and stripped current words
        common_len = 0
        min_len = min(len(self.last_hypothesis_words), len(stripped_words))
        for i in range(min_len):
            w1 = re.sub(r'[^\w\s]', '', self.last_hypothesis_words[i].lower())
            w2 = re.sub(r'[^\w\s]', '', stripped_words[i].lower())
            if w1 == w2 and w1 != "":
                common_len += 1
            else:
                break

        newly_committed: List[str] = []
        if common_len >= 1:
            newly_committed = stripped_words[:common_len]
            self.committed_words.extend(newly_committed)
            provisional_words = stripped_words[common_len:]
            self.last_hypothesis_words = provisional_words
        else:
            provisional_words = stripped_words
            self.last_hypothesis_words = stripped_words

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
        """Flushes remaining provisional words into committed text without duplication."""
        if self.last_hypothesis_words:
            clean_tail = strip_overlap_prefix(self.committed_words, self.last_hypothesis_words)
            self.committed_words.extend(clean_tail)
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
    Singleton Faster-Whisper wrapper.
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

    def transcribe(
        self,
        audio: np.ndarray,
        language_mode: str = "hi",
        prompt: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Transcribes audio with language and bilingual prompt conditioning.
        """
        if len(audio) < int(settings.SAMPLE_RATE * 0.2):
            return "", 0.0

        t0 = time.perf_counter()
        
        # Determine language parameter
        # "hi" gives optimal mixed Hinglish (Hindi in Devanagari + English loanwords)
        # "en" forces Roman transliteration
        # "auto" detects automatically
        lang = None if language_mode == "auto" else language_mode
        initial_prompt = prompt if prompt else DEFAULT_HINGLISH_PROMPT

        segments, info = self.model.transcribe(
            audio,
            beam_size=1,
            temperature=0.0,
            language=lang,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            without_timestamps=True
        )

        texts = [segment.text for segment in segments]
        hypothesis = " ".join(texts).strip()
        elapsed = time.perf_counter() - t0
        
        return hypothesis, elapsed
