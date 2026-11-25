import re
import time
import logging
from collections import Counter, deque
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from faster_whisper import WhisperModel
from backend.config import settings

logger = logging.getLogger(__name__)

# ── Bilingual decoder prompt ──────────────────────────────────────────────────
# This anchors Whisper's beam search toward mixed Hindi/English vocabulary.
DEFAULT_HINGLISH_PROMPT = (
    "नमस्ते, यह एक Hindi और English Hinglish बातचीत है। "
    "आज शाम चार बजे meeting है, please be on time. "
    "Features deploy ho chuke hain aur code review complete hai."
)

# ── Hallucination guard ───────────────────────────────────────────────────────
# If a single token represents > this fraction of all tokens, the output is a
# repetition hallucination (e.g. "नमस्ते, नमस्ते, नमस्ते …").
MAX_SINGLE_TOKEN_RATIO = 0.5


def is_hallucination(text: str) -> bool:
    """
    Detects Whisper repetition hallucinations.

    A hypothesis is flagged when:
      1. Any single word accounts for >50 % of all words, OR
      2. The same 2‑gram repeats more than 4 times.
    """
    if not text.strip():
        return False

    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    if len(words) < 3:
        return False

    # Word‑level repetition check
    counts = Counter(words)
    top_word, top_count = counts.most_common(1)[0]
    if top_count / len(words) > MAX_SINGLE_TOKEN_RATIO:
        logger.debug("Hallucination detected – '%s' repeated %d/%d", top_word, top_count, len(words))
        return True

    # 2‑gram repetition check
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    if bigrams:
        _, bg_count = Counter(bigrams).most_common(1)[0]
        if bg_count > 4:
            logger.debug("Bigram hallucination detected, count=%d", bg_count)
            return True

    return False


def count_language_switches(text: str) -> int:
    """
    Counts Devanagari ↔ Latin script transitions in Hinglish text.
    """
    if not text:
        return 0
    words = text.strip().split()
    if len(words) < 2:
        return 0

    dev = re.compile(r'[\u0900-\u097F]')
    lat = re.compile(r'[a-zA-Z]')
    current, switches = None, 0

    for w in words:
        if dev.search(w) and not lat.search(w):
            script = "d"
        elif lat.search(w) and not dev.search(w):
            script = "l"
        else:
            continue
        if current is not None and script != current:
            switches += 1
        current = script

    return switches


def strip_overlap_prefix(committed: List[str], new_words: List[str], max_overlap: int = 6) -> List[str]:
    """
    Removes repeated prefix from *new_words* that already exists as the tail
    of *committed*, preventing word duplication across inference passes.
    """
    if not committed or not new_words:
        return new_words

    norm_c = [re.sub(r'[^\w\s]', '', w.lower()) for w in committed]
    norm_n = [re.sub(r'[^\w\s]', '', w.lower()) for w in new_words]
    limit  = min(len(norm_c), len(norm_n), max_overlap)

    for k in range(limit, 0, -1):
        if norm_c[-k:] == norm_n[:k]:
            return new_words[k:]
    return new_words


# ─────────────────────────────────────────────────────────────────────────────

class StreamingTranscriber:
    """
    Per‑session streaming state machine.

    Responsibilities
    ────────────────
    • Tracks *committed* (stable) and *provisional* (draft) word lists.
    • Applies local agreement: words must appear in two consecutive hypotheses
      before they are committed.
    • Strips duplicate overlaps introduced by Whisper's rolling window.
    • Collects per‑session latency metrics.
    """

    def __init__(self, max_latency_samples: int = 200):
        self.max_latency_samples = max_latency_samples
        self.reset()

    def reset(self) -> None:
        """Clears all accumulated transcription and telemetry state."""
        self.committed_words:       List[str] = []
        self.last_hypothesis_words: List[str] = []
        self.session_start          = time.perf_counter()
        self.first_partial_ms:      Optional[float] = None
        self.inference_latencies:   deque = deque(maxlen=self.max_latency_samples)
        self.total_inference_sec:   float = 0.0
        self.total_received_audio_sec: float = 0.0
        self.hallucination_count:   int = 0

    def process_hypothesis(
        self,
        hypothesis_text: str,
        audio_duration_sec: float,
        inference_time_sec: float,
        new_audio_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        now = time.perf_counter()
        if self.first_partial_ms is None and hypothesis_text.strip():
            self.first_partial_ms = (now - self.session_start) * 1000.0

        self.inference_latencies.append(inference_time_sec * 1000.0)
        self.total_inference_sec += inference_time_sec
        added_audio = new_audio_sec if new_audio_sec is not None else audio_duration_sec
        self.total_received_audio_sec += max(0.0, added_audio)

        window_rtf = inference_time_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0
        stream_rtf = (
            self.total_inference_sec / self.total_received_audio_sec
            if self.total_received_audio_sec > 0 else 0.0
        )

        # ── Hallucination guard ──────────────────────────────────────────────
        if is_hallucination(hypothesis_text):
            self.hallucination_count += 1
            logger.warning("Hallucination #%d suppressed: %.60s…", self.hallucination_count, hypothesis_text)
            # Return current state unchanged — do NOT update committed/provisional
            return self._make_result([], self.last_hypothesis_words, inference_time_sec, stream_rtf, window_rtf)

        raw_words    = hypothesis_text.strip().split()
        # Strip words already committed from the new hypothesis
        stripped     = strip_overlap_prefix(self.committed_words, raw_words)

        # ── Local agreement ──────────────────────────────────────────────────
        common_len = 0
        min_len = min(len(self.last_hypothesis_words), len(stripped))
        for i in range(min_len):
            w1 = re.sub(r'[^\w\s]', '', self.last_hypothesis_words[i].lower())
            w2 = re.sub(r'[^\w\s]', '', stripped[i].lower())
            if w1 == w2 and w1:
                common_len += 1
            else:
                break

        if common_len >= 1:
            newly = stripped[:common_len]
            self.committed_words.extend(newly)
            provisional = stripped[common_len:]
        else:
            newly = []
            provisional = stripped

        self.last_hypothesis_words = provisional
        return self._make_result(newly, provisional, inference_time_sec, stream_rtf, window_rtf)

    def _make_result(
        self,
        newly_committed: List[str],
        provisional: List[str],
        inference_time_sec: float,
        stream_rtf: float,
        window_rtf: float,
    ) -> Dict[str, Any]:
        committed_text   = " ".join(self.committed_words).strip()
        provisional_text = " ".join(provisional).strip()
        full_text        = f"{committed_text} {provisional_text}".strip()
        p50 = float(np.median(self.inference_latencies)) if self.inference_latencies else 0.0
        p95 = float(np.percentile(self.inference_latencies, 95)) if self.inference_latencies else 0.0

        return {
            "committed_text":      committed_text,
            "provisional_text":    provisional_text,
            "full_text":           full_text,
            "words_committed_count": len(newly_committed),
            "metrics": {
                "inference_ms": round(inference_time_sec * 1000, 1),
                "rtf":          round(stream_rtf, 3),
                "window_rtf":   round(window_rtf, 3),
                "p50_ms":       round(p50, 1),
                "p95_ms":       round(p95, 1),
                "ttfp_ms":      round(self.first_partial_ms, 1) if self.first_partial_ms else None,
                "switches":     count_language_switches(full_text),
            },
        }

    def finalize(self) -> Dict[str, Any]:
        """Flush provisional tail into committed text without duplication."""
        if self.last_hypothesis_words:
            clean = strip_overlap_prefix(self.committed_words, self.last_hypothesis_words)
            self.committed_words.extend(clean)
            self.last_hypothesis_words = []

        final_text = " ".join(self.committed_words).strip()
        p50 = float(np.median(self.inference_latencies)) if self.inference_latencies else 0.0
        p95 = float(np.percentile(self.inference_latencies, 95)) if self.inference_latencies else 0.0
        stream_rtf = (
            self.total_inference_sec / self.total_received_audio_sec
            if self.total_received_audio_sec > 0 else 0.0
        )

        return {
            "final_text": final_text,
            "metrics": {
                "p50_ms":          round(p50, 1),
                "p95_ms":          round(p95, 1),
                "rtf":             round(stream_rtf, 3),
                "ttfp_ms":         round(self.first_partial_ms, 1) if self.first_partial_ms else None,
                "switches":        count_language_switches(final_text),
                "total_audio_sec": round(self.total_received_audio_sec, 2),
                "hallucinations_suppressed": self.hallucination_count,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────

class WhisperEngine:
    """
    Singleton faster‑whisper wrapper.
    Model is loaded once at startup and reused for all WebSocket sessions.
    """

    _instance: Optional["WhisperEngine"] = None

    def __init__(self):
        logger.info(
            "Loading faster‑whisper  model=%s  compute=%s  device=%s",
            settings.WHISPER_MODEL, settings.WHISPER_COMPUTE_TYPE, settings.WHISPER_DEVICE,
        )
        self.model = WhisperModel(
            settings.WHISPER_MODEL,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
            cpu_threads=settings.WHISPER_CPU_THREADS,
        )
        logger.info("faster‑whisper ready.")

    @classmethod
    def get_instance(cls) -> "WhisperEngine":
        if cls._instance is None:
            cls._instance = WhisperEngine()
        return cls._instance

    def transcribe(
        self,
        audio: np.ndarray,
        language_mode: str = "hi",
        prompt: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Synchronous transcription — always call via asyncio.to_thread().

        Parameters
        ──────────
        audio          float32 16 kHz mono array
        language_mode  "hi"   → Hindi + English loanwords in native scripts (best for Hinglish)
                       "en"   → Roman‑script Hinglish
                       "auto" → auto language detection
        prompt         Optional decoder prompt override.
        """
        if len(audio) < int(settings.SAMPLE_RATE * 0.3):   # < 300 ms → skip
            return "", 0.0

        t0  = time.perf_counter()
        lang = None if language_mode == "auto" else language_mode

        segments, _ = self.model.transcribe(
            audio,
            language=lang,
            beam_size=1,
            temperature=0.0,
            initial_prompt=prompt,
            # Suppress hallucination helpers
            condition_on_previous_text=False,
            no_speech_threshold=0.6,       # skip segments with low speech probability
            compression_ratio_threshold=2.0,  # discard outputs that look like repetitions
            log_prob_threshold=-0.8,       # discard low‑confidence segments
            without_timestamps=True,
        )

        texts = [s.text for s in segments]
        elapsed = time.perf_counter() - t0
        return " ".join(texts).strip(), elapsed
