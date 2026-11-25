import re
import time
import numpy as np
from typing import List, Tuple

# Hinglish test corpus representing common code-switched phrases
HINGLISH_CORPUS = [
    "aaj meeting mein hum new feature deploy karenge",
    "kal client demo hai so please prepare the slides",
    "kya aapne latest code review check kiya",
    "server down ho gaya hai logs check karo",
    "main kal morning office jaunga",
    "database query slow hai optimize karna padega",
    "pull request merge kar di hai deploy start ho gaya",
    "aaj sham ko standup call kitne baje hai",
    "bhai frontend mein dark mode toggle add kar diya",
    "realtime audio streaming bahut smooth chal raha hai"
]

def plain_agreement(words_a: List[str], words_b: List[str]) -> List[str]:
    """Exact word match agreement."""
    common = []
    for w1, w2 in zip(words_a, words_b):
        if w1.lower() == w2.lower():
            common.append(w1)
        else:
            break
    return common

# Simple phonetic skeleton mapper (collapsing vowels, normalising common Hinglish phonemes)
def to_skeleton(word: str) -> str:
    w = word.lower()
    # Normalize common phonemes
    w = re.sub(r'[aeiou]+', 'v', w)  # vowel reduction
    w = w.replace('ph', 'f').replace('kh', 'k').replace('gh', 'g').replace('ch', 'c')
    w = re.sub(r'(.)\1+', r'\1', w)   # deduplicate double consonants
    return w

def skeleton_agreement(words_a: List[str], words_b: List[str]) -> List[str]:
    """Phonetic skeleton agreement."""
    common = []
    for w1, w2 in zip(words_a, words_b):
        if to_skeleton(w1) == to_skeleton(w2):
            common.append(w1)
        else:
            break
    return common

def run_benchmark():
    print("=" * 60)
    print(" Hinglish Agreement & Latency Micro-Benchmark")
    print("=" * 60)

    # 1. Benchmark Agreement Execution Speed
    iterations = 100_000
    w_a = ["aaj", "ki", "meeting", "me", "hum", "deploy", "karenge"]
    w_b = ["aaj", "kee", "meeting", "mein", "hum", "deploy", "karenge"]

    t0 = time.perf_counter()
    for _ in range(iterations):
        plain_agreement(w_a, w_b)
    t_plain = (time.perf_counter() - t0) * 1000 / iterations

    t0 = time.perf_counter()
    for _ in range(iterations):
        skeleton_agreement(w_a, w_b)
    t_skeleton = (time.perf_counter() - t0) * 1000 / iterations

    print(f"Plain Agreement Time:    {t_plain * 1000:.2f} µs per pass")
    print(f"Skeleton Agreement Time: {t_skeleton * 1000:.2f} µs per pass")

    # 2. Buffer Concatenation Benchmark
    n_chunks = 600  # 60 seconds of 100ms chunks
    chunk = b"\x00" * 3200

    # Naive append
    t0 = time.perf_counter()
    buf_naive = bytearray()
    for _ in range(n_chunks):
        buf_naive.extend(chunk)
    t_append = (time.perf_counter() - t0) * 1000

    # Bounded buffer with sliding window
    from backend.audio import BoundedAudioBuffer
    t0 = time.perf_counter()
    ring = BoundedAudioBuffer(max_seconds=5.0)
    for _ in range(n_chunks):
        ring.append(chunk)
    t_bounded = (time.perf_counter() - t0) * 1000

    print(f"Appending 60s of 100ms chunks: {t_append:.2f} ms")
    print(f"Bounded Rolling Buffer 60s:    {t_bounded:.2f} ms")
    print("=" * 60)
    print("Benchmark complete.")

if __name__ == "__main__":
    run_benchmark()
