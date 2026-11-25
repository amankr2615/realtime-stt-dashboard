# Real-Time Hinglish Speech-to-Text Studio & Dashboard

A production-grade, low-latency multilingual Speech-to-Text (STT) web service tailored for **Hinglish (Hindi + English)** code-switched speech. Features real-time audio streaming via WebSockets and AudioWorklet, local agreement stabilization with `faster-whisper`, PostgreSQL transcription persistence with full CRUD operations, and open-source Text-to-Speech (TTS).

---

## 🌟 Key Features

1. **Real-Time Streaming STT (AudioWorklet + WebSocket):**
   - High-performance `AudioWorklet` captures 16 kHz mono linear PCM16 directly on a dedicated Web Audio thread.
   - Low-latency WebSocket transmission in 100 ms batches (3,200 bytes per frame).
   - Zero browser audio encoding overhead.

2. **Hinglish Stabilization & Local Agreement:**
   - Powered by `faster-whisper` (`tiny`, `int8`), loaded once at server startup.
   - Distinct state separation: **Committed (confirmed)** words vs. **Provisional (faint)** hypothesis to eliminate word repetition and UI flicker.
   - Non-blocking inference: Model execution is dispatched to a background worker thread (`asyncio.to_thread`) to guarantee the FastAPI async event loop never starves.

3. **Persistent History with PostgreSQL CRUD:**
   - Full CRUD REST API (`GET`, `POST`, `PUT`, `DELETE` at `/api/transcriptions`).
   - Stores metadata: transcript text, audio duration, code-switch frequency, and latency telemetry.
   - Full-text search and filtering across saved transcripts.
   - SQLite zero-configuration local fallback, PostgreSQL in production.

4. **Open-Source Text-to-Speech (TTS):**
   - Built-in neural TTS synthesis using `edge-tts` (supports natural Hindi and Indian English neural voices) with browser Web Speech API fallback.

5. **Live Telemetry & Metrics HUD:**
   - Live displays for **Inference Latency (ms)**, **Real-Time Factor (RTF)**, and **Language Switch Points**.

---

## 📐 Architecture & Data Flow

```
[Browser Microphone]
       │
       ▼ (16kHz Mono Float32)
[AudioWorklet: pcm16-processor.js]
       │
       ▼ (100ms PCM16 binary chunks, 3200 bytes)
[WebSocket Stream: /ws/transcribe]
       │
       ▼
[FastAPI Backend (BoundedAudioBuffer: 5s max)]
       │
       ▼ (asyncio.to_thread - Offloaded worker)
[faster-whisper: tiny / int8 (CPU)]
       │
       ▼
[Local Agreement Engine]
       ├──► Emits newly committed words
       └──► Retains provisional tail
       │
       ▼ (JSON response via WebSocket)
[Live UI Canvas] ──► [PostgreSQL Database (CRUD Storage)]
```

---

## 🔬 Measured Benchmarks (Apple Silicon M4)

Benchmarks executed on local hardware (MacBook Air M4, 16 GB unified RAM):

| Benchmark Stage | Measured Time | Notes |
| :--- | :--- | :--- |
| **Plain Word Agreement (36 words)** | `0.18 µs` | Zero-overhead exact string matching |
| **Phonetic Skeleton Agreement (36 words)** | `5.11 µs` | Handles phonetic variance (`kal` vs `kall`) |
| **Appending 60s of 100ms audio chunks** | `0.17 ms` | Buffer throughput |
| **Bounded Rolling Buffer (Sliding Window)** | `0.37 ms` | Fixed upper-bound memory usage |
| **Inference Time (Whisper Tiny int8, CPU)** | `~120 - 180 ms` | Running on 2 CPU threads |
| **Real-Time Factor (RTF)** | `~0.15 - 0.28` | **Well below 1.0 threshold** (sustainable live streaming) |

---

## 📁 Repository Structure

```
.
├── backend/
│   ├── __init__.py
│   ├── config.py              # Application settings & parameters
│   ├── database.py            # SQLAlchemy engine, session & auto-fallback
│   ├── models.py              # PostgreSQL database models
│   ├── schemas.py             # Pydantic schemas for CRUD & TTS
│   ├── audio.py               # BoundedAudioBuffer & PCM16 conversions
│   ├── transcriber.py         # WhisperEngine & StreamingTranscriber
│   ├── tts.py                 # Open-source neural TTS synthesis
│   └── main.py                # FastAPI app (REST + WebSocket + Static)
├── static/
│   ├── index.html             # Responsive Tailwind CSS dashboard
│   ├── audio-processor.js     # 16kHz PCM16 AudioWorkletProcessor
│   ├── app.js                 # WebSocket client, CRUD & UI controller
│   └── style.css              # Custom styling & pulse animations
├── tests/
│   ├── __init__.py
│   ├── test_audio.py          # Buffer bounds & PCM conversion tests
│   ├── test_transcriber.py    # Local agreement & code-switch tests
│   └── test_crud.py           # Full CRUD endpoint integration tests
├── benchmarks/
│   └── benchmark.py           # Empirical performance measurement script
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Multi-stage production container
├── render.yaml                # Render Blueprint (Web + PostgreSQL)
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart Guide (Local Development)

### 1. Clone & Set Up Virtual Environment

```bash
git clone <your-repo-url>
cd Hinglish

# Create virtual environment (Python 3.11 or 3.12)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Tests

```bash
PYTHONPATH=. pytest -v
```

All 7 unit and integration tests will verify audio buffering, agreement logic, and database CRUD.

### 3. Run Performance Benchmarks

```bash
PYTHONPATH=. python benchmarks/benchmark.py
```

### 4. Start Development Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in your browser. Grant microphone access and speak in Hindi, English, or Hinglish!

---

## ☁️ Deployment on Render

This repository includes a pre-configured `render.yaml` Blueprint and `Dockerfile`:

1. Push your repository to GitHub.
2. In [Render Dashboard](https://dashboard.render.com), click **New +** $\rightarrow$ **Blueprint**.
3. Select your repository. Render will automatically provision:
   - **Web Service:** Containerized FastAPI app with pre-cached Whisper model.
   - **PostgreSQL Database:** Managed relational database connected via `DATABASE_URL`.
4. Once deployed, open your live `https://<app-name>.onrender.com` URL (HTTPS ensures microphone access works in all modern browsers).

---

## 🛡️ License

MIT License. Developed for technical assessment.
