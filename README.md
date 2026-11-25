# Real-Time Hinglish Speech-to-Text Studio & Dashboard

A lightweight, low-latency multilingual Speech-to-Text (STT) web application tailored for **Hinglish (Hindi + English)** code-switched speech. Built with FastAPI, WebSockets, `AudioWorklet`, `faster-whisper`, PostgreSQL/SQLite persistence with session-scoped CRUD, and Text-to-Speech (TTS).

---

## 🌟 Key Features

1. **Real-Time Streaming STT (AudioWorklet + WebSocket):**
   - `AudioWorklet` captures 16 kHz mono linear PCM16 on the browser audio thread (preserving silent pauses as zero-valued PCM samples).
   - Streams 100 ms batches (3,200 bytes per frame) over WebSocket without `MediaRecorder` container muxing overhead.

2. **Hinglish Stabilization & Local Agreement:**
   - Powered by `faster-whisper` (`base` `int8` by default locally; `tiny` `int8` on Render's 512 MB free tier), loaded once at server startup.
   - Distinct state separation: **Committed (confirmed)** words vs. **Provisional (faint)** hypothesis to reduce word repetition and UI flicker.
   - Non-blocking inference: Model execution is dispatched via `asyncio.to_thread` behind a global concurrency semaphore (`Semaphore(1)`) so the FastAPI async event loop stays responsive.

3. **Persistent History with Session-Scoped CRUD:**
   - Full CRUD REST API (`GET`, `POST`, `PUT`, `DELETE` at `/api/transcriptions`) with `limit`/`offset` pagination.
   - Anonymous per-browser `session_id` isolation via `X-Session-ID` header.
   - SQLite zero-configuration local fallback, PostgreSQL on Render.

4. **Text-to-Speech (TTS):**
   - Browser-native `window.speechSynthesis` (`hi-IN`) as primary zero-latency playback, with backend `edge-tts` endpoint (`/api/tts`) as fallback when outbound network is available.

5. **Live Telemetry & Metrics HUD:**
   - Live displays for **Inference Latency (ms)**, **Stream Real-Time Factor (RTF)**, and **Language Switch Points**.

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

## 🔬 Measured vs. Estimated Benchmarks

### Measured Micro-Benchmarks (Apple Silicon M4, 16 GB RAM)
Executed via `PYTHONPATH=. python benchmarks/benchmark.py` (7-word agreement over 100,000 iterations, and 60 s of 100 ms PCM16 chunks):

| Benchmark Stage | Measured Time | Type |
| :--- | :--- | :--- |
| **Plain Word Agreement** | `0.18 µs / pass` | Measured (`benchmarks/benchmark.py`) |
| **Phonetic Skeleton Agreement** | `5.11 µs / pass` | Measured (`benchmarks/benchmark.py`) |
| **Appending 60s of 100ms chunks** | `0.17 ms total` | Measured (`benchmarks/benchmark.py`) |
| **Bounded Rolling Buffer (60s)** | `0.37 ms total` | Measured (`benchmarks/benchmark.py`) |

### Live Inference (Hardware-Dependent)
| Environment | Expected Inference Time | Expected Stream RTF | Type |
| :--- | :--- | :--- | :--- |
| **Apple M4 CPU (`base-int8`)** | `~150 - 450 ms` | `< 0.4` | Observed in live sessions |
| **Render Free Tier (`tiny-int8`, ~0.1 vCPU)** | `~1.0 - 3.0 s` | `~0.5 - 1.5` (bursty) | Estimated (shared CPU) |

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
├── Dockerfile                 # Container image with pre-cached Whisper model
├── render.yaml                # Render Blueprint (Web + PostgreSQL)
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart Guide (Local Development)

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/amankr2615/realtime-stt-dashboard.git
cd realtime-stt-dashboard

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

The pytest suite verifies audio buffering, local agreement, RTF tracking, reset state, session isolation, pagination, and database CRUD.

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
