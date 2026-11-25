FROM python:3.12-slim

# Prevent Python from writing .pyc, buffer stdout/stderr, and store HF cache in a writable path
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

# Create unprivileged user (UID 1000) required by Hugging Face Docker Spaces
RUN useradd -m -u 1000 user

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Whisper tiny model into /app/.cache/huggingface for zero cold-start delay
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cpu', compute_type='int8')"

# Copy application files and grant ownership to UID 1000
COPY --chown=user:user . .
RUN chown -R user:user /app

USER user

# Expose default port
EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
