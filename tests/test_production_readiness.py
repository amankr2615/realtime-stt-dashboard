import os
import subprocess
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.database import create_db_engine, is_production_env
from backend.main import app, is_allowed_ws_origin


def test_dockerfile_port_expansion():
    dockerfile = Path("Dockerfile").read_text()
    assert 'CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]' in dockerfile

    # Verify actual shell expansion behavior with and without PORT
    env_unset = {k: v for k, v in os.environ.items() if k != "PORT"}
    out_default = subprocess.check_output(["sh", "-c", "echo ${PORT:-8000}"], env=env_unset, text=True).strip()
    assert out_default == "8000"

    env_custom = {**env_unset, "PORT": "10000"}
    out_custom = subprocess.check_output(["sh", "-c", "echo ${PORT:-8000}"], env=env_custom, text=True).strip()
    assert out_custom == "10000"


def test_render_yaml_has_health_check():
    render_yaml = Path("render.yaml").read_text()
    assert "healthCheckPath: /api/health" in render_yaml


def test_database_fail_fast_in_production():
    secret_pw = "super_secret_password_123"
    bad_pg_url = f"postgresql://myuser:{secret_pw}@127.0.0.1:1/nonexistent_db"

    with pytest.raises(RuntimeError) as exc_info:
        create_db_engine(bad_pg_url, is_production=True)

    # Ensure credentials are never leaked in error message
    assert secret_pw not in str(exc_info.value)
    assert "127.0.0.1:1/nonexistent_db" in str(exc_info.value)


def test_database_local_sqlite_fallback():
    bad_pg_url = "postgresql://myuser:pw@127.0.0.1:1/nonexistent_db"
    eng = create_db_engine(bad_pg_url, is_production=False)
    assert str(eng.url).startswith("sqlite:///")
    eng.dispose()


def test_is_production_env(monkeypatch):
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    assert is_production_env() is False

    monkeypatch.setenv("RENDER", "true")
    assert is_production_env() is True


def test_ws_origin_helper(monkeypatch):
    # Same-origin production
    assert is_allowed_ws_origin("https://hinglish-stt-studio.onrender.com", "hinglish-stt-studio.onrender.com") is True
    # Local development origins
    assert is_allowed_ws_origin("http://localhost:8000", "hinglish-stt-studio.onrender.com") is True
    assert is_allowed_ws_origin("http://127.0.0.1:8000", "localhost:8000") is True
    # Non-browser client omitting Origin
    assert is_allowed_ws_origin(None, "localhost:8000") is True
    # Untrusted cross-site origin
    assert is_allowed_ws_origin("https://evil.example", "hinglish-stt-studio.onrender.com") is False
    # Configured RENDER_EXTERNAL_URL
    monkeypatch.setenv("RENDER_EXTERNAL_URL", "https://custom-app.onrender.com")
    assert is_allowed_ws_origin("https://custom-app.onrender.com", "internal-host:8000") is True


def test_websocket_origin_enforcement():
    with TestClient(app) as client:
        # Allowed local origin connects and finalizes cleanly
        with client.websocket_connect("/ws/transcribe", headers={"Origin": "http://localhost:8000"}) as ws:
            ws.send_json({"action": "finalize"})
            msg = ws.receive_json()
            assert msg["type"] == "final"

        # Untrusted origin is rejected before accept
        with pytest.raises((WebSocketDisconnect, Exception)):
            with client.websocket_connect("/ws/transcribe", headers={"Origin": "https://evil.example"}):
                pass
