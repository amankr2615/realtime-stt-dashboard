import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import init_db

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_crud_lifecycle(client):
    # 1. Create a transcription
    payload = {
        "title": "Test Hinglish Note",
        "text": "Aaj meeting 4 baje hogi please join on time.",
        "language": "Hinglish",
        "duration_seconds": 3.5,
        "metrics": {"p50_ms": 140.5, "rtf": 0.28, "switches": 2}
    }
    create_resp = client.post("/api/transcriptions", json=payload)
    assert create_resp.status_code == 201
    created_data = create_resp.json()
    item_id = created_data["id"]
    assert created_data["title"] == payload["title"]
    assert created_data["text"] == payload["text"]

    # 2. Read (Get all)
    get_all_resp = client.get("/api/transcriptions")
    assert get_all_resp.status_code == 200
    items = get_all_resp.json()
    assert any(i["id"] == item_id for i in items)

    # 3. Read (Get by ID)
    get_one_resp = client.get(f"/api/transcriptions/{item_id}")
    assert get_one_resp.status_code == 200
    assert get_one_resp.json()["id"] == item_id

    # 4. Search
    search_resp = client.get("/api/transcriptions?search=meeting")
    assert search_resp.status_code == 200
    search_results = search_resp.json()
    assert len(search_results) > 0

    # 5. Update (Edit)
    update_payload = {"title": "Updated Hinglish Note", "text": "Meeting postponed to 5 PM."}
    update_resp = client.put(f"/api/transcriptions/{item_id}", json=update_payload)
    assert update_resp.status_code == 200
    assert update_resp.json()["title"] == "Updated Hinglish Note"
    assert update_resp.json()["text"] == "Meeting postponed to 5 PM."

    # 6. Delete
    delete_resp = client.delete(f"/api/transcriptions/{item_id}")
    assert delete_resp.status_code == 204

    # 7. Verify deletion
    verify_resp = client.get(f"/api/transcriptions/{item_id}")
    assert verify_resp.status_code == 404


def test_session_isolation(client):
    headers_a = {"X-Session-ID": "session-alpha"}
    headers_b = {"X-Session-ID": "session-beta"}

    # Create under session A
    resp = client.post(
        "/api/transcriptions",
        json={"title": "Secret A", "text": "Private note for A"},
        headers=headers_a,
    )
    assert resp.status_code == 201
    item_id = resp.json()["id"]

    # Session B cannot see it in list, get, update, or delete
    list_b = client.get("/api/transcriptions", headers=headers_b).json()
    assert all(item["id"] != item_id for item in list_b)
    assert client.get(f"/api/transcriptions/{item_id}", headers=headers_b).status_code == 404
    assert client.put(f"/api/transcriptions/{item_id}", json={"text": "Hacked"}, headers=headers_b).status_code == 404
    assert client.delete(f"/api/transcriptions/{item_id}", headers=headers_b).status_code == 404

    # Session A can read and delete it
    assert client.get(f"/api/transcriptions/{item_id}", headers=headers_a).status_code == 200
    assert client.delete(f"/api/transcriptions/{item_id}", headers=headers_a).status_code == 204


def test_pagination(client):
    headers = {"X-Session-ID": "session-page-test"}
    created_ids = []
    for i in range(3):
        r = client.post(
            "/api/transcriptions",
            json={"title": f"Note {i}", "text": f"Text {i}"},
            headers=headers,
        )
        assert r.status_code == 201
        created_ids.append(r.json()["id"])

    page1 = client.get("/api/transcriptions?limit=2&offset=0", headers=headers).json()
    page2 = client.get("/api/transcriptions?limit=2&offset=2", headers=headers).json()
    assert len(page1) == 2
    assert len(page2) == 1

    # Cleanup
    for cid in created_ids:
        client.delete(f"/api/transcriptions/{cid}", headers=headers)


def test_reject_blank_transcription(client):
    headers = {"X-Session-ID": "session-validation"}
    # Empty or whitespace-only text on POST must fail validation (422)
    assert client.post("/api/transcriptions", json={"text": ""}, headers=headers).status_code == 422
    assert client.post("/api/transcriptions", json={"text": "   \n\t  "}, headers=headers).status_code == 422

    # Valid create followed by whitespace-only PUT must fail validation (422)
    created = client.post("/api/transcriptions", json={"text": "valid text"}, headers=headers)
    assert created.status_code == 201
    cid = created.json()["id"]
    assert client.put(f"/api/transcriptions/{cid}", json={"text": "   "}, headers=headers).status_code == 422
    client.delete(f"/api/transcriptions/{cid}", headers=headers)


def test_health_endpoint_reports_configured_model(client):
    from backend.config import settings
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == f"{settings.WHISPER_MODEL}-{settings.WHISPER_COMPUTE_TYPE}"


def test_cors_no_wildcard_with_credentials(client):
    resp = client.get("/api/health", headers={"Origin": "https://untrusted.example"})
    assert resp.headers.get("access-control-allow-origin") != "*"
    assert resp.headers.get("access-control-allow-credentials") != "true"
