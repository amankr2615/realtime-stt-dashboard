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
