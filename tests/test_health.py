from fastapi.testclient import TestClient

from app.config.settings import settings
from app.main import app


def test_config_shows_settings_without_secrets():
    with TestClient(app) as client:
        response = client.get("/config")

    body = response.json()["data"]
    assert response.status_code == 200
    assert body["chunk_size"] == settings.CHUNK_SIZE
    assert body["history_limit"] == settings.HISTORY_LIMIT
    assert body["min_score"] == settings.MIN_SCORE
    text = response.text.lower()
    for secret in ("key", "password", "host"):
        assert secret not in text


def test_health_reports_dependencies():
    # Runs inside docker compose, where Qdrant and MySQL are reachable.
    with TestClient(app) as client:
        response = client.get("/health")

    body = response.json()["data"]
    assert response.status_code == 200, body
    assert body == {"status": "ok", "qdrant": "ok", "mysql": "ok"}
