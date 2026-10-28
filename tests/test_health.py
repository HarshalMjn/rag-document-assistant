from fastapi.testclient import TestClient

from app.main import app


def test_test_route():
    with TestClient(app) as client:
        response = client.get("/test")

    assert response.status_code == 200
    assert response.json() == {"message": "RAG Document Assistant API is running"}


def test_health_reports_dependencies():
    # Runs inside docker compose, where Qdrant and MySQL are reachable.
    with TestClient(app) as client:
        response = client.get("/health")

    body = response.json()
    assert response.status_code == 200, body
    assert body == {"status": "ok", "qdrant": "ok", "mysql": "ok"}
