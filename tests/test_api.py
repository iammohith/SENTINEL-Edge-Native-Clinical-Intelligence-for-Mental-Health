import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from api.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_auth_token_endpoint(client):
    response = client.get("/auth/token")
    assert response.status_code == 200
    json_data = response.json()
    assert "token" in json_data
    assert len(json_data["token"]) > 0


def test_unauthenticated_api_access(client):
    # Accessing authenticated routes without token should return 401
    response = client.get("/api/conditions")
    assert response.status_code == 401


def test_authenticated_api_access(client):
    # First get a valid token
    auth_resp = client.get("/auth/token")
    token = auth_resp.json()["token"]
    
    # Access with header
    headers = {"x-session-token": token}
    response = client.get("/api/conditions", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "conditions" in data
    assert "DEP" in data["conditions"]


@patch("sentinel.store.vector_store.VectorStore.get_instance")
def test_system_status_endpoint(mock_get_instance, client):
    # Mock vector store row count
    mock_store = MagicMock()
    mock_store._table.count_rows.return_value = 142
    mock_get_instance.return_value = mock_store

    # Mock Ollama tags request
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"models": [{"name": "gemma4:e4b"}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["ollama"] == "HEALTHY"
        assert data["index_rows"] == 142
