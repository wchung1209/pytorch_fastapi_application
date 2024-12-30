import pytest
from fastapi.testclient import TestClient
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend

from src.main import app


@pytest.fixture
def client():
    FastAPICache.init(InMemoryBackend())
    with TestClient(app) as c:
        yield c

def test_health(client):
    response = client.get("/project/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict(client):
    data = {"text": ["I hate you.", "I love you."]}
    response = client.post(
        "/project/bulk-predict",
        json=data,
    )
    print(response.json())
    assert response.status_code == 200
    assert isinstance(response.json()["predictions"], list)
    assert isinstance(response.json()["predictions"][0], list)
    assert isinstance(response.json()["predictions"][0][0], dict)
    assert isinstance(response.json()["predictions"][1][0], dict)
    assert set(response.json()["predictions"][0][0].keys()) == {"label", "score"}
    assert set(response.json()["predictions"][0][1].keys()) == {"label", "score"}
    assert set(response.json()["predictions"][1][0].keys()) == {"label", "score"}
    assert set(response.json()["predictions"][1][1].keys()) == {"label", "score"}
    assert response.json()["predictions"][0][0]["label"] == "NEGATIVE"
    assert response.json()["predictions"][0][1]["label"] == "POSITIVE"
    assert response.json()["predictions"][1][0]["label"] == "POSITIVE"
    assert response.json()["predictions"][1][1]["label"] == "NEGATIVE"

def test_predict_empty_input(client):
    data = {"text": []}
    response = client.post(
        "/project/bulk-predict",
        json=data,
    )
    assert response.status_code == 200
    assert response.json() == {"predictions": []}

def test_predict_invalid_input(client):
    data = {"text": "This is not a list"}  # text must be a list of strings
    response = client.post(
        "/project/bulk-predict",
        json=data,
    )
    assert response.status_code == 422
    assert "detail" in response.json()

def test_predict_extra_fields(client):
    data = {"text": ["I hate you.", "I love you."], 
            "extra_field": "not needed"}
    response = client.post(
        "/project/bulk-predict",
        json=data,
    )
    assert response.status_code == 422
    assert "detail" in response.json()

def test_predict_empty_string(client):
    data = {"text": ["I hate you.", "    ", "I love you."]}
    response = client.post(
        "/project/bulk-predict",
        json=data,
    )
    assert response.status_code == 422
    assert "detail" in response.json()
