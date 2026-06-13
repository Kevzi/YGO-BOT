import pytest
import httpx
import json
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_import_deck_success(httpx_mock):
    # Arrange
    httpx_mock.add_response(
        url="http://localhost:8000/api/v1/parse",
        json={"main": [1234, 5678], "extra": [9012], "side": []}
    )
    
    # Act
    response = client.post(
        "/api/v1/decks/import",
        json={"deckData": "some_ydk_content"}
    )
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["main"] == [1234, 5678]
    assert data["extra"] == [9012]
    assert data["side"] == []
    
    # Validate outbound payload
    request = httpx_mock.get_request()
    assert request is not None
    request_body = json.loads(request.read().decode("utf-8"))
    assert request_body == {"deck": "some_ydk_content"}

def test_import_deck_microservice_error(httpx_mock):
    httpx_mock.add_response(url="http://localhost:8000/api/v1/parse", status_code=500)
    response = client.post("/api/v1/decks/import", json={"deckData": "ydk"})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "DECK_PARSER_ERROR"

def test_import_deck_network_timeout(httpx_mock):
    httpx_mock.add_exception(httpx.RequestError("Timeout"), url="http://localhost:8000/api/v1/parse")
    response = client.post("/api/v1/decks/import", json={"deckData": "ydk"})
    assert response.status_code == 500
    assert "Network error" in response.json()["error"]["detail"]

def test_import_deck_invalid_json(httpx_mock):
    httpx_mock.add_response(url="http://localhost:8000/api/v1/parse", json=[]) # not a dict
    response = client.post("/api/v1/decks/import", json={"deckData": "ydk"})
    assert response.status_code == 500
    assert "invalid JSON format" in response.json()["error"]["detail"]

def test_import_deck_invalid_payload():
    response = client.post("/api/v1/decks/import", json={"deckData": ""})
    assert response.status_code == 422 # Pydantic validation error (min_length=1)
