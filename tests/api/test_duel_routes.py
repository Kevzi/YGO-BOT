import pytest
from fastapi.testclient import TestClient
from core.ygoenv.wrapper import EngineCrashError
from api.duel_routes import get_db
from api.main import app

@pytest.fixture(autouse=True)
def override_get_db(session):
    app.dependency_overrides[get_db] = lambda: session
    yield
    app.dependency_overrides.clear()

client = TestClient(app)

def test_camel_case_to_snake_case_parsing(monkeypatch):
    # Mock the engine so it doesn't actually run C++
    def mock_get_legal_actions(self, state):
        assert state == {"current_phase": 1, "turn_player": 0}
        return [{"action_type": 1, "card_id": 123456}]
    
    from core.ygoenv.wrapper import YgoEngine
    monkeypatch.setattr(YgoEngine, "get_legal_actions", mock_get_legal_actions)

    payload = {
        "duelId": "123",
        "gameState": {
            "currentPhase": 1,
            "turnPlayer": 0
        }
    }
    
    response = client.post("/api/v1/actions", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    # Now it should return a single action
    assert "actionType" in data
    assert data["actionType"] == 1

def test_engine_crash_fail_fast(monkeypatch):
    def mock_crash(self, state):
        from core.ygoenv.wrapper import EngineCrashError
        raise EngineCrashError("Engine crashed due to invalid state")
        
    from core.ygoenv.wrapper import YgoEngine
    monkeypatch.setattr(YgoEngine, "get_legal_actions", mock_crash)
    
    payload = {
        "duelId": "123",
        "gameState": {
            "currentPhase": 1,
            "turnPlayer": 0
        }
    }
    
    response = client.post("/api/v1/actions", json=payload)
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "ENGINE_CRASH"

def test_db_insertion_on_action(monkeypatch, session):
    def mock_get_legal_actions(self, state):
        return [{"action_type": 1, "card_id": 123456}]
    
    from core.ygoenv.wrapper import YgoEngine
    monkeypatch.setattr(YgoEngine, "get_legal_actions", mock_get_legal_actions)

    from db.models import DuelStats
    # Create a dummy duel stats to satisfy foreign key
    stats = DuelStats(id=123, player_1_deck="deck1", player_2_deck="deck2")
    session.add(stats)
    session.commit()

    payload = {
        "duelId": "123",
        "gameState": {
            "currentPhase": 1,
            "turnPlayer": 0
        }
    }
    response = client.post("/api/v1/actions", json=payload)
    assert response.status_code == 200

    from db.models import GameTransition
    transition = session.query(GameTransition).filter(GameTransition.duel_id == 123).first()
    assert transition is not None
    assert transition.action["action_type"] == 1

def test_cors_local():
    headers = {
        "Origin": "http://localhost:8080",
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/api/v1/actions", headers=headers)
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
