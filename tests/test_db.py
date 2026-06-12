from db.models import DuelStats, GameTransition
from schemas.duel import DuelStatsCreate, GameTransitionCreate

def test_duel_stats_model(session):
    duel = DuelStats(player_1_deck="deck_1", player_2_deck="deck_2")
    session.add(duel)
    session.commit()
    
    assert duel.id is not None
    assert duel.player_1_deck == "deck_1"
    assert duel.player_2_deck == "deck_2"
    assert duel.winner is None

def test_game_transition_model(session):
    duel = DuelStats(player_1_deck="deck_1", player_2_deck="deck_2")
    session.add(duel)
    session.commit()
    
    transition = GameTransition(
        duel_id=duel.id,
        step=1,
        state={"field": "empty"},
        action={"play": "card"},
        reward=1.0
    )
    session.add(transition)
    session.commit()
    
    assert transition.id is not None
    assert transition.duel_id == duel.id
    assert transition.step == 1
    assert transition.reward == 1.0

def test_pydantic_camel_case_alias():
    # Test that we can create from camelCase
    data = {
        "player1Deck": "Blue-Eyes",
        "player2Deck": "Dark Magician"
    }
    duel_schema = DuelStatsCreate(**data)
    assert duel_schema.player_1_deck == "Blue-Eyes"
    assert duel_schema.player_2_deck == "Dark Magician"

    # Test serialization to camelCase
    dumped = duel_schema.model_dump(by_alias=True)
    assert "player1Deck" in dumped
    assert dumped["player1Deck"] == "Blue-Eyes"
