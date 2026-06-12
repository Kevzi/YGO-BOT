import pytest
import asyncio
from unittest.mock import patch
from scripts.sync_ygoprodeck import fetch_cards, sync_cards_to_db
from db.models import Card
from db.session import SessionLocal

MOCK_API_RESPONSE = {
    "data": [
        {
            "id": 89631139,
            "name": "Blue-Eyes White Dragon",
            "type": "Normal Monster",
            "desc": "This legendary dragon is a powerful engine of destruction.",
            "atk": 3000,
            "def": 2500,
            "level": 8,
            "race": "Dragon",
            "attribute": "LIGHT",
            "archetype": "Blue-Eyes"
        },
        {
            "id": 46986414,
            "name": "Dark Magician",
            "type": "Normal Monster",
            "desc": "The ultimate wizard in terms of attack and defense.",
            "atk": 2500,
            "def": 2100,
            "level": 7,
            "race": "Spellcaster",
            "attribute": "DARK",
            "archetype": "Dark Magician"
        }
    ]
}

@pytest.mark.asyncio
async def test_fetch_cards(httpx_mock):
    # httpx_mock is provided by pytest-httpx
    from scripts.sync_ygoprodeck import API_URL
    httpx_mock.add_response(url=API_URL, json=MOCK_API_RESPONSE)
    
    cards = await fetch_cards()
    
    assert len(cards) == 2
    assert cards[0]["name"] == "Blue-Eyes White Dragon"
    assert cards[1]["id"] == 46986414

def test_sync_cards_to_db():
    sync_cards_to_db(MOCK_API_RESPONSE["data"])
    
    db = SessionLocal()
    try:
        be_card = db.query(Card).filter(Card.id == 89631139).first()
        assert be_card is not None
        assert be_card.name == "Blue-Eyes White Dragon"
        assert be_card.atk == 3000
        assert be_card.level == 8
        assert be_card.attribute == "LIGHT"
        
        dm_card = db.query(Card).filter(Card.id == 46986414).first()
        assert dm_card is not None
        assert dm_card.name == "Dark Magician"
    finally:
        db.close()
