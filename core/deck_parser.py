import httpx
from typing import Dict, List, Any

import os

class DeckParserError(Exception):
    pass

async def parse_deck(deck_data: str, parser_url: str = None) -> Dict[str, List[int]]:
    """
    Appelle le microservice omega-api-decks pour parser un deck.
    Retourne un dictionnaire avec 'main', 'extra', et 'side'.
    """
    if parser_url is None:
        parser_url = os.environ.get("YGO_PARSER_URL", "http://localhost:8000/api/v1/parse")
        
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                parser_url,
                json={"deck": deck_data},
                timeout=10.0
            )
            response.raise_for_status()
        data = response.json()
        
        if not isinstance(data, dict):
            raise DeckParserError("Parser returned invalid JSON format (expected dict)")
            
        return {
            "main": data.get("main") or [],
            "extra": data.get("extra") or [],
            "side": data.get("side") or []
        }
    except httpx.RequestError as e:
        raise DeckParserError(f"Network error while connecting to parser: {str(e)}")
    except httpx.HTTPStatusError as e:
        raise DeckParserError(f"Parser returned an HTTP error: {e.response.status_code}")
    except DeckParserError:
        raise
    except Exception as e:
        raise DeckParserError(f"Unexpected error parsing deck: {str(e)}")
