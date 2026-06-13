import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import DeckImportRequest, DeckImportResponse
from core.deck_parser import parse_deck, DeckParserError
from core.ygoenv.env import YgoEnv

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/v1/decks/import", response_model=DeckImportResponse)
async def import_deck(request: DeckImportRequest):
    try:
        parsed_deck = await parse_deck(request.deck_data)
        
        env = None
        try:
            env = YgoEnv()
            env.set_deck(parsed_deck)
        finally:
            if env is not None and hasattr(env, 'engine'):
                env.engine.destroy_duel()
        
        return DeckImportResponse(
            main=parsed_deck["main"],
            extra=parsed_deck["extra"],
            side=parsed_deck["side"]
        )
    except DeckParserError as e:
        logger.error(f"Deck parsing failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "DECK_PARSER_ERROR",
                    "detail": str(e)
                }
            }
        )
    except Exception as e:
        logger.exception("Unexpected error in import_deck")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "detail": "An unexpected error occurred"
                }
            }
        )
