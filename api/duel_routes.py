import logging
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from core.ygoenv.wrapper import YgoEngine, EngineCrashError
from ai.agent import DummyAgent
from db.session import get_db
from db.models import GameTransition

logger = logging.getLogger(__name__)

router = APIRouter()

# Single engine instance for the API
engine = YgoEngine()

def get_agent():
    return DummyAgent()

class GameState(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="allow")
    current_phase: int
    turn_player: int

class GameStateRequest(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    
    duel_id: str
    game_state: GameState

class ActionResponse(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    
    action_type: int
    card_id: Optional[int] = None

@router.post("/v1/actions", response_model=ActionResponse)
def fetch_legal_actions(request: GameStateRequest, db: Session = Depends(get_db), agent: DummyAgent = Depends(get_agent)):
    global engine
    try:
        try:
            duel_id_int = int(request.duel_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="duel_id must be an integer")
            
        state_dict = request.game_state.model_dump(by_alias=False)
        actions = engine.get_legal_actions(state_dict)
        
        if not actions:
            raise HTTPException(status_code=400, detail="No legal actions returned by engine")
        
        chosen_action = agent.select_action(actions)
        
        if "action_type" not in chosen_action:
            raise ValueError("action_type is missing from engine output")
            
        # Get next step counter
        step_count = db.query(GameTransition).filter(GameTransition.duel_id == duel_id_int).count()
            
        # Historisation de la transition en base de données
        transition = GameTransition(
            duel_id=duel_id_int,
            step=step_count + 1,
            state=state_dict,
            action=chosen_action
        )
        db.add(transition)
        db.commit()
            
        return ActionResponse(
            action_type=chosen_action["action_type"],
            card_id=chosen_action.get("card_id")
        )
    except EngineCrashError as e:
        logger.error(f"Engine crash detected: {e}")
        # Fail fast recovery: try recreating the engine
        try:
            engine = YgoEngine()
            logger.info("Engine instance recreated successfully.")
        except Exception as recreate_error:
            logger.critical(f"Failed to recreate engine after crash: {recreate_error}")

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "ENGINE_CRASH",
                    "detail": str(e)
                }
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in fetch_legal_actions")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "detail": "An unexpected error occurred"
                }
            }
        )
