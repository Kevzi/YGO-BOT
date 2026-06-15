import logging
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from core.ygoenv.wrapper import YgoEngine, EngineCrashError
from ai.inference import PPOInferenceAgent
from db.session import get_db
from db.models import GameTransition

logger = logging.getLogger(__name__)

router = APIRouter()

# Single engine instance for the API
engine = YgoEngine()

def get_agent():
    return PPOInferenceAgent()

class GameState(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="allow")
    action_type: str
    data: Dict[str, Any]

class GameStateRequest(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    
    duel_id: str
    game_state: GameState

class ActionResponse(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    
    action_type: str
    action_idx: int
    card_id: Optional[int] = None

@router.post("/v1/actions", response_model=ActionResponse)
def fetch_legal_actions(request: GameStateRequest, db: Session = Depends(get_db), agent: PPOInferenceAgent = Depends(get_agent)):
    global engine
    try:
        try:
            duel_id_int = int(request.duel_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="duel_id must be an integer")
            
        state_dict = request.game_state.model_dump(by_alias=False)
        
        # Determine if payload is from OmegaBotWrapper Actions.cs
        if "action_type" in state_dict:
            # Bypass engine and construct actions directly from Omega JSON
            action_type = state_dict["action_type"]
            omega_data = state_dict.get("data", {})
            choices = []
            idx_counter = 0
            
            if action_type == "OnIdleCmd":
                for card in omega_data.get("SummonableCards", []):
                    choices.append({"type": "Summon", "action_idx": idx_counter, "code": card.get("Code", 0)})
                    idx_counter += 1
                for card in omega_data.get("SpecialSummonableCards", []):
                    choices.append({"type": "SpSummon", "action_idx": idx_counter, "code": card.get("Code", 0)})
                    idx_counter += 1
                for card in omega_data.get("ActivableCards", []):
                    choices.append({"type": "Activate", "action_idx": idx_counter, "code": card.get("Code", 0)})
                    idx_counter += 1
                for card in omega_data.get("MonsterSetableCards", []):
                    choices.append({"type": "MonsterSet", "action_idx": idx_counter, "code": card.get("Code", 0)})
                    idx_counter += 1
                for card in omega_data.get("SpellSetableCards", []):
                    choices.append({"type": "SpellSet", "action_idx": idx_counter, "code": card.get("Code", 0)})
                    idx_counter += 1
                for card in omega_data.get("ReposableCards", []):
                    choices.append({"type": "Reposition", "action_idx": idx_counter, "code": card.get("Code", 0)})
                    idx_counter += 1
                if omega_data.get("CanBattlePhase"):
                    choices.append({"type": "ToBattlePhase", "action_idx": idx_counter})
                    idx_counter += 1
                if omega_data.get("CanEndPhase", True):
                    choices.append({"type": "ToEndPhase", "action_idx": idx_counter})
                    idx_counter += 1
                actions = [{"msg": "MSG_SELECT_IDLECMD", "choices": choices}]
                
            elif action_type == "OnBattleCmd":
                for card in omega_data.get("AttackableCards", []):
                    choices.append({"type": "Attack", "action_idx": idx_counter, "code": card.get("Code", 0)})
                    idx_counter += 1
                for card in omega_data.get("ActivableCards", []):
                    choices.append({"type": "Activate", "action_idx": idx_counter, "code": card.get("Code", 0)})
                    idx_counter += 1
                if omega_data.get("CanMainPhaseTwo"):
                    choices.append({"type": "ToMainPhaseTwo", "action_idx": idx_counter})
                    idx_counter += 1
                if omega_data.get("CanEndPhase", True):
                    choices.append({"type": "ToEndPhase", "action_idx": idx_counter})
                    idx_counter += 1
                actions = [{"msg": "MSG_SELECT_BATTLECMD", "choices": choices}]
                
            elif action_type == "OnSelectCard":
                choices.append({"type": "SelectCard", "action_idx": 0})
                actions = [{"msg": "MSG_SELECT_CARD", "choices": choices}]
                
            elif action_type == "OnSelectChain":
                if not omega_data.get("Forced"):
                    choices.append({"type": "Skip", "action_idx": idx_counter})
                    idx_counter += 1
                choices.append({"type": "Activate", "action_idx": idx_counter})
                idx_counter += 1
                actions = [{"msg": "MSG_SELECT_CHAIN", "choices": choices}]
            
            logger.info(f"Omega [{action_type}] -> {len(choices)} choices: {[c['type'] for c in choices]}")
        else:
            # Fallback to YgoEngine logic if standard payload
            actions = engine.get_legal_actions(state_dict)
        
        if not actions:
            raise HTTPException(status_code=400, detail="No legal actions available")
        
        chosen_action = agent.select_action(actions, engine, state_dict)
        
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
            action_idx=chosen_action.get("action_idx", -1),
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
