from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )

class DuelStatsBase(CamelModel):
    player_1_deck: str
    player_2_deck: str
    winner: Optional[int] = None

class DuelStatsCreate(DuelStatsBase):
    pass

class DuelStatsResponse(DuelStatsBase):
    id: int
    created_at: datetime
    updated_at: datetime

class GameTransitionBase(CamelModel):
    duel_id: int
    step: int
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float = 0.0

class GameTransitionCreate(GameTransitionBase):
    pass

class GameTransitionResponse(GameTransitionBase):
    id: int
    created_at: datetime
