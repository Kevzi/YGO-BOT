from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )

class DuelStatsBase(CamelModel):
    player_1_deck: str = Field(max_length=255)
    player_2_deck: str = Field(max_length=255)

class DuelStatsCreate(DuelStatsBase):
    pass

class DuelStatsResponse(DuelStatsBase):
    id: int
    winner: Optional[int] = Field(default=None, ge=1, le=2)
    created_at: datetime
    updated_at: datetime

class GameTransitionBase(CamelModel):
    duel_id: int
    step: int = Field(ge=0)
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float = Field(default=0.0, allow_inf_nan=False)

class GameTransitionCreate(GameTransitionBase):
    pass

class GameTransitionResponse(GameTransitionBase):
    id: int
    created_at: datetime
