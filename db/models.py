from typing import Optional, Dict, Any
from sqlalchemy import String, Integer, JSON, ForeignKey, DateTime, func, Float
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base

class DuelStats(Base):
    __tablename__ = "duel_stats"

    id: Mapped[int] = mapped_column(primary_key=True)
    player_1_deck: Mapped[str] = mapped_column(String(255))
    player_2_deck: Mapped[str] = mapped_column(String(255))
    winner: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

class GameTransition(Base):
    __tablename__ = "game_transitions"

    id: Mapped[int] = mapped_column(primary_key=True)
    duel_id: Mapped[int] = mapped_column(ForeignKey("duel_stats.id"))
    step: Mapped[int] = mapped_column(Integer)
    state: Mapped[Dict[str, Any]] = mapped_column(JSON)
    action: Mapped[Dict[str, Any]] = mapped_column(JSON)
    reward: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())
