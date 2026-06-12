from typing import Optional, Dict, Any
from sqlalchemy import String, Integer, JSON, ForeignKey, DateTime, func, Float
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base

class DuelStats(Base):
    __tablename__ = "duel_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_1_deck: Mapped[str] = mapped_column(String(255), nullable=False)
    player_2_deck: Mapped[str] = mapped_column(String(255), nullable=False)
    winner: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class GameTransition(Base):
    __tablename__ = "game_transitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    duel_id: Mapped[int] = mapped_column(ForeignKey("duel_stats.id", ondelete="CASCADE"), index=True)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    action: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    reward: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class Card(Base):
    __tablename__ = "cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # passcode
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(100), nullable=False)
    desc: Mapped[str] = mapped_column(String, nullable=False)
    race: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    archetype: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    atk: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    def_: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    level: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    attribute: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
