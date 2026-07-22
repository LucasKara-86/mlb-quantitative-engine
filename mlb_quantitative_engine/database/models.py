from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarativa para todos os modelos ORM do projeto."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Game(Base):
    """Um jogo da MLB agendado para uma data específica."""

    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(Integer, unique=True, nullable=False, index=True)
    game_date: Mapped[str] = mapped_column(String, nullable=False, index=True)
    game_datetime: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    home_team: Mapped[str] = mapped_column(String, nullable=False)
    away_team: Mapped[str] = mapped_column(String, nullable=False)
    venue: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    home_probable_pitcher: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    away_probable_pitcher: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    projections: Mapped[List["Projection"]] = relationship(
        back_populates="game", cascade="all, delete-orphan"
    )


class Projection(Base):
    """Projeção quantitativa de corridas geradas pelo motor analítico para um jogo."""

    __tablename__ = "projections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), nullable=False, index=True)
    projected_home_runs: Mapped[float] = mapped_column(Float, nullable=False)
    projected_away_runs: Mapped[float] = mapped_column(Float, nullable=False)
    projected_total_runs: Mapped[float] = mapped_column(Float, nullable=False)
    probability_over: Mapped[float] = mapped_column(Float, nullable=False)
    probability_under: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    game: Mapped["Game"] = relationship(back_populates="projections")
    value_bets: Mapped[List["ValueBet"]] = relationship(
        back_populates="projection", cascade="all, delete-orphan"
    )


class ValueBet(Base):
    """Avaliação de uma aposta específica (Game Total Over/Under), comparando a probabilidade
    projetada pelo modelo com a probabilidade implícita (justa, sem vig) do mercado."""

    __tablename__ = "value_bets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    projection_id: Mapped[int] = mapped_column(ForeignKey("projections.id"), nullable=False, index=True)
    market: Mapped[str] = mapped_column(String, nullable=False)
    bookmaker: Mapped[str] = mapped_column(String, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    point: Mapped[float] = mapped_column(Float, nullable=False)
    projection_probability: Mapped[float] = mapped_column(Float, nullable=False)
    implied_probability_raw: Mapped[float] = mapped_column(Float, nullable=False)
    implied_probability_fair: Mapped[float] = mapped_column(Float, nullable=False)
    edge: Mapped[float] = mapped_column(Float, nullable=False)
    expected_value: Mapped[float] = mapped_column(Float, nullable=False)
    kelly_fraction: Mapped[float] = mapped_column(Float, nullable=False)
    kelly_fraction_quarter: Mapped[float] = mapped_column(Float, nullable=False)
    suggested_stake_fraction: Mapped[float] = mapped_column(Float, nullable=False)
    minimum_acceptable_price: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    meets_criteria: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    alert_sent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    result_notified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    outcome: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # "win" | "loss" | "push"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    projection: Mapped["Projection"] = relationship(back_populates="value_bets")


class ProcessedBatch(Base):
    """Marca um lote de atualização incremental (ver analytics/batch_scheduling.py) como
    já processado, para o cron não buscar dados de novo para o mesmo lote."""

    __tablename__ = "processed_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_date: Mapped[str] = mapped_column(String, nullable=False, index=True)
    anchor_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ParameterChangeLog(Base):
    """Histórico de cada ajuste de parâmetro tentado por services/auto_tuning_service.py --
    aplicado (passou no gate de testes e foi commitado) ou revertido (falhou o gate). É o
    que dá o "cooldown" (não reajustar o mesmo parâmetro cedo demais) e o que aparece na
    seção "mudanças recentes em observação" do relatório diário do Telegram."""

    __tablename__ = "parameter_change_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parameter_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    old_value: Mapped[float] = mapped_column(Float, nullable=False)
    new_value: Mapped[float] = mapped_column(Float, nullable=False)
    rationale: Mapped[str] = mapped_column(String, nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    applied: Mapped[bool] = mapped_column(Boolean, nullable=False)  # False = tentado, revertido (testes falharam)
    git_commit_sha: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class PendingLineupRetry(Base):
    """Um jogo processado com lineup ainda não oficial (confiança baixa) — agenda uma
    nova tentativa que rebusca só a lineup (gratuita) e recalcula com as odds já
    obtidas (persistidas nos ValueBet da última projeção), sem gastar créditos de novo."""

    __tablename__ = "pending_lineup_retries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    game_date: Mapped[str] = mapped_column(String, nullable=False, index=True)
    retry_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
