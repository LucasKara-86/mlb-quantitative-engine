from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from mlb_quantitative_engine.config import settings
from mlb_quantitative_engine.database.models import (
    Base,
    Game,
    PendingLineupRetry,
    ProcessedBatch,
    Projection,
    ValueBet,
)
from mlb_quantitative_engine.utils.logger import log


class Repository:
    """Camada de acesso a dados: cria o schema e expõe as operações de CRUD do domínio.

    Baseada em SQLAlchemy para que a troca futura de SQLite por PostgreSQL
    (prevista na arquitetura do projeto) exija apenas trocar a connection string.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        path = Path(db_path or settings.database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{path}", future=True)
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False, future=True)
        Base.metadata.create_all(self._engine)
        log.debug(f"Repository inicializado com banco em: {path}")

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # --- Games ---

    def upsert_game(
        self,
        game_pk: int,
        game_date: str,
        home_team: str,
        away_team: str,
        venue: Optional[str] = None,
        status: Optional[str] = None,
        home_probable_pitcher: Optional[str] = None,
        away_probable_pitcher: Optional[str] = None,
        game_datetime: Optional[str] = None,
    ) -> Game:
        """Cria o jogo se ele ainda não existir (por game_pk), ou atualiza os campos mutáveis."""
        with self.session() as session:
            existing = session.query(Game).filter_by(game_pk=game_pk).one_or_none()
            if existing is not None:
                existing.game_date = game_date
                existing.game_datetime = game_datetime
                existing.home_team = home_team
                existing.away_team = away_team
                existing.venue = venue
                existing.status = status
                existing.home_probable_pitcher = home_probable_pitcher
                existing.away_probable_pitcher = away_probable_pitcher
                session.flush()
                session.refresh(existing)
                return existing

            game = Game(
                game_pk=game_pk,
                game_date=game_date,
                game_datetime=game_datetime,
                home_team=home_team,
                away_team=away_team,
                venue=venue,
                status=status,
                home_probable_pitcher=home_probable_pitcher,
                away_probable_pitcher=away_probable_pitcher,
            )
            session.add(game)
            session.flush()
            session.refresh(game)
            return game

    def get_game_by_pk(self, game_pk: int) -> Optional[Game]:
        with self.session() as session:
            return session.query(Game).filter_by(game_pk=game_pk).one_or_none()

    def list_games_by_date(self, game_date: str) -> Sequence[Game]:
        with self.session() as session:
            return session.query(Game).filter_by(game_date=game_date).all()

    # --- Projections ---

    def save_projection(
        self,
        game_id: int,
        projected_home_runs: float,
        projected_away_runs: float,
        projected_total_runs: float,
        probability_over: float,
        probability_under: float,
        model_version: Optional[str] = None,
    ) -> Projection:
        with self.session() as session:
            projection = Projection(
                game_id=game_id,
                projected_home_runs=projected_home_runs,
                projected_away_runs=projected_away_runs,
                projected_total_runs=projected_total_runs,
                probability_over=probability_over,
                probability_under=probability_under,
                model_version=model_version,
            )
            session.add(projection)
            session.flush()
            session.refresh(projection)
            return projection

    def list_projections_for_game(self, game_id: int) -> Sequence[Projection]:
        with self.session() as session:
            return (
                session.query(Projection)
                .filter_by(game_id=game_id)
                .order_by(Projection.created_at.desc())
                .all()
            )

    # --- Value Bets ---

    def save_value_bet(
        self,
        projection_id: int,
        market: str,
        bookmaker: str,
        price: float,
        point: float,
        projection_probability: float,
        implied_probability_raw: float,
        implied_probability_fair: float,
        edge: float,
        expected_value: float,
        kelly_fraction: float,
        kelly_fraction_quarter: float,
        suggested_stake_fraction: float,
        minimum_acceptable_price: float,
        confidence_score: float,
        meets_criteria: bool,
        alert_sent: bool = False,
        result_notified: bool = False,
    ) -> ValueBet:
        with self.session() as session:
            value_bet = ValueBet(
                projection_id=projection_id,
                market=market,
                bookmaker=bookmaker,
                price=price,
                point=point,
                projection_probability=projection_probability,
                implied_probability_raw=implied_probability_raw,
                implied_probability_fair=implied_probability_fair,
                edge=edge,
                expected_value=expected_value,
                kelly_fraction=kelly_fraction,
                kelly_fraction_quarter=kelly_fraction_quarter,
                suggested_stake_fraction=suggested_stake_fraction,
                minimum_acceptable_price=minimum_acceptable_price,
                confidence_score=confidence_score,
                meets_criteria=meets_criteria,
                alert_sent=alert_sent,
                result_notified=result_notified,
            )
            session.add(value_bet)
            session.flush()
            session.refresh(value_bet)
            return value_bet

    def list_value_bets(self, meets_criteria_only: bool = False) -> Sequence[ValueBet]:
        with self.session() as session:
            query = session.query(ValueBet)
            if meets_criteria_only:
                query = query.filter_by(meets_criteria=True)
            return query.order_by(ValueBet.created_at.desc()).all()

    def list_value_bets_for_projection(self, projection_id: int) -> Sequence[ValueBet]:
        with self.session() as session:
            return (
                session.query(ValueBet)
                .filter_by(projection_id=projection_id)
                .order_by(ValueBet.created_at.desc())
                .all()
            )

    def mark_alert_sent(self, value_bet_id: int) -> None:
        with self.session() as session:
            session.query(ValueBet).filter_by(id=value_bet_id).update({"alert_sent": True})

    def mark_result_notified(self, value_bet_id: int) -> None:
        with self.session() as session:
            session.query(ValueBet).filter_by(id=value_bet_id).update({"result_notified": True})

    def mark_bet_outcome(self, value_bet_id: int, outcome: str) -> None:
        """Marca uma aposta como resolvida (`outcome`: "win"/"loss"/"push") e notificada
        — chamado pelo bet_result_checker depois de enviar o GREEN/RED/PUSH ao Telegram.
        `outcome` é o que alimenta o harness de calibração (analytics/calibration.py)."""
        with self.session() as session:
            session.query(ValueBet).filter_by(id=value_bet_id).update(
                {"outcome": outcome, "result_notified": True}
            )

    def list_resolved_value_bets(self) -> Sequence[ValueBet]:
        """Todas as apostas com resultado conhecido (`outcome` preenchido) — a base de
        dados usada pelo harness de calibração."""
        with self.session() as session:
            return session.query(ValueBet).filter(ValueBet.outcome.isnot(None)).all()

    def has_alert_been_sent(self, game_pk: int, market: str) -> bool:
        """Diz se já existe um ValueBet com `alert_sent=True` para esse jogo+mercado,
        em QUALQUER projeção (não só a mais recente) — usado para nunca reenviar a
        mesma recomendação duas vezes, mesmo que o jogo seja reavaliado de novo
        (retentativa de lineup, reprocessamento por atraso de agendamento etc.)."""
        with self.session() as session:
            return (
                session.query(ValueBet)
                .join(Projection, ValueBet.projection_id == Projection.id)
                .join(Game, Projection.game_id == Game.id)
                .filter(Game.game_pk == game_pk, ValueBet.market == market, ValueBet.alert_sent.is_(True))
                .first()
                is not None
            )

    def list_bets_pending_result_check(self) -> Sequence[Tuple[ValueBet, Game]]:
        """Retorna (ValueBet, Game) de cada aposta cujo alerta já foi enviado ao Telegram
        mas cujo resultado (GREEN/RED/PUSH) ainda não foi notificado. Uma vez notificado
        (`mark_result_notified`), a aposta some desta lista para sempre — é assim que se
        evita reenviar o mesmo resultado."""
        with self.session() as session:
            return (
                session.query(ValueBet, Game)
                .join(Projection, ValueBet.projection_id == Projection.id)
                .join(Game, Projection.game_id == Game.id)
                .filter(ValueBet.alert_sent.is_(True), ValueBet.result_notified.is_(False))
                .all()
            )

    # --- Lotes de atualização incremental (batch_scheduling) ---

    def is_batch_processed(self, game_date: str, anchor_time: datetime) -> bool:
        with self.session() as session:
            return (
                session.query(ProcessedBatch)
                .filter_by(game_date=game_date, anchor_time=anchor_time)
                .first()
                is not None
            )

    def mark_batch_processed(self, game_date: str, anchor_time: datetime) -> None:
        with self.session() as session:
            session.add(ProcessedBatch(game_date=game_date, anchor_time=anchor_time))

    # --- Retentativas de lineup (quando a confiança fica baixa demais) ---

    def upsert_pending_lineup_retry(self, game_pk: int, game_date: str, retry_at: datetime) -> PendingLineupRetry:
        """Agenda (ou reagenda) uma retentativa para um jogo. Só existe uma retentativa
        ativa (não resolvida) por jogo por vez — reagendar substitui o horário anterior."""
        with self.session() as session:
            existing = (
                session.query(PendingLineupRetry)
                .filter_by(game_pk=game_pk, resolved=False)
                .one_or_none()
            )
            if existing is not None:
                existing.retry_at = retry_at
                session.flush()
                session.refresh(existing)
                return existing

            retry = PendingLineupRetry(game_pk=game_pk, game_date=game_date, retry_at=retry_at)
            session.add(retry)
            session.flush()
            session.refresh(retry)
            return retry

    def list_due_lineup_retries(self, now: datetime) -> Sequence[PendingLineupRetry]:
        with self.session() as session:
            return (
                session.query(PendingLineupRetry)
                .filter(PendingLineupRetry.resolved.is_(False), PendingLineupRetry.retry_at <= now)
                .all()
            )

    def mark_lineup_retry_resolved(self, game_pk: int) -> None:
        with self.session() as session:
            session.query(PendingLineupRetry).filter_by(game_pk=game_pk, resolved=False).update(
                {"resolved": True}
            )
