from __future__ import annotations

"""Orquestra o pipeline diário completo: schedule -> lineup -> ataque agregado
(com encolhimento bayesiano de amostra pequena) -> pitching do titular
misturado com bullpen -> projeção de corridas -> Monte Carlo (incerteza de
parâmetro sobre Binomial Negativa) -> comparação com odds reais -> Value Bet
-> persistência.

Escopo desta etapa: cobre os insumos já implementados (ataque real via
lineup, pitching real do titular misturado com o bullpen, park factor real do
estádio, encolhimento bayesiano de wRC+/FIP por tamanho de amostra, camada de
Monte Carlo sobre Binomial Negativa, odds reais de Game Total E Team Total
por time, avaliação de Value Bet com Edge/EV/Kelly nos três mercados). Clima
e umpire ainda não existem — os campos correspondentes do relatório completo
previsto na especificação ficam de fora por enquanto e serão adicionados por
etapas futuras sem quebrar esta estrutura.

Team Total usa o endpoint por evento da The Odds API (1 crédito adicional por
jogo, além do bulk de Game Total) — só é buscado quando `game_odds.event_id`
está disponível (não fica no endpoint bulk).

A probabilidade projetada é calculada na LINHA REAL do mercado (consensus
total) quando há odds disponíveis para o jogo; cai para DEFAULT_TOTAL_LINE
apenas quando não há odds casadas. Comparar em linhas diferentes não faria
sentido estatístico.

Confiança (usada como um dos critérios de Value Bet) é, nesta etapa, a média
do confidence_score das lineups dos dois times (oficial=90, provável=40) —
um proxy simples e honesto, não o "Sistema de Score" completo da
especificação (que combina muitos outros fatores e é uma etapa futura maior).

Quando o bullpen de um time não pode ser determinado (team_id ausente), a
projeção cai de volta para usar apenas o FIP do titular — mesmo comportamento
de antes da integração do bullpen.

As odds de mercado são buscadas UMA VEZ por execução (uma chamada cobre o
slate do dia inteiro) e depois casadas por nome de time a cada jogo — cada
chamada à The Odds API consome créditos da cota mensal da chave, então evita-se
buscar por jogo individualmente. Se a busca de odds falhar (rede, cota
esgotada), o relatório continua normalmente, só sem os campos de mercado e
sem avaliação de Value Bet.

Um jogo é pulado (não gera projeção) quando faltam dados essenciais — ex.:
arremessador titular ainda não anunciado, ou nenhum jogador da lineup com
estatísticas de temporada disponíveis (início de temporada, recém-promovido).
O jogo ainda assim é registrado no banco (schedule), só a projeção fica ausente.

Alertas do Telegram (opt-in): se um `telegram_notifier` for injetado no
construtor, toda avaliação de Value Bet que `meets_criteria` dispara um
alerta automaticamente. Sem notifier (padrão), nenhuma mensagem é enviada —
evita que reruns manuais, backtests ou testes disparem mensagens reais no
canal. O `incremental_runner` (pipeline automático) injeta um notifier real;
como cada lote só é processado uma vez (rastreado no banco), cada jogo
dispara no máximo um alerta por mercado qualificado.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd

from mlb_quantitative_engine.analytics.monte_carlo import simulate_total_probability
from mlb_quantitative_engine.analytics.projections import OpposingPitcherInput, ProjectionEngine, TeamOffenseInput
from mlb_quantitative_engine.analytics.sabermetrics import BattingMetrics
from mlb_quantitative_engine.analytics.value_bet_calculator import (
    MIN_CONFIDENCE,
    evaluate_game_total_value_bets,
    evaluate_team_total_value_bets,
)
from mlb_quantitative_engine.api.mlb_api import GameSummary, MLBApiClient
from mlb_quantitative_engine.api.odds_api import OddsApiError
from mlb_quantitative_engine.database.repository import Repository
from mlb_quantitative_engine.models.game_projection import GameProjection
from mlb_quantitative_engine.models.value_bet import ValueBet
from mlb_quantitative_engine.services.bullpen_service import BullpenService, BullpenStatus
from mlb_quantitative_engine.services.lineup_service import LineupService, LineupSnapshot
from mlb_quantitative_engine.services.odds_service import GameOdds, OddsService
from mlb_quantitative_engine.services.offense_service import OffenseService
from mlb_quantitative_engine.services.park_factor_service import ParkFactorService
from mlb_quantitative_engine.services.pitching_service import PitchingService
from mlb_quantitative_engine.services.telegram_notifier import TelegramNotifier
from mlb_quantitative_engine.utils.logger import log

RETRY_DELAY = timedelta(minutes=30)


class ProjectionUnavailable(RuntimeError):
    """Levantado quando não há dados suficientes para projetar um jogo específico."""


@dataclass(frozen=True)
class _QuoteInputs:
    """Representação mínima de uma cotação Over/Under (Game Total ou Team Total),
    desacoplada de onde ela veio — busca ao vivo (`TotalsQuote`/`TeamTotalsQuote` da
    OddsService) ou reconstrução a partir de `ValueBet`s já persistidos (retentativa
    de lineup, que reaproveita odds sem gastar créditos novos)."""

    point: float
    over_price: float
    over_bookmaker: str
    under_price: float
    under_bookmaker: str

    @staticmethod
    def from_quote(quote: Any) -> "_QuoteInputs":
        return _QuoteInputs(
            point=quote.point,
            over_price=quote.over_price,
            over_bookmaker=quote.over_bookmaker,
            under_price=quote.under_price,
            under_bookmaker=quote.under_bookmaker,
        )


@dataclass(frozen=True)
class GameReportRow:
    """Uma linha do relatório diário: um jogo, com sua projeção e avaliação de Value Bet quando disponíveis."""

    game_pk: int
    game_date: str
    game_datetime: Optional[str]
    home_team: str
    away_team: str
    home_probable_pitcher: Optional[str]
    away_probable_pitcher: Optional[str]
    projected_home_runs: Optional[float] = None
    projected_away_runs: Optional[float] = None
    projected_total_runs: Optional[float] = None
    projected_probability_over: Optional[float] = None
    projected_probability_under: Optional[float] = None
    confidence_score: Optional[float] = None
    market_total_line: Optional[float] = None
    market_over_price: Optional[float] = None
    market_over_bookmaker: Optional[str] = None
    market_under_price: Optional[float] = None
    market_under_bookmaker: Optional[str] = None
    value_bet_recommendation: Optional[str] = None
    value_bet_minimum_price: Optional[float] = None
    value_bet_point: Optional[float] = None
    value_bet_edge: Optional[float] = None
    value_bet_expected_value: Optional[float] = None
    value_bet_suggested_stake_fraction: Optional[float] = None
    skip_reason: Optional[str] = None


class ReportGenerator:
    """Gera o relatório diário de projeções, persistindo jogos, projeções e value bets no banco."""

    DEFAULT_TOTAL_LINE = 8.5

    def __init__(
        self,
        api_client: Optional[MLBApiClient] = None,
        repository: Optional[Repository] = None,
        lineup_service: Optional[LineupService] = None,
        offense_service: Optional[OffenseService] = None,
        pitching_service: Optional[PitchingService] = None,
        park_factor_service: Optional[ParkFactorService] = None,
        bullpen_service: Optional[BullpenService] = None,
        odds_service: Optional[OddsService] = None,
        projection_engine: Optional[ProjectionEngine] = None,
        telegram_notifier: Optional[TelegramNotifier] = None,
        season: Optional[int] = None,
    ) -> None:
        self.api_client = api_client or MLBApiClient()
        self.repository = repository or Repository()
        self.lineup_service = lineup_service or LineupService(self.api_client)
        self.offense_service = offense_service or OffenseService(self.api_client)
        self.pitching_service = pitching_service or PitchingService(self.api_client)
        self.park_factor_service = park_factor_service or ParkFactorService()
        self.bullpen_service = bullpen_service or BullpenService(self.api_client)
        self.odds_service = odds_service or OddsService()
        self.projection_engine = projection_engine or ProjectionEngine()
        self.telegram_notifier = telegram_notifier  # opt-in: None -> nenhum alerta é enviado
        self.season = season or datetime.now(timezone.utc).year

    def generate_daily_report(self, date: Optional[str] = None) -> List[GameReportRow]:
        """Busca o schedule do dia e produz uma linha de relatório para cada jogo."""
        games = self.api_client.get_games_for_date(date)
        all_odds = self.fetch_all_odds()
        return [self.build_row(game, all_odds) for game in games]

    def fetch_all_odds(self) -> List[GameOdds]:
        try:
            return self.odds_service.get_all_game_odds()
        except OddsApiError as exc:
            log.warning(f"Não foi possível buscar odds de mercado: {exc}")
            return []

    def build_row(self, game: GameSummary, all_odds: List[GameOdds]) -> GameReportRow:
        self.repository.upsert_game(
            game_pk=game.game_pk,
            game_date=game.game_date,
            home_team=game.home_team,
            away_team=game.away_team,
            venue=game.venue,
            status=game.status,
            home_probable_pitcher=game.home_probable_pitcher,
            away_probable_pitcher=game.away_probable_pitcher,
            game_datetime=game.game_datetime,
        )

        game_odds = self.odds_service.find_game_odds(all_odds, game.home_team, game.away_team)

        base_kwargs = dict(
            game_pk=game.game_pk,
            game_date=game.game_date,
            game_datetime=game.game_datetime,
            home_team=game.home_team,
            away_team=game.away_team,
            home_probable_pitcher=game.home_probable_pitcher,
            away_probable_pitcher=game.away_probable_pitcher,
            **self._market_fields(game_odds),
        )

        if not game.home_probable_pitcher_id or not game.away_probable_pitcher_id:
            log.warning(f"Jogo {game.game_pk} sem arremessador titular definido; pulando projeção")
            return GameReportRow(**base_kwargs, skip_reason="arremessadores titulares indisponíveis")

        try:
            projection, confidence_score = self._project(game)
        except ProjectionUnavailable as exc:
            log.warning(f"Jogo {game.game_pk}: {exc}")
            return GameReportRow(**base_kwargs, skip_reason=str(exc))

        self._schedule_or_resolve_lineup_retry(game, confidence_score)

        game_row = self.repository.get_game_by_pk(game.game_pk)
        consensus = game_odds.consensus_total if game_odds else None
        game_quote = _QuoteInputs.from_quote(consensus) if consensus is not None else None

        home_team_quote: Optional[_QuoteInputs] = None
        away_team_quote: Optional[_QuoteInputs] = None
        if game_odds is not None and game_odds.event_id:
            team_totals = self.odds_service.get_team_totals(game_odds.event_id, game.home_team, game.away_team)
            if team_totals.home is not None:
                home_team_quote = _QuoteInputs.from_quote(team_totals.home)
            if team_totals.away is not None:
                away_team_quote = _QuoteInputs.from_quote(team_totals.away)

        total_line = game_quote.point if game_quote is not None else self.DEFAULT_TOTAL_LINE
        total_simulation = simulate_total_probability(projection.projected_total_runs, total_line)
        prob_over = total_simulation.probability_over
        prob_under = total_simulation.probability_under

        projection_row = self.repository.save_projection(
            game_id=game_row.id,
            projected_home_runs=projection.projected_home_runs,
            projected_away_runs=projection.projected_away_runs,
            projected_total_runs=projection.projected_total_runs,
            probability_over=prob_over,
            probability_under=prob_under,
            model_version="nb-shrinkage-mc-v1",
        )

        candidates = self._evaluate_candidates(
            game_pk=game.game_pk,
            home_team=game.home_team,
            away_team=game.away_team,
            projected_home_runs=projection.projected_home_runs,
            projected_away_runs=projection.projected_away_runs,
            projected_total_runs=projection.projected_total_runs,
            confidence_score=confidence_score,
            game_quote=game_quote,
            home_team_quote=home_team_quote,
            away_team_quote=away_team_quote,
        )
        for bet in candidates:
            saved_bet = self._persist_value_bet(projection_row.id, bet)
            self._maybe_send_telegram_alert(bet, saved_bet.id)
        value_bet_fields = self._best_value_bet_fields(candidates)

        return GameReportRow(
            **base_kwargs,
            projected_home_runs=projection.projected_home_runs,
            projected_away_runs=projection.projected_away_runs,
            projected_total_runs=projection.projected_total_runs,
            projected_probability_over=round(prob_over, 4),
            projected_probability_under=round(prob_under, 4),
            confidence_score=round(confidence_score, 1),
            **value_bet_fields,
        )

    def _project(self, game: GameSummary) -> Tuple[GameProjection, float]:
        home_lineup = self.lineup_service.get_batting_order(game.game_pk, "home")
        away_lineup = self.lineup_service.get_batting_order(game.game_pk, "away")
        confidence_score = (home_lineup.confidence_score + away_lineup.confidence_score) / 2.0

        home_offense = self._team_offense(home_lineup)
        away_offense = self._team_offense(away_lineup)
        if home_offense is None or away_offense is None:
            raise ProjectionUnavailable("estatísticas de ataque indisponíveis para uma das lineups")

        home_pitcher = self.pitching_service.get_pitching_metrics(game.home_probable_pitcher_id, self.season)
        away_pitcher = self.pitching_service.get_pitching_metrics(game.away_probable_pitcher_id, self.season)
        if home_pitcher is None or away_pitcher is None:
            raise ProjectionUnavailable("estatísticas de um dos arremessadores titulares indisponíveis")

        park_factor = self.park_factor_service.get_park_factors(game.venue).run_factor

        home_bullpen = self._bullpen_status(game.home_team_id, game.game_date)
        away_bullpen = self._bullpen_status(game.away_team_id, game.game_date)

        projection = self.projection_engine.project_game(
            home_team=game.home_team,
            away_team=game.away_team,
            home_offense=TeamOffenseInput(
                wrc_plus=home_offense.wrc_plus, plate_appearances=home_offense.plate_appearances
            ),
            away_offense=TeamOffenseInput(
                wrc_plus=away_offense.wrc_plus, plate_appearances=away_offense.plate_appearances
            ),
            home_starting_pitcher=self._opposing_pitcher_input(
                home_pitcher.fip, home_bullpen, home_pitcher.innings_pitched
            ),
            away_starting_pitcher=self._opposing_pitcher_input(
                away_pitcher.fip, away_bullpen, away_pitcher.innings_pitched
            ),
            park_factor=park_factor,
        )
        return projection, confidence_score

    def _schedule_or_resolve_lineup_retry(self, game: GameSummary, confidence_score: float) -> None:
        """Se a confiança da lineup ainda está abaixo do limiar de Value Bet
        (`MIN_CONFIDENCE`), agenda uma retentativa gratuita (só lineup) daqui a 30
        minutos; caso contrário, resolve qualquer retentativa pendente para o jogo —
        a lineup já está confiável o bastante e não precisa mais ser reconsultada."""
        if confidence_score < MIN_CONFIDENCE:
            retry_at = datetime.now(timezone.utc) + RETRY_DELAY
            self.repository.upsert_pending_lineup_retry(game.game_pk, game.game_date, retry_at=retry_at)
            log.info(
                f"Jogo {game.game_pk}: confiança {confidence_score:.1f} abaixo do limiar "
                f"({MIN_CONFIDENCE}); retentativa de lineup agendada para {retry_at.isoformat()}"
            )
        else:
            self.repository.mark_lineup_retry_resolved(game.game_pk)

    def _evaluate_candidates(
        self,
        game_pk: int,
        home_team: str,
        away_team: str,
        projected_home_runs: float,
        projected_away_runs: float,
        projected_total_runs: float,
        confidence_score: float,
        game_quote: Optional[_QuoteInputs],
        home_team_quote: Optional[_QuoteInputs],
        away_team_quote: Optional[_QuoteInputs],
    ) -> List[ValueBet]:
        """Avalia os candidatos de Value Bet (Game Total + Team Total dos dois times)
        a partir de cotações já resolvidas (`_QuoteInputs`) — usado tanto pelo fluxo
        normal (cotações recém-buscadas) quanto pela retentativa de lineup (cotações
        reconstruídas dos ValueBets já persistidos, sem gastar créditos novos)."""
        candidates: List[ValueBet] = []

        if game_quote is not None:
            game_over, game_under = evaluate_game_total_value_bets(
                game_pk=game_pk, home_team=home_team, away_team=away_team,
                projected_total_runs=projected_total_runs, point=game_quote.point,
                over_price=game_quote.over_price, over_bookmaker=game_quote.over_bookmaker,
                under_price=game_quote.under_price, under_bookmaker=game_quote.under_bookmaker,
                confidence_score=confidence_score,
            )
            candidates.extend([game_over, game_under])

        if home_team_quote is not None:
            home_over, home_under = evaluate_team_total_value_bets(
                game_pk=game_pk, home_team=home_team, away_team=away_team,
                team_label="home_team_total", projected_team_runs=projected_home_runs,
                point=home_team_quote.point, over_price=home_team_quote.over_price,
                over_bookmaker=home_team_quote.over_bookmaker, under_price=home_team_quote.under_price,
                under_bookmaker=home_team_quote.under_bookmaker, confidence_score=confidence_score,
            )
            candidates.extend([home_over, home_under])

        if away_team_quote is not None:
            away_over, away_under = evaluate_team_total_value_bets(
                game_pk=game_pk, home_team=home_team, away_team=away_team,
                team_label="away_team_total", projected_team_runs=projected_away_runs,
                point=away_team_quote.point, over_price=away_team_quote.over_price,
                over_bookmaker=away_team_quote.over_bookmaker, under_price=away_team_quote.under_price,
                under_bookmaker=away_team_quote.under_bookmaker, confidence_score=confidence_score,
            )
            candidates.extend([away_over, away_under])

        return candidates

    def _bullpen_status(self, team_id: Optional[int], reference_date: str) -> Optional[BullpenStatus]:
        if team_id is None:
            return None
        return self.bullpen_service.get_bullpen_status(team_id, reference_date=reference_date, season=self.season)

    @staticmethod
    def _opposing_pitcher_input(
        starter_fip: float, bullpen: Optional[BullpenStatus], starter_innings_pitched: Optional[float] = None
    ) -> OpposingPitcherInput:
        if bullpen is None or bullpen.metrics is None:
            return OpposingPitcherInput(starter_fip=starter_fip, starter_innings_pitched=starter_innings_pitched)
        return OpposingPitcherInput(
            starter_fip=starter_fip,
            bullpen_fip=bullpen.metrics.fip,
            bullpen_fatigue_index=bullpen.fatigue_index,
            bullpen_unavailable_count=bullpen.unavailable_count,
            starter_innings_pitched=starter_innings_pitched,
        )

    def _team_offense(self, lineup: LineupSnapshot) -> Optional[BattingMetrics]:
        if lineup.has_embedded_stats:
            return self.offense_service.get_team_offense_metrics_from_raw_stats(
                [entry.raw_batting_stats for entry in lineup.entries]
            )
        return self.offense_service.get_team_offense_metrics(lineup.player_ids, self.season)

    def _persist_value_bet(self, projection_id: int, value_bet: ValueBet):
        return self.repository.save_value_bet(
            projection_id=projection_id,
            market=value_bet.market,
            bookmaker=value_bet.bookmaker,
            price=value_bet.price,
            point=value_bet.point,
            projection_probability=value_bet.projected_probability,
            implied_probability_raw=value_bet.implied_probability_raw,
            implied_probability_fair=value_bet.implied_probability_fair,
            edge=value_bet.edge,
            expected_value=value_bet.expected_value,
            kelly_fraction=value_bet.kelly_fraction,
            kelly_fraction_quarter=value_bet.kelly_fraction_quarter,
            suggested_stake_fraction=value_bet.suggested_stake_fraction,
            minimum_acceptable_price=value_bet.minimum_acceptable_price,
            confidence_score=value_bet.confidence_score,
            meets_criteria=value_bet.meets_criteria,
        )

    def _maybe_send_telegram_alert(self, value_bet: ValueBet, value_bet_id: int) -> None:
        """Envia um alerta ao Telegram se houver notifier injetado e a aposta qualificar.

        Marca `alert_sent=True` no registro persistido somente após um envio
        bem-sucedido — é esse flag que o verificador de resultados (GREEN/RED/PUSH,
        ver services/bet_result_checker.py) usa para saber quais apostas realmente
        chegaram ao canal e portanto precisam de um resultado, evitando notificar o
        resultado de algo que nunca foi anunciado como recomendação.

        Falhas de envio (rede, canal errado, etc.) são logadas e engolidas — um
        problema no Telegram não deve derrubar o processamento do resto do lote.
        """
        if self.telegram_notifier is None or not value_bet.meets_criteria:
            return
        try:
            self.telegram_notifier.send_value_bet_alert(value_bet)
        except Exception as exc:  # noqa: BLE001 - notificação é best-effort, nunca deve propagar
            log.error(f"Falha ao enviar alerta ao Telegram para {value_bet.market}: {exc}")
            return
        self.repository.mark_alert_sent(value_bet_id)

    @staticmethod
    def _best_value_bet_fields(candidates: List[ValueBet]) -> dict:
        """Escolhe, entre todas as pontas avaliadas (Game Total e Team Total dos dois
        times), qual destacar na linha do relatório.

        Prioriza qualquer candidato que satisfaça os critérios de Value Bet
        (EV > 5%, Edge > 4%, Confiança > 70%); se nenhum qualificar, ainda
        expõe os números do candidato com maior EV, para transparência — sem
        emitir uma recomendação formal (`value_bet_recommendation` fica None).
        """
        if not candidates:
            return {}

        qualifying = [bet for bet in candidates if bet.meets_criteria]
        if qualifying:
            best = max(qualifying, key=lambda bet: bet.expected_value)
            recommendation = best.market
        else:
            best = max(candidates, key=lambda bet: bet.expected_value)
            recommendation = None

        return {
            "value_bet_recommendation": recommendation,
            "value_bet_minimum_price": best.minimum_acceptable_price,
            "value_bet_point": best.point,
            "value_bet_edge": best.edge,
            "value_bet_expected_value": best.expected_value,
            "value_bet_suggested_stake_fraction": best.suggested_stake_fraction,
        }

    @staticmethod
    def _market_fields(game_odds: Optional[GameOdds]) -> dict:
        consensus = game_odds.consensus_total if game_odds else None
        quote = _QuoteInputs.from_quote(consensus) if consensus is not None else None
        return ReportGenerator._market_fields_from_quote(quote)

    @staticmethod
    def _market_fields_from_quote(quote: Optional[_QuoteInputs]) -> dict:
        if quote is None:
            return {
                "market_total_line": None,
                "market_over_price": None,
                "market_over_bookmaker": None,
                "market_under_price": None,
                "market_under_bookmaker": None,
            }
        return {
            "market_total_line": quote.point,
            "market_over_price": quote.over_price,
            "market_over_bookmaker": quote.over_bookmaker,
            "market_under_price": quote.under_price,
            "market_under_bookmaker": quote.under_bookmaker,
        }

    @staticmethod
    def _quote_from_cached_bets(bets: Sequence[Any], market_prefix: str) -> Optional[_QuoteInputs]:
        """Reconstrói uma cotação Over/Under a partir dos ValueBets já persistidos de
        uma projeção anterior — usada pela retentativa de lineup para reaproveitar
        odds já pagas em vez de rebuscá-las (`market_prefix` ex.: "game_total",
        "home_team_total", "away_team_total")."""
        over = next((bet for bet in bets if bet.market == f"{market_prefix}_over"), None)
        under = next((bet for bet in bets if bet.market == f"{market_prefix}_under"), None)
        if over is None or under is None:
            return None
        return _QuoteInputs(
            point=over.point,
            over_price=over.price,
            over_bookmaker=over.bookmaker,
            under_price=under.price,
            under_bookmaker=under.bookmaker,
        )

    def retry_game(self, game_pk: int, now: Optional[datetime] = None) -> Optional[GameReportRow]:
        """Reavalia um jogo cuja lineup ainda não estava oficial na última passada.

        Rebusca só o schedule/lineup (endpoints gratuitos da MLB API) e reaproveita
        as odds já persistidas (ValueBets da última projeção deste jogo) como fonte
        de preços — não gasta créditos novos da The Odds API. Se a confiança
        continuar baixa, reagenda outra retentativa em 30 minutos; se o jogo já
        tiver começado, desiste e resolve a retentativa pendente.
        """
        now = now or datetime.now(timezone.utc)
        game_row = self.repository.get_game_by_pk(game_pk)
        if game_row is None:
            log.warning(f"retry_game: jogo {game_pk} não encontrado no banco")
            return None

        games = self.api_client.get_games_for_date(game_row.game_date)
        game = next((candidate for candidate in games if candidate.game_pk == game_pk), None)
        if game is None:
            log.warning(f"retry_game: jogo {game_pk} não encontrado no schedule de {game_row.game_date}")
            self.repository.mark_lineup_retry_resolved(game_pk)
            return None

        if game.game_datetime:
            start_time = datetime.fromisoformat(game.game_datetime.replace("Z", "+00:00"))
            if now >= start_time:
                log.info(f"retry_game: jogo {game_pk} já começou; encerrando retentativas de lineup")
                self.repository.mark_lineup_retry_resolved(game_pk)
                return None

        if not game.home_probable_pitcher_id or not game.away_probable_pitcher_id:
            log.info(f"retry_game: jogo {game_pk} ainda sem arremessadores titulares; reagendando")
            self.repository.upsert_pending_lineup_retry(game_pk, game_row.game_date, retry_at=now + RETRY_DELAY)
            return None

        try:
            projection, confidence_score = self._project(game)
        except ProjectionUnavailable as exc:
            log.warning(f"retry_game: jogo {game_pk}: {exc}; reagendando")
            self.repository.upsert_pending_lineup_retry(game_pk, game_row.game_date, retry_at=now + RETRY_DELAY)
            return None

        last_projections = self.repository.list_projections_for_game(game_row.id)
        cached_bets = (
            self.repository.list_value_bets_for_projection(last_projections[0].id) if last_projections else []
        )
        game_quote = self._quote_from_cached_bets(cached_bets, "game_total")
        home_team_quote = self._quote_from_cached_bets(cached_bets, "home_team_total")
        away_team_quote = self._quote_from_cached_bets(cached_bets, "away_team_total")

        total_line = game_quote.point if game_quote is not None else self.DEFAULT_TOTAL_LINE
        total_simulation = simulate_total_probability(projection.projected_total_runs, total_line)
        prob_over = total_simulation.probability_over
        prob_under = total_simulation.probability_under

        projection_row = self.repository.save_projection(
            game_id=game_row.id,
            projected_home_runs=projection.projected_home_runs,
            projected_away_runs=projection.projected_away_runs,
            projected_total_runs=projection.projected_total_runs,
            probability_over=prob_over,
            probability_under=prob_under,
            model_version="nb-shrinkage-mc-v1-retry",
        )

        candidates = self._evaluate_candidates(
            game_pk=game.game_pk,
            home_team=game.home_team,
            away_team=game.away_team,
            projected_home_runs=projection.projected_home_runs,
            projected_away_runs=projection.projected_away_runs,
            projected_total_runs=projection.projected_total_runs,
            confidence_score=confidence_score,
            game_quote=game_quote,
            home_team_quote=home_team_quote,
            away_team_quote=away_team_quote,
        )
        for bet in candidates:
            saved_bet = self._persist_value_bet(projection_row.id, bet)
            self._maybe_send_telegram_alert(bet, saved_bet.id)
        value_bet_fields = self._best_value_bet_fields(candidates)

        if confidence_score >= MIN_CONFIDENCE:
            self.repository.mark_lineup_retry_resolved(game_pk)
        else:
            self.repository.upsert_pending_lineup_retry(game_pk, game_row.game_date, retry_at=now + RETRY_DELAY)

        self.repository.upsert_game(
            game_pk=game.game_pk,
            game_date=game.game_date,
            home_team=game.home_team,
            away_team=game.away_team,
            venue=game.venue,
            status=game.status,
            home_probable_pitcher=game.home_probable_pitcher,
            away_probable_pitcher=game.away_probable_pitcher,
            game_datetime=game.game_datetime,
        )

        return GameReportRow(
            game_pk=game.game_pk,
            game_date=game.game_date,
            game_datetime=game.game_datetime,
            home_team=game.home_team,
            away_team=game.away_team,
            home_probable_pitcher=game.home_probable_pitcher,
            away_probable_pitcher=game.away_probable_pitcher,
            projected_home_runs=projection.projected_home_runs,
            projected_away_runs=projection.projected_away_runs,
            projected_total_runs=projection.projected_total_runs,
            projected_probability_over=round(prob_over, 4),
            projected_probability_under=round(prob_under, 4),
            confidence_score=round(confidence_score, 1),
            **self._market_fields_from_quote(game_quote),
            **value_bet_fields,
        )


def rows_to_dataframe(rows: List[GameReportRow]) -> pd.DataFrame:
    """Converte as linhas do relatório em um DataFrame pandas, pronto para exibição/exportação."""
    return pd.DataFrame([asdict(row) for row in rows])
