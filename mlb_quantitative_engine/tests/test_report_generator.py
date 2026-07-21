from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pytest

from mlb_quantitative_engine.analytics.sabermetrics import BattingMetrics, PitchingMetrics
from mlb_quantitative_engine.api.mlb_api import GameSummary
from mlb_quantitative_engine.database.repository import Repository
from mlb_quantitative_engine.reports.report_generator import ReportGenerator, rows_to_dataframe
from mlb_quantitative_engine.services.bullpen_service import BullpenStatus
from mlb_quantitative_engine.services.lineup_service import LineupEntry, LineupSnapshot


def _game_summary(
    game_pk: int = 1,
    home_pitcher_id: Optional[int] = 100,
    away_pitcher_id: Optional[int] = 200,
    venue: str = "Some Park",
    game_datetime: str = "2026-07-17T23:05:00Z",
) -> GameSummary:
    return GameSummary(
        game_pk=game_pk,
        game_date="2026-07-17",
        game_datetime=game_datetime,
        home_team="Home Team",
        away_team="Away Team",
        venue=venue,
        status="Scheduled",
        home_probable_pitcher="Home Pitcher",
        away_probable_pitcher="Away Pitcher",
        home_probable_pitcher_id=home_pitcher_id,
        away_probable_pitcher_id=away_pitcher_id,
        home_team_id=1001,
        away_team_id=1002,
    )


def _batting_metrics(wrc_plus: float, plate_appearances: int = 3000) -> BattingMetrics:
    """`plate_appearances` grande por padrão (amostra de temporada completa da lineup
    inteira) para que o encolhimento bayesiano (ProjectionEngine) não distorça os
    cenários de teste, que usam wrc_plus diretamente para simular ataques fortes/fracos."""
    return BattingMetrics(
        avg=0.260, obp=0.330, slg=0.420, ops=0.750, iso=0.160,
        babip=0.300, woba=0.330, wrc_plus=wrc_plus, runs_created=80.0,
        plate_appearances=plate_appearances,
    )


def _pitching_metrics(fip: float, innings_pitched: float = 150.0) -> PitchingMetrics:
    """`innings_pitched` grande por padrão (amostra de temporada completa) pelo mesmo
    motivo de `_batting_metrics` -- ver ProjectionEngine.PA_STABILIZATION_POINT/IP_STABILIZATION_POINT."""
    return PitchingMetrics(
        era=3.50, whip=1.10, k_percent=0.22, bb_percent=0.07, k_minus_bb_percent=0.15,
        hr_per_9=1.0, fip=fip, xfip=None, lob_percent=0.72, gb_percent=None, fb_percent=None,
        innings_pitched=innings_pitched,
    )


class _FakeApiClient:
    def __init__(self, games: List[GameSummary]) -> None:
        self.games = games

    def get_games_for_date(self, date: Optional[str] = None) -> List[GameSummary]:
        return self.games

    # Métodos abaixo alimentam o BullpenService padrão quando nenhum é injetado.
    # Roster vazio -> BullpenStatus sem relievers -> metrics=None -> equivale a
    # "sem dados de bullpen", preservando o comportamento anterior à integração.
    def get_team_roster(self, team_id: int, roster_type: str = "active") -> List:
        return []

    def get_player_season_stats(self, person_id: int, group: str, season: Optional[int] = None) -> dict:
        return {}

    def get_player_game_log(self, person_id: int, group: str, season: Optional[int] = None) -> List:
        return []

    # Classificação vazia -> StandingsService padrão não bloqueia nenhum mercado,
    # preservando o comportamento dos testes que não exercitam o filtro por qualidade.
    def get_standings(self, season: int) -> dict:
        return {}


class _FakeLineupService:
    def __init__(self, home_snapshot: LineupSnapshot, away_snapshot: LineupSnapshot) -> None:
        self.home_snapshot = home_snapshot
        self.away_snapshot = away_snapshot

    def get_batting_order(self, game_pk: int, team_side: str) -> LineupSnapshot:
        return self.home_snapshot if team_side == "home" else self.away_snapshot


class _QueueOffenseService:
    """Devolve os resultados em ordem de chamada (home primeiro, depois away)."""

    def __init__(self, results: List[Optional[BattingMetrics]]) -> None:
        self._results = list(results)

    def get_team_offense_metrics_from_raw_stats(self, raw_stats_list) -> Optional[BattingMetrics]:
        return self._results.pop(0)

    def get_team_offense_metrics(self, player_ids, season=None) -> Optional[BattingMetrics]:
        return self._results.pop(0)


class _QueuePitchingService:
    def __init__(self, results: List[Optional[PitchingMetrics]]) -> None:
        self._results = list(results)

    def get_pitching_metrics(self, person_id: int, season: Optional[int] = None) -> Optional[PitchingMetrics]:
        return self._results.pop(0)


class _FakeBullpenService:
    """Devolve um BullpenStatus fixo por team_id."""

    def __init__(self, status_by_team_id: dict) -> None:
        self.status_by_team_id = status_by_team_id

    def get_bullpen_status(self, team_id, reference_date=None, season=None) -> Optional[BullpenStatus]:
        return self.status_by_team_id.get(team_id)


class _NeutralWeatherService:
    """Fator climático sempre neutro -- evita que testes de outros insumos (park factor
    etc.) dependam de uma chamada de rede real à API de previsão do tempo."""

    def get_weather_conditions(self, venue, game_datetime):
        from mlb_quantitative_engine.services.weather_service import WeatherConditions

        return WeatherConditions(venue or "desconhecido", None, None, None, False, 1.0)


class _FakeOddsService:
    """Devolve uma lista fixa de GameOdds (vazia por padrão -> sem dados de mercado)."""

    def __init__(self, games_odds: List = None, team_totals_by_event: dict = None) -> None:
        self.games_odds = games_odds or []
        self.team_totals_by_event = team_totals_by_event or {}

    def get_all_game_odds(self) -> List:
        return self.games_odds

    def find_game_odds(self, games, home_team: str, away_team: str):
        for game in games:
            if game.home_team.lower() == home_team.lower() and game.away_team.lower() == away_team.lower():
                return game
        return None

    def get_team_totals(self, event_id: str, home_team: str, away_team: str):
        from mlb_quantitative_engine.services.odds_service import GameTeamTotals
        return self.team_totals_by_event.get(event_id, GameTeamTotals(home=None, away=None))


class _CountingOddsService(_FakeOddsService):
    """Como _FakeOddsService, mas conta chamadas a get_team_totals -- usado para provar
    que a retentativa de lineup NÃO gasta créditos novos de odds (reaproveita as odds
    já persistidas em vez de rebuscar)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.get_team_totals_calls = 0

    def get_team_totals(self, event_id: str, home_team: str, away_team: str):
        self.get_team_totals_calls += 1
        return super().get_team_totals(event_id, home_team, away_team)


def _bullpen_status(team_id: int, fip: float, fatigue_index: float = 0.0, unavailable_count: int = 0) -> BullpenStatus:
    return BullpenStatus(
        team_id=team_id,
        reference_date="2026-07-17",
        relievers=[],
        closer_player_id=None,
        setup_player_id=None,
        unavailable_count=unavailable_count,
        fatigue_index=fatigue_index,
        metrics=_pitching_metrics(fip),
    )


_OFFICIAL_HOME_LINEUP = LineupSnapshot(
    team_side="home",
    source="official",
    confidence_score=90,
    entries=[LineupEntry(player_id=1, batting_order=1, raw_batting_stats={"atBats": 300, "hits": 90})],
)
_OFFICIAL_AWAY_LINEUP = LineupSnapshot(
    team_side="away",
    source="official",
    confidence_score=90,
    entries=[LineupEntry(player_id=2, batting_order=1, raw_batting_stats={"atBats": 300, "hits": 80})],
)
_PROBABLE_HOME_LINEUP = LineupSnapshot(
    team_side="home",
    source="probable_roster",
    confidence_score=40,
    entries=[LineupEntry(player_id=1, batting_order=1)],
)


@pytest.fixture()
def repository(tmp_path: Path) -> Repository:
    return Repository(db_path=str(tmp_path / "report.db"))


def test_generate_daily_report_creates_projection_for_complete_game(repository: Repository) -> None:
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(110.0), _batting_metrics(95.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(3.60), _pitching_metrics(4.50)]),
        odds_service=_FakeOddsService(),
    )

    rows = generator.generate_daily_report("2026-07-17")

    assert len(rows) == 1
    row = rows[0]
    assert row.skip_reason is None
    assert row.projected_home_runs is not None
    assert row.projected_away_runs is not None
    assert row.projected_total_runs == pytest.approx(row.projected_home_runs + row.projected_away_runs)
    assert 0.0 <= row.projected_probability_over <= 1.0
    assert row.confidence_score == pytest.approx(90.0)  # ambas as lineups são oficiais


def test_generate_daily_report_applies_real_park_factor(tmp_path: Path) -> None:
    """Coors Field (hitter-friendly) deve produzir uma projeção maior que Oracle Park (pitcher-friendly)
    para o mesmo ataque e pitching, confirmando que o park factor real está sendo aplicado."""

    def _make_generator(venue: str) -> ReportGenerator:
        return ReportGenerator(
            api_client=_FakeApiClient([_game_summary(venue=venue)]),
            repository=Repository(db_path=str(tmp_path / f"{venue}.db")),
            lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
            offense_service=_QueueOffenseService([_batting_metrics(100.0), _batting_metrics(100.0)]),
            pitching_service=_QueuePitchingService([_pitching_metrics(4.0), _pitching_metrics(4.0)]),
            odds_service=_FakeOddsService(),
            weather_service=_NeutralWeatherService(),  # isola o teste do park factor da rede
        )

    coors_rows = _make_generator("Coors Field").generate_daily_report("2026-07-17")
    oracle_rows = _make_generator("Oracle Park").generate_daily_report("2026-07-17")

    assert coors_rows[0].projected_total_runs > oracle_rows[0].projected_total_runs


def test_generate_daily_report_applies_weather_factor(tmp_path: Path) -> None:
    """Um fator climático quente (>1.0) deve elevar a projeção total vs. um fator frio (<1.0),
    para o mesmo ataque/pitching/estádio, confirmando que o clima entra na projeção."""

    class _FixedWeatherService:
        def __init__(self, factor: float) -> None:
            self.factor = factor

        def get_weather_conditions(self, venue, game_datetime):
            from mlb_quantitative_engine.services.weather_service import WeatherConditions

            return WeatherConditions(venue or "x", 85.0, 5.0, 180.0, False, self.factor)

    def _make_generator(factor: float) -> ReportGenerator:
        return ReportGenerator(
            api_client=_FakeApiClient([_game_summary()]),
            repository=Repository(db_path=str(tmp_path / f"weather-{factor}.db")),
            lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
            offense_service=_QueueOffenseService([_batting_metrics(100.0), _batting_metrics(100.0)]),
            pitching_service=_QueuePitchingService([_pitching_metrics(4.0), _pitching_metrics(4.0)]),
            odds_service=_FakeOddsService(),
            weather_service=_FixedWeatherService(factor),
        )

    hot_rows = _make_generator(1.08).generate_daily_report("2026-07-17")
    cold_rows = _make_generator(0.92).generate_daily_report("2026-07-17")

    assert hot_rows[0].projected_total_runs > cold_rows[0].projected_total_runs


def test_generate_daily_report_applies_bullpen_fatigue(tmp_path: Path) -> None:
    """Um bullpen fatigado dos dois lados deve produzir uma projeção de corridas maior
    que o mesmo confronto com bullpens descansados, confirmando a integração do bullpen."""

    def _make_generator(fatigue_index: float) -> ReportGenerator:
        bullpen_service = _FakeBullpenService(
            {
                1001: _bullpen_status(1001, fip=4.00, fatigue_index=fatigue_index),
                1002: _bullpen_status(1002, fip=4.00, fatigue_index=fatigue_index),
            }
        )
        return ReportGenerator(
            api_client=_FakeApiClient([_game_summary()]),
            repository=Repository(db_path=str(tmp_path / f"fatigue-{fatigue_index}.db")),
            lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
            offense_service=_QueueOffenseService([_batting_metrics(100.0), _batting_metrics(100.0)]),
            pitching_service=_QueuePitchingService([_pitching_metrics(4.0), _pitching_metrics(4.0)]),
            bullpen_service=bullpen_service,
            odds_service=_FakeOddsService(),
        )

    fresh_rows = _make_generator(fatigue_index=0.0).generate_daily_report("2026-07-17")
    tired_rows = _make_generator(fatigue_index=100.0).generate_daily_report("2026-07-17")

    assert tired_rows[0].projected_total_runs > fresh_rows[0].projected_total_runs


def test_generate_daily_report_falls_back_to_starter_only_when_bullpen_unavailable(repository: Repository) -> None:
    """Quando o bullpen não pode ser determinado, a projeção deve continuar funcionando
    usando apenas o FIP do titular (comportamento anterior à integração)."""
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(100.0), _batting_metrics(100.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(4.0), _pitching_metrics(4.0)]),
        bullpen_service=_FakeBullpenService({}),  # nenhum team_id cadastrado
        odds_service=_FakeOddsService(),
    )

    rows = generator.generate_daily_report("2026-07-17")

    assert rows[0].skip_reason is None
    assert rows[0].projected_total_runs is not None


def test_bullpen_status_returns_none_when_team_id_is_missing(repository: Repository) -> None:
    """Exercita diretamente o fallback de _bullpen_status quando team_id é None."""
    generator = ReportGenerator(
        api_client=_FakeApiClient([]),
        repository=repository,
        bullpen_service=_FakeBullpenService({1001: _bullpen_status(1001, fip=4.0)}),
        odds_service=_FakeOddsService(),
    )

    assert generator._bullpen_status(None, "2026-07-17") is None


def test_generate_daily_report_uses_fallback_offense_path_when_lineup_not_official(repository: Repository) -> None:
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_PROBABLE_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(100.0), _batting_metrics(100.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(4.0), _pitching_metrics(4.0)]),
        odds_service=_FakeOddsService(),
    )

    rows = generator.generate_daily_report("2026-07-17")

    assert rows[0].skip_reason is None


def test_generate_daily_report_skips_game_without_probable_pitchers(repository: Repository) -> None:
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary(home_pitcher_id=None)]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([]),
        pitching_service=_QueuePitchingService([]),
        odds_service=_FakeOddsService(),
    )

    rows = generator.generate_daily_report("2026-07-17")

    assert rows[0].skip_reason == "arremessadores titulares indisponíveis"
    assert rows[0].projected_total_runs is None


def test_generate_daily_report_skips_game_when_offense_metrics_unavailable(repository: Repository) -> None:
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([None, _batting_metrics(100.0)]),
        pitching_service=_QueuePitchingService([]),
        odds_service=_FakeOddsService(),
    )

    rows = generator.generate_daily_report("2026-07-17")

    assert "ataque" in rows[0].skip_reason


def test_generate_daily_report_skips_game_when_pitcher_metrics_unavailable(repository: Repository) -> None:
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(100.0), _batting_metrics(100.0)]),
        pitching_service=_QueuePitchingService([None, _pitching_metrics(4.0)]),
        odds_service=_FakeOddsService(),
    )

    rows = generator.generate_daily_report("2026-07-17")

    assert "arremessador" in rows[0].skip_reason


def test_generate_daily_report_persists_game_and_projection(repository: Repository) -> None:
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(110.0), _batting_metrics(95.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(3.60), _pitching_metrics(4.50)]),
        odds_service=_FakeOddsService(),
    )

    generator.generate_daily_report("2026-07-17")

    saved_game = repository.get_game_by_pk(1)
    assert saved_game is not None
    projections = repository.list_projections_for_game(saved_game.id)
    assert len(projections) == 1
    assert projections[0].model_version == "nb-shrinkage-mc-v1"


def test_game_is_still_persisted_even_when_projection_is_skipped(repository: Repository) -> None:
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary(home_pitcher_id=None)]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([]),
        pitching_service=_QueuePitchingService([]),
        odds_service=_FakeOddsService(),
    )

    generator.generate_daily_report("2026-07-17")

    saved_game = repository.get_game_by_pk(1)
    assert saved_game is not None
    assert repository.list_projections_for_game(saved_game.id) == []


def test_rows_to_dataframe_produces_one_row_per_game(repository: Repository) -> None:
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary(game_pk=1), _game_summary(game_pk=2, home_pitcher_id=None)]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(110.0), _batting_metrics(95.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(3.60), _pitching_metrics(4.50)]),
        odds_service=_FakeOddsService(),
    )

    rows = generator.generate_daily_report("2026-07-17")
    df = rows_to_dataframe(rows)

    assert len(df) == 2
    assert "projected_total_runs" in df.columns
    assert "skip_reason" in df.columns


def test_generate_daily_report_includes_market_odds_when_available(repository: Repository) -> None:
    from mlb_quantitative_engine.services.odds_service import GameOdds, MoneylineQuote, TotalsQuote

    game_odds = GameOdds(
        home_team="Home Team",
        away_team="Away Team",
        commence_time="2026-07-17T23:05:00Z",
        totals=[
            TotalsQuote(
                point=8.5, bookmaker_count=2,
                over_price=1.92, over_bookmaker="DraftKings",
                under_price=1.87, under_bookmaker="FanDuel",
            )
        ],
        moneyline=MoneylineQuote(home_price=1.80, home_bookmaker="FanDuel", away_price=2.10, away_bookmaker="DraftKings"),
    )

    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(100.0), _batting_metrics(100.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(4.0), _pitching_metrics(4.0)]),
        odds_service=_FakeOddsService([game_odds]),
    )

    rows = generator.generate_daily_report("2026-07-17")

    assert rows[0].market_total_line == 8.5
    assert rows[0].market_over_price == 1.92
    assert rows[0].market_over_bookmaker == "DraftKings"
    assert rows[0].market_under_price == 1.87
    assert rows[0].market_under_bookmaker == "FanDuel"


def test_generate_daily_report_market_fields_are_none_without_matching_odds(repository: Repository) -> None:
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(100.0), _batting_metrics(100.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(4.0), _pitching_metrics(4.0)]),
        odds_service=_FakeOddsService(),  # sem odds cadastradas
    )

    rows = generator.generate_daily_report("2026-07-17")

    assert rows[0].market_total_line is None
    assert rows[0].market_over_price is None


def test_generate_daily_report_flags_clear_value_bet(repository: Repository) -> None:
    """Projeção de ~11 corridas contra uma linha de mercado de 8.5 é um edge grande e óbvio;
    com as duas lineups oficiais (confiança 90%), deve gerar uma recomendação de OVER."""
    from mlb_quantitative_engine.services.odds_service import GameOdds, MoneylineQuote, TotalsQuote

    game_odds = GameOdds(
        home_team="Home Team",
        away_team="Away Team",
        commence_time="2026-07-17T23:05:00Z",
        totals=[
            TotalsQuote(
                point=8.5, bookmaker_count=2,
                over_price=1.90, over_bookmaker="DraftKings",
                under_price=1.90, under_bookmaker="FanDuel",
            )
        ],
        moneyline=MoneylineQuote(home_price=1.80, home_bookmaker="FanDuel", away_price=2.10, away_bookmaker="DraftKings"),
    )

    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(160.0), _batting_metrics(160.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(6.0), _pitching_metrics(6.0)]),
        odds_service=_FakeOddsService([game_odds]),
    )

    rows = generator.generate_daily_report("2026-07-17")
    row = rows[0]

    assert row.projected_total_runs > 8.5
    assert row.value_bet_recommendation == "game_total_over"
    assert row.value_bet_edge is not None and row.value_bet_edge > 0.04
    assert row.value_bet_expected_value is not None and row.value_bet_expected_value > 0.05
    assert row.value_bet_suggested_stake_fraction is not None and row.value_bet_suggested_stake_fraction > 0
    assert row.value_bet_suggested_stake_fraction <= 0.02  # nunca mais que 2% da banca
    assert row.value_bet_minimum_price is not None


class _FakeTelegramNotifier:
    def __init__(self, raise_error: bool = False) -> None:
        self.sent_bets = []
        self.sent_datetimes = []
        self.raise_error = raise_error

    def send_value_bet_alert(self, bet, game_datetime=None):
        if self.raise_error:
            raise RuntimeError("Telegram indisponível (simulado)")
        self.sent_bets.append(bet)
        self.sent_datetimes.append(game_datetime)
        return {"ok": True}


def _game_odds_with_clear_edge():
    from mlb_quantitative_engine.services.odds_service import GameOdds, MoneylineQuote, TotalsQuote

    return GameOdds(
        home_team="Home Team",
        away_team="Away Team",
        commence_time="2026-07-17T23:05:00Z",
        totals=[
            TotalsQuote(
                point=8.5, bookmaker_count=2,
                over_price=1.90, over_bookmaker="DraftKings",
                under_price=1.90, under_bookmaker="FanDuel",
            )
        ],
        moneyline=MoneylineQuote(home_price=1.80, home_bookmaker="FanDuel", away_price=2.10, away_bookmaker="DraftKings"),
    )


def test_telegram_alert_is_sent_when_a_bet_qualifies(repository: Repository) -> None:
    notifier = _FakeTelegramNotifier()
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(160.0), _batting_metrics(160.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(6.0), _pitching_metrics(6.0)]),
        odds_service=_FakeOddsService([_game_odds_with_clear_edge()]),
        telegram_notifier=notifier,
    )

    generator.generate_daily_report("2026-07-17")

    assert len(notifier.sent_bets) >= 1
    assert all(bet.meets_criteria for bet in notifier.sent_bets)


def test_no_telegram_alert_without_notifier_injected(repository: Repository) -> None:
    """Comportamento padrão (sem notifier) não deve tentar enviar nada -- só confirma
    que não há erro/nenhuma tentativa quando o notifier simplesmente não existe."""
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(160.0), _batting_metrics(160.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(6.0), _pitching_metrics(6.0)]),
        odds_service=_FakeOddsService([_game_odds_with_clear_edge()]),
    )

    rows = generator.generate_daily_report("2026-07-17")
    assert rows[0].value_bet_recommendation is not None  # a aposta ainda qualifica normalmente


def test_no_telegram_alert_when_no_bet_qualifies(repository: Repository) -> None:
    notifier = _FakeTelegramNotifier()
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_PROBABLE_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),  # confiança baixa
        offense_service=_QueueOffenseService([_batting_metrics(160.0), _batting_metrics(160.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(6.0), _pitching_metrics(6.0)]),
        odds_service=_FakeOddsService([_game_odds_with_clear_edge()]),
        telegram_notifier=notifier,
    )

    generator.generate_daily_report("2026-07-17")

    assert notifier.sent_bets == []


def test_only_one_alert_sent_per_game_and_it_is_the_highest_probability(repository: Repository) -> None:
    """No máximo UMA aposta por jogo é enviada (a de maior probabilidade projetada entre
    as que qualificam) -- as demais avaliações ficam persistidas no banco mas com
    alert_sent=False. É o flag alert_sent que o verificador de resultado
    (bet_result_checker) usa depois para saber o que precisa de GREEN/RED/PUSH."""
    notifier = _FakeTelegramNotifier()
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(160.0), _batting_metrics(160.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(6.0), _pitching_metrics(6.0)]),
        odds_service=_FakeOddsService([_game_odds_with_clear_edge()]),
        telegram_notifier=notifier,
    )

    generator.generate_daily_report("2026-07-17")

    assert len(notifier.sent_bets) == 1  # exatamente uma, mesmo com várias qualificando
    sent = notifier.sent_bets[0]

    all_bets = repository.list_value_bets()
    flagged = [bet for bet in all_bets if bet.alert_sent]
    assert len(flagged) == 1
    assert flagged[0].meets_criteria

    # a enviada é a de maior probabilidade entre as que qualificam
    qualifying = [bet for bet in all_bets if bet.meets_criteria]
    best_prob = max(bet.projection_probability for bet in qualifying)
    assert flagged[0].projection_probability == pytest.approx(best_prob)
    assert sent.market == flagged[0].market


def test_alert_sent_stays_false_when_telegram_send_fails(repository: Repository) -> None:
    notifier = _FakeTelegramNotifier(raise_error=True)
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(160.0), _batting_metrics(160.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(6.0), _pitching_metrics(6.0)]),
        odds_service=_FakeOddsService([_game_odds_with_clear_edge()]),
        telegram_notifier=notifier,
    )

    generator.generate_daily_report("2026-07-17")

    assert all(not bet.alert_sent for bet in repository.list_value_bets())


def test_no_alert_sent_flag_without_notifier_injected(repository: Repository) -> None:
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(160.0), _batting_metrics(160.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(6.0), _pitching_metrics(6.0)]),
        odds_service=_FakeOddsService([_game_odds_with_clear_edge()]),
    )

    generator.generate_daily_report("2026-07-17")

    assert all(not bet.alert_sent for bet in repository.list_value_bets())


def test_telegram_send_failure_does_not_crash_report_generation(repository: Repository) -> None:
    notifier = _FakeTelegramNotifier(raise_error=True)
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(160.0), _batting_metrics(160.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(6.0), _pitching_metrics(6.0)]),
        odds_service=_FakeOddsService([_game_odds_with_clear_edge()]),
        telegram_notifier=notifier,
    )

    rows = generator.generate_daily_report("2026-07-17")  # não deve levantar exceção

    assert rows[0].value_bet_recommendation is not None


def test_generate_daily_report_persists_value_bets_for_both_sides(repository: Repository) -> None:
    from mlb_quantitative_engine.services.odds_service import GameOdds, MoneylineQuote, TotalsQuote

    game_odds = GameOdds(
        home_team="Home Team",
        away_team="Away Team",
        commence_time="2026-07-17T23:05:00Z",
        totals=[
            TotalsQuote(
                point=8.5, bookmaker_count=2,
                over_price=1.90, over_bookmaker="DraftKings",
                under_price=1.90, under_bookmaker="FanDuel",
            )
        ],
        moneyline=MoneylineQuote(home_price=1.80, home_bookmaker="FanDuel", away_price=2.10, away_bookmaker="DraftKings"),
    )

    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(100.0), _batting_metrics(100.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(4.0), _pitching_metrics(4.0)]),
        odds_service=_FakeOddsService([game_odds]),
    )

    generator.generate_daily_report("2026-07-17")

    all_bets = repository.list_value_bets()
    assert len(all_bets) == 2
    markets = {bet.market for bet in all_bets}
    assert markets == {"game_total_over", "game_total_under"}


def test_generate_daily_report_no_value_bet_fields_without_market_odds(repository: Repository) -> None:
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(100.0), _batting_metrics(100.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(4.0), _pitching_metrics(4.0)]),
        odds_service=_FakeOddsService(),  # sem odds
    )

    rows = generator.generate_daily_report("2026-07-17")

    assert rows[0].value_bet_recommendation is None
    assert rows[0].value_bet_edge is None
    assert repository.list_value_bets() == []


def test_generate_daily_report_low_confidence_blocks_recommendation_despite_edge(repository: Repository) -> None:
    """Mesmo com edge grande, lineup provável (confiança 40%) reduz a confiança média
    abaixo do limiar de 70% e não deve gerar recomendação."""
    from mlb_quantitative_engine.services.odds_service import GameOdds, MoneylineQuote, TotalsQuote

    game_odds = GameOdds(
        home_team="Home Team",
        away_team="Away Team",
        commence_time="2026-07-17T23:05:00Z",
        totals=[
            TotalsQuote(
                point=8.5, bookmaker_count=2,
                over_price=1.90, over_bookmaker="DraftKings",
                under_price=1.90, under_bookmaker="FanDuel",
            )
        ],
        moneyline=MoneylineQuote(home_price=1.80, home_bookmaker="FanDuel", away_price=2.10, away_bookmaker="DraftKings"),
    )

    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_PROBABLE_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(160.0), _batting_metrics(160.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(6.0), _pitching_metrics(6.0)]),
        odds_service=_FakeOddsService([game_odds]),
    )

    rows = generator.generate_daily_report("2026-07-17")
    row = rows[0]

    assert row.confidence_score < 70.0
    assert row.value_bet_recommendation is None


def test_is_within_send_window_true_when_close_to_game_start(repository: Repository) -> None:
    generator = ReportGenerator(api_client=_FakeApiClient([]), repository=repository, odds_service=_FakeOddsService())
    now = datetime(2026, 7, 17, 20, 0, tzinfo=timezone.utc)
    assert generator._is_within_send_window("2026-07-17T20:30:00Z", now=now) is True  # 30 min depois


def test_is_within_send_window_false_when_game_is_far_away(repository: Repository) -> None:
    """Regressão do bug real: uma retentativa de lineup não deve poder disparar um
    alerta horas antes do jogo só porque a confiança finalmente ficou alta."""
    generator = ReportGenerator(api_client=_FakeApiClient([]), repository=repository, odds_service=_FakeOddsService())
    now = datetime(2026, 7, 17, 14, 0, tzinfo=timezone.utc)
    assert generator._is_within_send_window("2026-07-17T20:30:00Z", now=now) is False  # 6h30 depois


def test_is_within_send_window_true_when_game_datetime_unknown(repository: Repository) -> None:
    generator = ReportGenerator(api_client=_FakeApiClient([]), repository=repository, odds_service=_FakeOddsService())
    assert generator._is_within_send_window(None) is True


def test_no_telegram_alert_when_game_is_far_in_the_future(repository: Repository) -> None:
    far_away = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat().replace("+00:00", "Z")
    notifier = _FakeTelegramNotifier()
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary(game_datetime=far_away)]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(160.0), _batting_metrics(160.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(6.0), _pitching_metrics(6.0)]),
        odds_service=_FakeOddsService([_game_odds_with_clear_edge()]),
        telegram_notifier=notifier,
    )

    rows = generator.generate_daily_report("2026-07-17")

    assert rows[0].value_bet_recommendation is not None  # ainda qualifica e aparece no relatório
    assert notifier.sent_bets == []  # mas não é enviado -- jogo longe demais
    qualifying = [b for b in repository.list_value_bets() if b.meets_criteria]
    assert qualifying and all(not b.alert_sent for b in qualifying)


def test_telegram_alert_sent_when_within_the_send_window(repository: Repository) -> None:
    soon = (datetime.now(timezone.utc) + timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
    notifier = _FakeTelegramNotifier()
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary(game_datetime=soon)]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(160.0), _batting_metrics(160.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(6.0), _pitching_metrics(6.0)]),
        odds_service=_FakeOddsService([_game_odds_with_clear_edge()]),
        telegram_notifier=notifier,
    )

    generator.generate_daily_report("2026-07-17")

    assert len(notifier.sent_bets) >= 1
    # o horário do jogo é repassado ao notifier para compor a data/hora na mensagem
    assert notifier.sent_datetimes[0] == soon


def test_duplicate_alert_is_not_resent_when_game_is_reevaluated(repository: Repository) -> None:
    """Regressão do bug real: reavaliar o mesmo jogo (retentativa, reprocessamento
    atrasado etc.) não deve reenviar a mesma recomendação de novo."""
    soon = (datetime.now(timezone.utc) + timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
    notifier = _FakeTelegramNotifier()
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary(game_datetime=soon)]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService(
            [_batting_metrics(160.0), _batting_metrics(160.0), _batting_metrics(160.0), _batting_metrics(160.0)]
        ),
        pitching_service=_QueuePitchingService(
            [_pitching_metrics(6.0), _pitching_metrics(6.0), _pitching_metrics(6.0), _pitching_metrics(6.0)]
        ),
        odds_service=_FakeOddsService([_game_odds_with_clear_edge()]),
        telegram_notifier=notifier,
    )

    generator.generate_daily_report("2026-07-17")
    first_sent_count = len(notifier.sent_bets)
    assert first_sent_count >= 1

    generator.generate_daily_report("2026-07-17")  # reavaliação subsequente do mesmo jogo

    assert len(notifier.sent_bets) == first_sent_count  # não duplicou


def test_generate_daily_report_evaluates_team_totals_when_event_id_available(repository: Repository) -> None:
    """Quando o jogo tem event_id e o mercado team_totals está disponível, a recomendação
    pode vir de um total de time (não só do total do jogo) -- e deve ser persistida."""
    from mlb_quantitative_engine.services.odds_service import (
        GameOdds, GameTeamTotals, MoneylineQuote, TeamTotalsQuote, TotalsQuote,
    )

    game_odds = GameOdds(
        home_team="Home Team",
        away_team="Away Team",
        commence_time="2026-07-17T23:05:00Z",
        totals=[
            TotalsQuote(point=8.5, bookmaker_count=2, over_price=1.90, over_bookmaker="FanDuel",
                        under_price=1.90, under_bookmaker="DraftKings"),
        ],
        moneyline=MoneylineQuote(home_price=1.80, home_bookmaker="FanDuel", away_price=2.10, away_bookmaker="DraftKings"),
        event_id="evt1",
    )
    team_totals = GameTeamTotals(
        home=TeamTotalsQuote(team="Home Team", point=3.5, bookmaker_count=2,
                              over_price=1.85, over_bookmaker="FanDuel",
                              under_price=1.95, under_bookmaker="DraftKings"),
        away=TeamTotalsQuote(team="Away Team", point=3.5, bookmaker_count=2,
                              over_price=1.90, over_bookmaker="FanDuel",
                              under_price=1.90, under_bookmaker="DraftKings"),
    )

    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(160.0), _batting_metrics(100.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(6.0), _pitching_metrics(4.0)]),
        odds_service=_FakeOddsService([game_odds], team_totals_by_event={"evt1": team_totals}),
    )

    rows = generator.generate_daily_report("2026-07-17")
    row = rows[0]

    saved_game = repository.get_game_by_pk(1)
    projections = repository.list_projections_for_game(saved_game.id)
    all_bets = repository.list_value_bets_for_projection(projections[0].id)

    # 2 (game total) + 2 (home team total) + 2 (away team total) = 6 avaliações persistidas
    assert len(all_bets) == 6
    markets = {bet.market for bet in all_bets}
    assert "home_team_total_over" in markets
    assert "away_team_total_over" in markets
    assert row.value_bet_recommendation is not None


# --- Filtro de envio por qualidade do time (StandingsService) ---


class _FakeStandingsService:
    """Devolve um mapa fixo {team_id: lado_suprimido} sem tocar na rede."""

    def __init__(self, blocked_sides: dict) -> None:
        self.blocked_sides = blocked_sides

    def get_blocked_team_market_sides(self, season=None) -> dict:
        return dict(self.blocked_sides)


def _candidate(market: str, prob: float, meets: bool = True):
    from mlb_quantitative_engine.models.value_bet import ValueBet

    return ValueBet(
        game_pk=1, home_team="Home Team", away_team="Away Team", market=market,
        bookmaker="X", price=1.90, point=3.5, projected_probability=prob,
        implied_probability_raw=0.53, implied_probability_fair=0.50, edge=0.10,
        expected_value=0.10, kelly_fraction=0.10, kelly_fraction_quarter=0.025,
        suggested_stake_fraction=0.02, minimum_acceptable_price=1.80,
        confidence_score=90.0, meets_criteria=meets,
    )


def test_select_bet_skips_blocked_market_and_falls_through_to_next_best() -> None:
    blocked_best = _candidate("away_team_total_over", 0.70)  # maior prob, mas bloqueada
    runner_up = _candidate("game_total_over", 0.60)
    chosen = ReportGenerator._select_bet_to_alert(
        [blocked_best, runner_up], frozenset({"away_team_total_over"})
    )
    assert chosen is runner_up  # cai para a próxima qualificada, não fica sem enviar


def test_select_bet_returns_none_when_the_only_qualifier_is_blocked() -> None:
    only = _candidate("away_team_total_over", 0.70)
    assert ReportGenerator._select_bet_to_alert([only], frozenset({"away_team_total_over"})) is None


def test_select_bet_unaffected_when_blocked_market_is_a_different_side() -> None:
    # bloquear o OVER do time não impede o UNDER do mesmo time de ser enviado
    under = _candidate("away_team_total_under", 0.66)
    assert ReportGenerator._select_bet_to_alert([under], frozenset({"away_team_total_over"})) is under


def test_blocked_team_total_markets_maps_team_ids_to_sides(repository: Repository) -> None:
    generator = ReportGenerator(
        api_client=_FakeApiClient([]),
        repository=repository,
        odds_service=_FakeOddsService(),
        standings_service=_FakeStandingsService({1001: "over", 1002: "under"}),
    )
    # _game_summary(): home_team_id=1001, away_team_id=1002
    blocked = generator._blocked_team_total_markets(_game_summary())
    assert blocked == frozenset({"home_team_total_over", "away_team_total_under"})


def test_weak_team_over_is_persisted_but_never_sent(repository: Repository) -> None:
    """Time fraco (classificação abaixo do limiar) tem o Team-Total OVER suprimido no
    ENVIO, mas a avaliação continua persistida para calibração."""
    from mlb_quantitative_engine.services.odds_service import (
        GameOdds, GameTeamTotals, MoneylineQuote, TeamTotalsQuote, TotalsQuote,
    )

    game_odds = GameOdds(
        home_team="Home Team", away_team="Away Team", commence_time="2026-07-17T23:05:00Z",
        totals=[TotalsQuote(point=8.5, bookmaker_count=2, over_price=1.90, over_bookmaker="FanDuel",
                            under_price=1.90, under_bookmaker="DraftKings")],
        moneyline=MoneylineQuote(home_price=1.80, home_bookmaker="FanDuel", away_price=2.10, away_bookmaker="DraftKings"),
        event_id="evt1",
    )
    team_totals = GameTeamTotals(
        home=TeamTotalsQuote(team="Home Team", point=3.5, bookmaker_count=2,
                              over_price=1.85, over_bookmaker="FanDuel",
                              under_price=1.95, under_bookmaker="DraftKings"),
        away=TeamTotalsQuote(team="Away Team", point=3.5, bookmaker_count=2,
                              over_price=1.90, over_bookmaker="FanDuel",
                              under_price=1.90, under_bookmaker="DraftKings"),
    )
    notifier = _FakeTelegramNotifier()
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(170.0), _batting_metrics(80.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(6.5), _pitching_metrics(3.5)]),
        odds_service=_FakeOddsService([game_odds], team_totals_by_event={"evt1": team_totals}),
        telegram_notifier=notifier,
        standings_service=_FakeStandingsService({1001: "over"}),  # Home Team fraco -> bloqueia OVER
    )

    generator.generate_daily_report("2026-07-17")

    # nunca envia o over do time fraco...
    assert all(bet.market != "home_team_total_over" for bet in notifier.sent_bets)
    # ...mas a avaliação segue no banco para calibração
    persisted = {bet.market for bet in repository.list_value_bets()}
    assert "home_team_total_over" in persisted


def test_low_confidence_row_schedules_a_lineup_retry(repository: Repository) -> None:
    """Confiança abaixo do limiar (lineup só provável de um dos lados) deve agendar uma
    retentativa daqui a 30 minutos, para reconsultar a lineup depois sem gastar
    créditos novos de odds."""
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_PROBABLE_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(100.0), _batting_metrics(100.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(4.0), _pitching_metrics(4.0)]),
        odds_service=_FakeOddsService(),
    )

    generator.generate_daily_report("2026-07-17")

    due = repository.list_due_lineup_retries(datetime.now(timezone.utc) + timedelta(hours=1))
    assert len(due) == 1
    assert due[0].game_pk == 1


def test_high_confidence_row_resolves_any_pending_retry(repository: Repository) -> None:
    """Se um jogo já tinha uma retentativa pendente e a lineup agora está oficial
    (confiança alta), a retentativa deve ser marcada como resolvida."""
    repository.upsert_pending_lineup_retry(
        game_pk=1, game_date="2026-07-17", retry_at=datetime.now(timezone.utc) + timedelta(minutes=5)
    )

    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_OFFICIAL_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(100.0), _batting_metrics(100.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(4.0), _pitching_metrics(4.0)]),
        odds_service=_FakeOddsService(),
    )

    generator.generate_daily_report("2026-07-17")

    due = repository.list_due_lineup_retries(datetime.now(timezone.utc) + timedelta(hours=1))
    assert due == []


def test_retry_game_reuses_cached_odds_without_spending_new_credits(repository: Repository) -> None:
    """O cenário central do pedido: a primeira passada gasta créditos de odds mas a
    lineup ainda não é oficial (confiança baixa); a retentativa deve rebuscar só a
    lineup (grátis) e recalcular usando as MESMAS odds já persistidas, sem chamar
    get_team_totals de novo."""
    from mlb_quantitative_engine.services.odds_service import (
        GameOdds, GameTeamTotals, MoneylineQuote, TeamTotalsQuote, TotalsQuote,
    )

    game_odds = GameOdds(
        home_team="Home Team",
        away_team="Away Team",
        commence_time="2026-07-17T23:05:00Z",
        totals=[
            TotalsQuote(point=8.5, bookmaker_count=2, over_price=1.90, over_bookmaker="FanDuel",
                        under_price=1.90, under_bookmaker="DraftKings"),
        ],
        moneyline=MoneylineQuote(home_price=1.80, home_bookmaker="FanDuel", away_price=2.10, away_bookmaker="DraftKings"),
        event_id="evt1",
    )
    team_totals = GameTeamTotals(
        home=TeamTotalsQuote(team="Home Team", point=3.5, bookmaker_count=2,
                              over_price=1.85, over_bookmaker="FanDuel",
                              under_price=1.95, under_bookmaker="DraftKings"),
        away=TeamTotalsQuote(team="Away Team", point=3.5, bookmaker_count=2,
                              over_price=1.90, over_bookmaker="FanDuel",
                              under_price=1.90, under_bookmaker="DraftKings"),
    )

    lineup_service = _FakeLineupService(_PROBABLE_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP)
    odds_service = _CountingOddsService([game_odds], team_totals_by_event={"evt1": team_totals})
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=lineup_service,
        offense_service=_QueueOffenseService(
            [_batting_metrics(160.0), _batting_metrics(160.0), _batting_metrics(160.0), _batting_metrics(160.0)]
        ),
        pitching_service=_QueuePitchingService(
            [_pitching_metrics(6.0), _pitching_metrics(6.0), _pitching_metrics(6.0), _pitching_metrics(6.0)]
        ),
        odds_service=odds_service,
    )

    all_odds = generator.fetch_all_odds()
    game = _game_summary()
    first_row = generator.build_row(game, all_odds)

    assert first_row.confidence_score < 70.0
    assert odds_service.get_team_totals_calls == 1
    due = repository.list_due_lineup_retries(datetime.now(timezone.utc) + timedelta(hours=1))
    assert len(due) == 1

    # Lineup agora oficial (o que uma retentativa 30min depois normalmente encontraria).
    lineup_service.home_snapshot = _OFFICIAL_HOME_LINEUP

    retry_row = generator.retry_game(game.game_pk, now=datetime(2026, 7, 17, 20, 0, tzinfo=timezone.utc))

    assert retry_row is not None
    assert retry_row.confidence_score == pytest.approx(90.0)
    # Nenhuma chamada nova a get_team_totals -- as odds vieram do cache (ValueBets persistidos).
    assert odds_service.get_team_totals_calls == 1
    assert retry_row.market_total_line == 8.5
    assert retry_row.market_over_price == 1.90
    assert retry_row.value_bet_recommendation is not None

    due_after = repository.list_due_lineup_retries(datetime.now(timezone.utc) + timedelta(hours=1))
    assert due_after == []


def test_retry_game_gives_up_once_the_game_has_started(repository: Repository) -> None:
    """Depois que o jogo já começou, a retentativa deve desistir (resolver) em vez de
    reagendar indefinidamente."""
    generator = ReportGenerator(
        api_client=_FakeApiClient([_game_summary()]),
        repository=repository,
        lineup_service=_FakeLineupService(_PROBABLE_HOME_LINEUP, _OFFICIAL_AWAY_LINEUP),
        offense_service=_QueueOffenseService([_batting_metrics(100.0), _batting_metrics(100.0)]),
        pitching_service=_QueuePitchingService([_pitching_metrics(4.0), _pitching_metrics(4.0)]),
        odds_service=_FakeOddsService(),
    )

    game = _game_summary()
    generator.build_row(game, [])

    # game_datetime do jogo é 2026-07-17T23:05:00Z -- 'now' bem depois disso.
    result = generator.retry_game(game.game_pk, now=datetime(2026, 7, 18, 2, 0, tzinfo=timezone.utc))

    assert result is None
    due_after = repository.list_due_lineup_retries(datetime.now(timezone.utc) + timedelta(hours=1))
    assert due_after == []
