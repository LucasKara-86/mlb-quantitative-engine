from __future__ import annotations

from typing import Any, Dict, List

import pytest

from mlb_quantitative_engine.services.odds_service import GameOdds, OddsService, implied_probability


class _FakeOddsApiClient:
    def __init__(self, events: List[Dict[str, Any]], event_odds: Dict[str, Any] | None = None) -> None:
        self.events = events
        self.event_odds = event_odds or {}

    def get_mlb_odds(self, regions: str = "us", markets: str = "h2h,totals", odds_format: str = "decimal"):
        return self.events

    def get_event_odds(self, event_id: str, regions: str = "us", markets: str = "team_totals", odds_format: str = "decimal"):
        return self.event_odds.get(event_id, {"bookmakers": []})


def _sample_event() -> Dict[str, Any]:
    """Baseado na estrutura real observada na The Odds API: várias casas, uma delas (mybookieag)
    oferecendo uma linha de total diferente das demais (5.5 x 3.5)."""
    return {
        "id": "abc123",
        "home_team": "New York Yankees",
        "away_team": "Los Angeles Dodgers",
        "commence_time": "2026-07-17T23:05:00Z",
        "bookmakers": [
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Los Angeles Dodgers", "price": 1.28},
                            {"name": "New York Yankees", "price": 3.60},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.89, "point": 5.5},
                            {"name": "Under", "price": 1.85, "point": 5.5},
                        ],
                    },
                ],
            },
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Los Angeles Dodgers", "price": 1.36},
                            {"name": "New York Yankees", "price": 3.05},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.92, "point": 5.5},
                            {"name": "Under", "price": 1.84, "point": 5.5},
                        ],
                    },
                ],
            },
            {
                "key": "mybookieag",
                "title": "MyBookie.ag",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Los Angeles Dodgers", "price": 2.65},
                            {"name": "New York Yankees", "price": 1.45},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.90, "point": 3.5},
                            {"name": "Under", "price": 1.80, "point": 3.5},
                        ],
                    },
                ],
            },
        ],
    }


def test_get_all_game_odds_parses_events() -> None:
    client = _FakeOddsApiClient([_sample_event()])
    service = OddsService(api_client=client)

    games = service.get_all_game_odds()

    assert len(games) == 1
    assert games[0].home_team == "New York Yankees"
    assert games[0].away_team == "Los Angeles Dodgers"


def test_consensus_total_picks_the_point_with_most_bookmakers() -> None:
    """5.5 tem 2 casas (FanDuel, DraftKings); 3.5 tem só 1 (MyBookie) -> consenso deve ser 5.5."""
    client = _FakeOddsApiClient([_sample_event()])
    service = OddsService(api_client=client)
    game = service.get_all_game_odds()[0]

    consensus = game.consensus_total
    assert consensus is not None
    assert consensus.point == 5.5
    assert consensus.bookmaker_count == 2


def test_best_price_is_selected_within_the_same_point() -> None:
    client = _FakeOddsApiClient([_sample_event()])
    service = OddsService(api_client=client)
    game = service.get_all_game_odds()[0]

    quote_5_5 = next(t for t in game.totals if t.point == 5.5)
    # DraftKings (1.92) > FanDuel (1.89) para o Over na linha 5.5
    assert quote_5_5.over_price == pytest.approx(1.92)
    assert quote_5_5.over_bookmaker == "DraftKings"
    # FanDuel (1.85) > DraftKings (1.84) para o Under na linha 5.5
    assert quote_5_5.under_price == pytest.approx(1.85)
    assert quote_5_5.under_bookmaker == "FanDuel"


def test_moneyline_best_price_per_side() -> None:
    client = _FakeOddsApiClient([_sample_event()])
    service = OddsService(api_client=client)
    game = service.get_all_game_odds()[0]

    # Yankees (mandantes): FanDuel 3.60 > DraftKings 3.05 > MyBookie 1.45 -> melhor é FanDuel
    assert game.moneyline.home_price == pytest.approx(3.60)
    assert game.moneyline.home_bookmaker == "FanDuel"
    # Dodgers (visitantes): MyBookie 2.65 > DraftKings 1.36 > FanDuel 1.28 -> melhor é MyBookie
    assert game.moneyline.away_price == pytest.approx(2.65)
    assert game.moneyline.away_bookmaker == "MyBookie.ag"


def test_find_game_odds_matches_by_team_names_case_insensitive() -> None:
    client = _FakeOddsApiClient([_sample_event()])
    service = OddsService(api_client=client)
    games = service.get_all_game_odds()

    found = service.find_game_odds(games, home_team="new york yankees", away_team="Los Angeles Dodgers")
    assert found is not None
    assert found.home_team == "New York Yankees"


def test_find_game_odds_returns_none_when_no_match() -> None:
    client = _FakeOddsApiClient([_sample_event()])
    service = OddsService(api_client=client)
    games = service.get_all_game_odds()

    assert service.find_game_odds(games, home_team="Boston Red Sox", away_team="Tampa Bay Rays") is None


def _event_with_thin_disjoint_totals_lines() -> Dict[str, Any]:
    """Três casas, cada uma cobrindo uma linha DIFERENTE (empate 1x1x1 em bookmaker_count)
    -- cenário real que causou o bug: sem desempate por mediana, a "consenso" cairia na
    primeira linha encontrada nos dados brutos (8.5), mesmo havendo uma linha isolada e
    destoante (11.0) que não representa o mercado."""
    return {
        "id": "thin1",
        "home_team": "Kansas City Royals",
        "away_team": "San Diego Padres",
        "commence_time": "2026-07-18T20:10:00Z",
        "bookmakers": [
            {
                "key": "book_low",
                "title": "BookLow",
                "markets": [{"key": "totals", "outcomes": [
                    {"name": "Over", "price": 1.90, "point": 8.5},
                    {"name": "Under", "price": 1.90, "point": 8.5},
                ]}],
            },
            {
                "key": "book_mid",
                "title": "BookMid",
                "markets": [{"key": "totals", "outcomes": [
                    {"name": "Over", "price": 1.91, "point": 9.5},
                    {"name": "Under", "price": 1.89, "point": 9.5},
                ]}],
            },
            {
                "key": "book_outlier",
                "title": "BookOutlier",
                "markets": [{"key": "totals", "outcomes": [
                    {"name": "Over", "price": 1.96, "point": 11.0},
                    {"name": "Under", "price": 1.84, "point": 11.0},
                ]}],
            },
        ],
    }


def test_consensus_total_tie_break_picks_median_not_first_seen_line() -> None:
    """Regressão do bug real: com bookmaker_count empatado (1x1x1), o consenso deve cair
    na linha mais próxima da mediana (9.5), não na primeira linha vista (8.5) nem na
    isolada/destoante (11.0)."""
    client = _FakeOddsApiClient([_event_with_thin_disjoint_totals_lines()])
    service = OddsService(api_client=client)
    game = service.get_all_game_odds()[0]

    consensus = game.consensus_total
    assert consensus is not None
    assert consensus.point == 9.5


def test_event_without_totals_market_returns_empty_totals_list() -> None:
    event = _sample_event()
    for bookmaker in event["bookmakers"]:
        bookmaker["markets"] = [m for m in bookmaker["markets"] if m["key"] != "totals"]
    client = _FakeOddsApiClient([event])
    service = OddsService(api_client=client)

    game = service.get_all_game_odds()[0]
    assert game.totals == []
    assert game.consensus_total is None


def test_implied_probability_is_inverse_of_decimal_odds() -> None:
    assert implied_probability(2.0) == pytest.approx(0.5)
    assert implied_probability(1.5) == pytest.approx(2 / 3)


def test_implied_probability_handles_zero_gracefully() -> None:
    assert implied_probability(0.0) == 0.0


def _sample_team_totals_event_odds() -> Dict[str, Any]:
    """Baseado na estrutura real observada no endpoint por evento: outcomes trazem
    "description" com o nome do time a que o Over/Under se refere."""
    return {
        "id": "evt1",
        "bookmakers": [
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "team_totals",
                        "outcomes": [
                            {"name": "Over", "description": "New York Yankees", "price": 1.83, "point": 3.5},
                            {"name": "Under", "description": "New York Yankees", "price": 1.94, "point": 3.5},
                            {"name": "Over", "description": "Los Angeles Dodgers", "price": 1.88, "point": 4.5},
                            {"name": "Under", "description": "Los Angeles Dodgers", "price": 1.88, "point": 4.5},
                        ],
                    }
                ],
            },
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "team_totals",
                        "outcomes": [
                            {"name": "Over", "description": "New York Yankees", "price": 1.90, "point": 3.5},
                            {"name": "Under", "description": "New York Yankees", "price": 1.87, "point": 3.5},
                            {"name": "Over", "description": "Los Angeles Dodgers", "price": 1.80, "point": 4.5},
                            {"name": "Under", "description": "Los Angeles Dodgers", "price": 1.95, "point": 4.5},
                        ],
                    }
                ],
            },
        ],
    }


def test_get_team_totals_parses_both_teams() -> None:
    client = _FakeOddsApiClient(events=[], event_odds={"evt1": _sample_team_totals_event_odds()})
    service = OddsService(api_client=client)

    result = service.get_team_totals("evt1", home_team="New York Yankees", away_team="Los Angeles Dodgers")

    assert result.home is not None
    assert result.home.team == "New York Yankees"
    assert result.home.point == 3.5
    assert result.away is not None
    assert result.away.team == "Los Angeles Dodgers"
    assert result.away.point == 4.5


def test_get_team_totals_picks_best_price_per_side() -> None:
    client = _FakeOddsApiClient(events=[], event_odds={"evt1": _sample_team_totals_event_odds()})
    service = OddsService(api_client=client)

    result = service.get_team_totals("evt1", home_team="New York Yankees", away_team="Los Angeles Dodgers")

    # Yankees Over: DraftKings 1.90 > FanDuel 1.83
    assert result.home.over_price == pytest.approx(1.90)
    assert result.home.over_bookmaker == "DraftKings"
    # Yankees Under: FanDuel 1.94 > DraftKings 1.87
    assert result.home.under_price == pytest.approx(1.94)
    assert result.home.under_bookmaker == "FanDuel"


def test_get_team_totals_returns_none_for_team_without_data() -> None:
    client = _FakeOddsApiClient(events=[], event_odds={"evt1": {"bookmakers": []}})
    service = OddsService(api_client=client)

    result = service.get_team_totals("evt1", home_team="New York Yankees", away_team="Los Angeles Dodgers")

    assert result.home is None
    assert result.away is None


def test_game_odds_includes_event_id() -> None:
    client = _FakeOddsApiClient([_sample_event()])
    service = OddsService(api_client=client)
    game = service.get_all_game_odds()[0]

    assert game.event_id == "abc123"
