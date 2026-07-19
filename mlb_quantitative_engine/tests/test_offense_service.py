from __future__ import annotations

from typing import Any, Dict, Optional

from mlb_quantitative_engine.services.offense_service import OffenseService


class _FakeMLBApiClient:
    def __init__(self, stat_payload: Dict[str, Any]) -> None:
        self.stat_payload = stat_payload
        self.calls: list[tuple[int, str, Optional[int]]] = []

    def get_player_season_stats(self, person_id: int, group: str, season: Optional[int] = None) -> Dict[str, Any]:
        self.calls.append((person_id, group, season))
        return self.stat_payload


_RAW_HITTING_STATS = {
    "atBats": 500,
    "hits": 150,
    "doubles": 30,
    "triples": 5,
    "homeRuns": 20,
    "baseOnBalls": 60,
    "intentionalWalks": 5,
    "hitByPitch": 5,
    "sacFlies": 5,
    "sacBunts": 0,
    "strikeOuts": 100,
}


def test_get_batting_stat_line_parses_raw_payload() -> None:
    client = _FakeMLBApiClient(_RAW_HITTING_STATS)
    service = OffenseService(api_client=client)

    stat_line = service.get_batting_stat_line(592450, season=2026)

    assert stat_line is not None
    assert stat_line.ab == 500
    assert stat_line.h == 150
    assert stat_line.hr == 20
    assert stat_line.k == 100
    assert client.calls == [(592450, "hitting", 2026)]


def test_get_batting_stat_line_returns_none_when_no_at_bats() -> None:
    client = _FakeMLBApiClient({"atBats": 0})
    service = OffenseService(api_client=client)

    assert service.get_batting_stat_line(1, season=2026) is None


def test_get_batting_stat_line_returns_none_when_payload_empty() -> None:
    client = _FakeMLBApiClient({})
    service = OffenseService(api_client=client)

    assert service.get_batting_stat_line(1, season=2026) is None


def test_get_batting_metrics_computes_from_stat_line() -> None:
    client = _FakeMLBApiClient(_RAW_HITTING_STATS)
    service = OffenseService(api_client=client)

    metrics = service.get_batting_metrics(592450, season=2026)

    assert metrics is not None
    assert metrics.avg == round(150 / 500, 4)
    assert metrics.ops == round(metrics.obp + metrics.slg, 4)


def test_get_batting_metrics_returns_none_when_stat_line_missing() -> None:
    client = _FakeMLBApiClient({"atBats": 0})
    service = OffenseService(api_client=client)

    assert service.get_batting_metrics(1, season=2026) is None


def test_missing_optional_fields_default_to_zero() -> None:
    client = _FakeMLBApiClient({"atBats": 100, "hits": 25})
    service = OffenseService(api_client=client)

    stat_line = service.get_batting_stat_line(1, season=2026)

    assert stat_line is not None
    assert stat_line.hr == 0
    assert stat_line.bb == 0
    assert stat_line.k == 0


class _FakeMultiPlayerMLBApiClient:
    """Cliente-dublê que devolve estatísticas diferentes por jogador (para testar agregação)."""

    def __init__(self, stats_by_player: Dict[int, Dict[str, Any]]) -> None:
        self.stats_by_player = stats_by_player

    def get_player_season_stats(self, person_id: int, group: str, season: Optional[int] = None) -> Dict[str, Any]:
        return self.stats_by_player.get(person_id, {})


_PLAYER_A_STATS = {"atBats": 300, "hits": 90, "homeRuns": 10, "baseOnBalls": 30, "strikeOuts": 60}
_PLAYER_B_STATS = {"atBats": 400, "hits": 100, "homeRuns": 15, "baseOnBalls": 40, "strikeOuts": 90}


def test_get_team_offense_metrics_aggregates_across_players() -> None:
    client = _FakeMultiPlayerMLBApiClient({1: _PLAYER_A_STATS, 2: _PLAYER_B_STATS})
    service = OffenseService(api_client=client)

    metrics = service.get_team_offense_metrics([1, 2], season=2026)

    assert metrics is not None
    # AVG do time deve ser sobre AB e H somados (190/700), não a média simples dos AVGs individuais
    assert metrics.avg == round((90 + 100) / (300 + 400), 4)


def test_get_team_offense_metrics_skips_players_without_stats() -> None:
    client = _FakeMultiPlayerMLBApiClient({1: _PLAYER_A_STATS, 2: {"atBats": 0}})
    service = OffenseService(api_client=client)

    metrics = service.get_team_offense_metrics([1, 2], season=2026)

    assert metrics is not None
    assert metrics.avg == round(90 / 300, 4)


def test_get_team_offense_metrics_returns_none_when_no_player_has_stats() -> None:
    client = _FakeMultiPlayerMLBApiClient({})
    service = OffenseService(api_client=client)

    assert service.get_team_offense_metrics([1, 2], season=2026) is None


def test_get_team_offense_metrics_from_raw_stats_aggregates_without_extra_calls() -> None:
    service = OffenseService(api_client=None)

    metrics = service.get_team_offense_metrics_from_raw_stats([_PLAYER_A_STATS, _PLAYER_B_STATS])

    assert metrics is not None
    assert metrics.avg == round((90 + 100) / (300 + 400), 4)


def test_get_team_offense_metrics_from_raw_stats_returns_none_for_empty_list() -> None:
    service = OffenseService(api_client=None)

    assert service.get_team_offense_metrics_from_raw_stats([]) is None
