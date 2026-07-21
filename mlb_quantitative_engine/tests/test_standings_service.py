from __future__ import annotations

from typing import Dict

from mlb_quantitative_engine.api.mlb_api import MLBApiError
from mlb_quantitative_engine.services.standings_service import StandingsService


class _FakeApiClient:
    def __init__(self, standings: Dict[int, float]) -> None:
        self.standings = standings
        self.calls = 0

    def get_standings(self, season: int) -> Dict[int, float]:
        self.calls += 1
        return self.standings


class _RaisingApiClient:
    def get_standings(self, season: int) -> Dict[int, float]:
        raise MLBApiError("rede indisponível")


def _service(standings: Dict[int, float]) -> StandingsService:
    return StandingsService(
        api_client=_FakeApiClient(standings),
        low_winpct_over_threshold=0.445,
        high_winpct_under_threshold=0.555,
    )


def test_weak_team_blocks_over() -> None:
    # 0.392 < 0.445 -> suprime OVER
    assert _service({10: 0.392}).get_blocked_team_market_sides(season=2026) == {10: "over"}


def test_strong_team_blocks_under() -> None:
    # 0.630 >= 0.555 -> suprime UNDER
    assert _service({20: 0.630}).get_blocked_team_market_sides(season=2026) == {20: "under"}


def test_mid_team_blocks_nothing() -> None:
    # 0.500 está entre os limiares -> nada suprimido
    assert _service({30: 0.500}).get_blocked_team_market_sides(season=2026) == {}


def test_boundaries_are_inclusive_exclusive_as_documented() -> None:
    # exatamente no low (0.445) NÃO bloqueia over (é estritamente menor); exatamente no
    # high (0.555) bloqueia under (é >=)
    sides = _service({1: 0.445, 2: 0.555, 3: 0.4449}).get_blocked_team_market_sides(season=2026)
    assert sides == {2: "under", 3: "over"}


def test_mixed_league_classifies_each_team() -> None:
    sides = _service({1: 0.392, 2: 0.500, 3: 0.600}).get_blocked_team_market_sides(season=2026)
    assert sides == {1: "over", 3: "under"}


def test_api_failure_fails_open_to_empty() -> None:
    service = StandingsService(api_client=_RaisingApiClient())
    assert service.get_win_pct_by_team(season=2026) == {}
    assert service.get_blocked_team_market_sides(season=2026) == {}


def test_thresholds_default_from_settings() -> None:
    service = StandingsService(api_client=_FakeApiClient({}))
    from mlb_quantitative_engine.config import settings

    assert service.low_winpct_over_threshold == settings.low_winpct_over_threshold
    assert service.high_winpct_under_threshold == settings.high_winpct_under_threshold
