from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from mlb_quantitative_engine.services.pitching_service import PitchingService


class _FakeMLBApiClient:
    def __init__(self, stat_payload: Dict[str, Any]) -> None:
        self.stat_payload = stat_payload
        self.calls: list[tuple[int, str, Optional[int]]] = []

    def get_player_season_stats(self, person_id: int, group: str, season: Optional[int] = None) -> Dict[str, Any]:
        self.calls.append((person_id, group, season))
        return self.stat_payload


_RAW_PITCHING_STATS = {
    "inningsPitched": "180.1",
    "hits": 150,
    "earnedRuns": 80,
    "runs": 85,
    "homeRuns": 10,
    "baseOnBalls": 30,
    "intentionalWalks": 2,
    "hitBatsmen": 5,
    "strikeOuts": 100,
    "battersFaced": 750,
}


def test_get_pitching_stat_line_parses_raw_payload() -> None:
    client = _FakeMLBApiClient(_RAW_PITCHING_STATS)
    service = PitchingService(api_client=client)

    stat_line = service.get_pitching_stat_line(543037, season=2026)

    assert stat_line is not None
    assert stat_line.outs == 541  # 180 innings * 3 + 1 out
    assert stat_line.innings_pitched == pytest.approx(180.333333, rel=1e-4)
    assert stat_line.h == 150
    assert stat_line.er == 80
    assert stat_line.batters_faced == 750
    assert client.calls == [(543037, "pitching", 2026)]


def test_get_pitching_stat_line_returns_none_when_no_innings() -> None:
    client = _FakeMLBApiClient({"inningsPitched": "0.0"})
    service = PitchingService(api_client=client)

    assert service.get_pitching_stat_line(1, season=2026) is None


def test_get_pitching_stat_line_returns_none_when_payload_empty() -> None:
    client = _FakeMLBApiClient({})
    service = PitchingService(api_client=client)

    assert service.get_pitching_stat_line(1, season=2026) is None


def test_get_pitching_metrics_computes_from_stat_line() -> None:
    client = _FakeMLBApiClient(_RAW_PITCHING_STATS)
    service = PitchingService(api_client=client)

    metrics = service.get_pitching_metrics(543037, season=2026)

    assert metrics is not None
    assert metrics.k_minus_bb_percent == pytest.approx(metrics.k_percent - metrics.bb_percent)
    assert metrics.gb_percent is None  # não disponível via MLB Stats API season stats
    assert metrics.xfip is None  # depende de fly_balls, também indisponível


@pytest.mark.parametrize(
    "innings_pitched,expected_outs",
    [
        ("180.0", 540),
        ("180.1", 541),
        ("180.2", 542),
        ("0.0", 0),
        ("", 0),
    ],
)
def test_innings_pitched_to_outs_conversion(innings_pitched: str, expected_outs: int) -> None:
    assert PitchingService._innings_pitched_to_outs(innings_pitched) == expected_outs
