from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from mlb_quantitative_engine.services.bullpen_service import BullpenService


class _FakeApiClient:
    def __init__(
        self,
        roster: List[Dict[str, Any]],
        season_stats_by_id: Dict[int, Dict[str, Any]],
        game_logs_by_id: Dict[int, List[Dict[str, Any]]],
    ) -> None:
        self.roster = roster
        self.season_stats_by_id = season_stats_by_id
        self.game_logs_by_id = game_logs_by_id

    def get_team_roster(self, team_id: int, roster_type: str = "active") -> List[Dict[str, Any]]:
        return self.roster

    def get_player_season_stats(self, person_id: int, group: str, season: Optional[int] = None) -> Dict[str, Any]:
        return self.season_stats_by_id.get(person_id, {})

    def get_player_game_log(
        self, person_id: int, group: str, season: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return self.game_logs_by_id.get(person_id, [])


def _roster_entry(player_id: int, name: str) -> Dict[str, Any]:
    return {"person": {"id": player_id, "fullName": name}, "position": {"abbreviation": "P"}}


def _game_log_entry(date: str, innings_pitched: str) -> Dict[str, Any]:
    return {"date": date, "stat": {"inningsPitched": innings_pitched}}


_CLOSER_ID = 1
_SETUP_ID = 2
_STARTER_ID = 3

_ROSTER = [
    _roster_entry(_CLOSER_ID, "Closer Reliever"),
    _roster_entry(_SETUP_ID, "Setup Reliever"),
    _roster_entry(_STARTER_ID, "Rotation Starter"),
]

_SEASON_STATS = {
    _CLOSER_ID: {
        "gamesStarted": 0, "saves": 20, "holds": 0, "inningsPitched": "60.0",
        "hits": 40, "earnedRuns": 20, "runs": 22, "homeRuns": 5, "baseOnBalls": 15,
        "hitBatsmen": 2, "strikeOuts": 70, "battersFaced": 250,
    },
    _SETUP_ID: {
        "gamesStarted": 0, "saves": 0, "holds": 5, "inningsPitched": "45.0",
        "hits": 35, "earnedRuns": 18, "runs": 19, "homeRuns": 4, "baseOnBalls": 12,
        "hitBatsmen": 1, "strikeOuts": 50, "battersFaced": 190,
    },
    _STARTER_ID: {
        "gamesStarted": 10, "saves": 0, "holds": 0, "inningsPitched": "60.0",
        "hits": 55, "earnedRuns": 25, "runs": 27, "homeRuns": 6, "baseOnBalls": 18,
        "hitBatsmen": 2, "strikeOuts": 55, "battersFaced": 260,
    },
}

_GAME_LOGS = {
    _CLOSER_ID: [
        _game_log_entry("2026-07-16", "1.0"),  # 1 dia atrás
        _game_log_entry("2026-07-15", "1.0"),  # 2 dias atrás
        _game_log_entry("2026-07-14", "1.0"),  # 3 dias atrás
        _game_log_entry("2026-06-01", "1.0"),  # fora da janela de 4 dias
    ],
    _SETUP_ID: [
        _game_log_entry("2026-07-13", "1.0"),  # 4 dias atrás
    ],
    _STARTER_ID: [],
}


@pytest.fixture()
def service() -> BullpenService:
    client = _FakeApiClient(_ROSTER, _SEASON_STATS, _GAME_LOGS)
    return BullpenService(api_client=client)


def test_starter_is_excluded_from_bullpen(service: BullpenService) -> None:
    status = service.get_bullpen_status(team_id=100, reference_date="2026-07-17", season=2026)

    player_ids = [r.player_id for r in status.relievers]
    assert _STARTER_ID not in player_ids
    assert len(status.relievers) == 2


def test_closer_and_setup_are_identified_by_saves_and_holds(service: BullpenService) -> None:
    status = service.get_bullpen_status(team_id=100, reference_date="2026-07-17", season=2026)

    assert status.closer_player_id == _CLOSER_ID
    assert status.setup_player_id == _SETUP_ID


def test_reliever_unavailable_after_three_consecutive_days(service: BullpenService) -> None:
    status = service.get_bullpen_status(team_id=100, reference_date="2026-07-17", season=2026)

    closer_status = next(r for r in status.relievers if r.player_id == _CLOSER_ID)
    setup_status = next(r for r in status.relievers if r.player_id == _SETUP_ID)

    assert closer_status.is_likely_unavailable is True
    assert setup_status.is_likely_unavailable is False
    assert status.unavailable_count == 1


def test_game_log_entries_outside_four_day_window_are_ignored(service: BullpenService) -> None:
    status = service.get_bullpen_status(team_id=100, reference_date="2026-07-17", season=2026)

    closer_status = next(r for r in status.relievers if r.player_id == _CLOSER_ID)
    # a aparição de 2026-06-01 não deve contar em nenhum dos buckets de 1-4 dias
    assert closer_status.innings_pitched_last_1d == pytest.approx(1.0)
    assert closer_status.innings_pitched_last_2d == pytest.approx(1.0)
    assert closer_status.innings_pitched_last_3d == pytest.approx(1.0)
    assert closer_status.innings_pitched_last_4d == pytest.approx(0.0)


def test_fatigue_index_reflects_recency_weighted_workload(service: BullpenService) -> None:
    status = service.get_bullpen_status(team_id=100, reference_date="2026-07-17", season=2026)

    # closer: 3*1.0 + 2*1.0 + 1*1.0 = 6.0 ; setup: 0.5*1.0 = 0.5 ; total = 6.5
    expected_raw_workload = 6.5
    expected_index = round(min(100.0, expected_raw_workload / BullpenService.FATIGUE_SCALE * 100.0), 1)
    assert status.fatigue_index == pytest.approx(expected_index)


def test_bullpen_metrics_aggregate_only_relievers_not_starter(service: BullpenService) -> None:
    status = service.get_bullpen_status(team_id=100, reference_date="2026-07-17", season=2026)

    assert status.metrics is not None
    # ER agregado = 20 + 18 = 38 ; IP agregado = 60 + 45 = 105 (exclui o starter)
    expected_era = round(38 / 105 * 9, 2)
    assert status.metrics.era == pytest.approx(expected_era)


def test_heavy_outing_alone_marks_reliever_unavailable() -> None:
    roster = [_roster_entry(9, "Heavy Outing Reliever")]
    season_stats = {
        9: {
            "gamesStarted": 0, "saves": 0, "holds": 0, "inningsPitched": "40.0",
            "hits": 30, "earnedRuns": 15, "runs": 16, "homeRuns": 3, "baseOnBalls": 10,
            "hitBatsmen": 1, "strikeOuts": 35, "battersFaced": 170,
        }
    }
    game_logs = {9: [_game_log_entry("2026-07-16", "1.2")]}  # só ontem, mas outing pesado
    client = _FakeApiClient(roster, season_stats, game_logs)
    service = BullpenService(api_client=client)

    status = service.get_bullpen_status(team_id=100, reference_date="2026-07-17", season=2026)

    assert status.relievers[0].is_likely_unavailable is True


def test_empty_bullpen_returns_degenerate_status() -> None:
    client = _FakeApiClient(roster=[], season_stats_by_id={}, game_logs_by_id={})
    service = BullpenService(api_client=client)

    status = service.get_bullpen_status(team_id=100, reference_date="2026-07-17", season=2026)

    assert status.relievers == []
    assert status.closer_player_id is None
    assert status.setup_player_id is None
    assert status.fatigue_index == 0.0
    assert status.metrics is None


def test_reference_date_defaults_to_today_when_not_provided(service: BullpenService) -> None:
    status = service.get_bullpen_status(team_id=100, season=2026)
    assert status.reference_date is not None
