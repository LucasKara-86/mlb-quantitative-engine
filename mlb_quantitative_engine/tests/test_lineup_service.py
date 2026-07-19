from __future__ import annotations

from typing import Any, Dict, List

import pytest

from mlb_quantitative_engine.services.lineup_service import LineupService


class _FakeMLBApiClient:
    def __init__(self, boxscore: Dict[str, Any], roster: List[Dict[str, Any]] | None = None) -> None:
        self.boxscore = boxscore
        self.roster = roster or []
        self.roster_calls: List[tuple[int, str]] = []

    def get_boxscore(self, game_pk: int) -> Dict[str, Any]:
        return self.boxscore

    def get_team_roster(self, team_id: int, roster_type: str = "active") -> List[Dict[str, Any]]:
        self.roster_calls.append((team_id, roster_type))
        return self.roster


def _boxscore_with_official_lineup() -> Dict[str, Any]:
    return {
        "teams": {
            "home": {
                "team": {"id": 108},
                "battingOrder": [111, 222, 333],
                "players": {
                    "ID111": {
                        "person": {"id": 111},
                        "seasonStats": {"batting": {"atBats": 300, "hits": 90}},
                    },
                    "ID222": {
                        "person": {"id": 222},
                        "seasonStats": {"batting": {"atBats": 350, "hits": 100}},
                    },
                    "ID333": {
                        "person": {"id": 333},
                        "seasonStats": {"batting": {"atBats": 400, "hits": 110}},
                    },
                },
            },
            "away": {"team": {"id": 109}, "battingOrder": [], "players": {}},
        }
    }


def test_get_batting_order_returns_official_lineup_when_available() -> None:
    client = _FakeMLBApiClient(boxscore=_boxscore_with_official_lineup())
    service = LineupService(api_client=client)

    snapshot = service.get_batting_order(game_pk=1, team_side="home")

    assert snapshot.source == "official"
    assert snapshot.confidence_score == LineupService.OFFICIAL_CONFIDENCE
    assert snapshot.player_ids == [111, 222, 333]
    assert snapshot.entries[0].batting_order == 1
    assert snapshot.has_embedded_stats


def test_get_batting_order_embeds_raw_batting_stats_from_boxscore() -> None:
    client = _FakeMLBApiClient(boxscore=_boxscore_with_official_lineup())
    service = LineupService(api_client=client)

    snapshot = service.get_batting_order(game_pk=1, team_side="home")

    assert snapshot.entries[0].raw_batting_stats == {"atBats": 300, "hits": 90}


def test_get_batting_order_falls_back_to_roster_when_no_official_lineup() -> None:
    boxscore = _boxscore_with_official_lineup()
    roster = [
        {"person": {"id": 501}, "position": {"abbreviation": "C"}},
        {"person": {"id": 502}, "position": {"abbreviation": "P"}},
        {"person": {"id": 503}, "position": {"abbreviation": "SS"}},
    ]
    client = _FakeMLBApiClient(boxscore=boxscore, roster=roster)
    service = LineupService(api_client=client)

    snapshot = service.get_batting_order(game_pk=1, team_side="away")

    assert snapshot.source == "probable_roster"
    assert snapshot.confidence_score == LineupService.PROBABLE_ROSTER_CONFIDENCE
    assert 502 not in snapshot.player_ids  # pitcher excluído
    assert set(snapshot.player_ids) == {501, 503}
    assert not snapshot.has_embedded_stats
    assert client.roster_calls == [(109, "active")]


def test_get_batting_order_rejects_invalid_team_side() -> None:
    client = _FakeMLBApiClient(boxscore=_boxscore_with_official_lineup())
    service = LineupService(api_client=client)

    with pytest.raises(ValueError):
        service.get_batting_order(game_pk=1, team_side="visitor")


def test_probable_lineup_returns_empty_snapshot_when_team_id_missing() -> None:
    boxscore = {"teams": {"home": {"battingOrder": [], "players": {}}}}
    client = _FakeMLBApiClient(boxscore=boxscore)
    service = LineupService(api_client=client)

    snapshot = service.get_batting_order(game_pk=1, team_side="home")

    assert snapshot.entries == []
    assert snapshot.confidence_score == 0
