from __future__ import annotations

from typing import Any, Dict

import requests
import tenacity.nap

from mlb_quantitative_engine.api.mlb_api import GameSummary, MLBApiClient, MLBApiError


def _sample_schedule_payload() -> Dict[str, Any]:
    return {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 745123,
                        "officialDate": "2026-07-17",
                        "gameDate": "2026-07-17T23:05:00Z",
                        "status": {"detailedState": "Scheduled"},
                        "venue": {"name": "Yankee Stadium"},
                        "teams": {
                            "home": {
                                "team": {"name": "New York Yankees"},
                                "probablePitcher": {"id": 543037, "fullName": "Gerrit Cole"},
                            },
                            "away": {
                                "team": {"name": "Boston Red Sox"},
                                "probablePitcher": {"id": 668881, "fullName": "Brayan Bello"},
                            },
                        },
                    }
                ]
            }
        ]
    }


class _FakeResponse:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


def test_get_games_for_date_parses_summary(monkeypatch) -> None:
    monkeypatch.setattr(
        "mlb_quantitative_engine.api.mlb_api.requests.get",
        lambda *args, **kwargs: _FakeResponse(_sample_schedule_payload()),
    )
    client = MLBApiClient()
    games = client.get_games_for_date("2026-07-17")

    assert len(games) == 1
    game = games[0]
    assert isinstance(game, GameSummary)
    assert game.game_pk == 745123
    assert game.game_date == "2026-07-17"
    assert game.home_team == "New York Yankees"
    assert game.away_team == "Boston Red Sox"
    assert game.home_probable_pitcher == "Gerrit Cole"
    assert game.away_probable_pitcher == "Brayan Bello"
    assert game.home_probable_pitcher_id == 543037
    assert game.away_probable_pitcher_id == 668881
    assert game.venue == "Yankee Stadium"
    assert game.status == "Scheduled"


def test_schedule_with_no_games_returns_empty_list(monkeypatch) -> None:
    monkeypatch.setattr(
        "mlb_quantitative_engine.api.mlb_api.requests.get",
        lambda *args, **kwargs: _FakeResponse({"dates": []}),
    )
    client = MLBApiClient()
    assert client.get_games_for_date("2026-07-17") == []


def test_missing_probable_pitcher_is_none(monkeypatch) -> None:
    payload = _sample_schedule_payload()
    del payload["dates"][0]["games"][0]["teams"]["home"]["probablePitcher"]
    monkeypatch.setattr(
        "mlb_quantitative_engine.api.mlb_api.requests.get",
        lambda *args, **kwargs: _FakeResponse(payload),
    )
    client = MLBApiClient()
    game = client.get_games_for_date("2026-07-17")[0]
    assert game.home_probable_pitcher is None


def test_response_is_cached_within_ttl(monkeypatch) -> None:
    call_count = {"count": 0}

    def fake_get(*args: Any, **kwargs: Any) -> _FakeResponse:
        call_count["count"] += 1
        return _FakeResponse(_sample_schedule_payload())

    monkeypatch.setattr("mlb_quantitative_engine.api.mlb_api.requests.get", fake_get)
    client = MLBApiClient(cache_ttl_seconds=60)

    client.get_schedule_raw("2026-07-17")
    client.get_schedule_raw("2026-07-17")

    assert call_count["count"] == 1


def test_different_params_are_not_served_from_cache(monkeypatch) -> None:
    call_count = {"count": 0}

    def fake_get(*args: Any, **kwargs: Any) -> _FakeResponse:
        call_count["count"] += 1
        return _FakeResponse(_sample_schedule_payload())

    monkeypatch.setattr("mlb_quantitative_engine.api.mlb_api.requests.get", fake_get)
    client = MLBApiClient(cache_ttl_seconds=60)

    client.get_schedule_raw("2026-07-17")
    client.get_schedule_raw("2026-07-18")

    assert call_count["count"] == 2


def test_request_retries_and_then_raises_mlb_api_error(monkeypatch) -> None:
    monkeypatch.setattr(tenacity.nap, "sleep", lambda seconds: None)

    call_count = {"count": 0}

    def failing_get(*args: Any, **kwargs: Any) -> None:
        call_count["count"] += 1
        raise requests.ConnectionError("boom")

    monkeypatch.setattr("mlb_quantitative_engine.api.mlb_api.requests.get", failing_get)
    client = MLBApiClient()

    try:
        client.get_schedule_raw("2026-07-17")
        assert False, "esperava MLBApiError"
    except MLBApiError:
        pass

    assert call_count["count"] == 3


def test_get_game_feed_uses_live_base_url(monkeypatch) -> None:
    captured_urls = []

    def fake_get(url: str, *args: Any, **kwargs: Any) -> _FakeResponse:
        captured_urls.append(url)
        return _FakeResponse({"gamePk": 745123})

    monkeypatch.setattr("mlb_quantitative_engine.api.mlb_api.requests.get", fake_get)
    client = MLBApiClient()
    client.get_game_feed(745123)

    assert captured_urls[0].startswith("https://statsapi.mlb.com/api/v1.1/")
    assert captured_urls[0].endswith("game/745123/feed/live")


def test_get_player_season_stats_extracts_stat_object(monkeypatch) -> None:
    payload = {
        "stats": [
            {
                "splits": [
                    {"stat": {"atBats": 500, "hits": 150, "homeRuns": 20}}
                ]
            }
        ]
    }
    monkeypatch.setattr(
        "mlb_quantitative_engine.api.mlb_api.requests.get",
        lambda *args, **kwargs: _FakeResponse(payload),
    )
    client = MLBApiClient()
    stats = client.get_player_season_stats(592450, group="hitting", season=2026)

    assert stats == {"atBats": 500, "hits": 150, "homeRuns": 20}


def test_get_team_roster_returns_roster_list(monkeypatch) -> None:
    payload = {
        "roster": [
            {"person": {"id": 1, "fullName": "Player One"}, "position": {"abbreviation": "C"}},
            {"person": {"id": 2, "fullName": "Player Two"}, "position": {"abbreviation": "P"}},
        ]
    }
    captured_params = {}

    def fake_get(url: str, params=None, **kwargs) -> _FakeResponse:
        captured_params.update(params or {})
        return _FakeResponse(payload)

    monkeypatch.setattr("mlb_quantitative_engine.api.mlb_api.requests.get", fake_get)
    client = MLBApiClient()
    roster = client.get_team_roster(108, roster_type="active")

    assert len(roster) == 2
    assert roster[0]["person"]["fullName"] == "Player One"
    assert captured_params == {"rosterType": "active"}


def test_get_player_season_stats_returns_empty_dict_when_no_splits(monkeypatch) -> None:
    monkeypatch.setattr(
        "mlb_quantitative_engine.api.mlb_api.requests.get",
        lambda *args, **kwargs: _FakeResponse({"stats": []}),
    )
    client = MLBApiClient()
    assert client.get_player_season_stats(1, group="hitting", season=2026) == {}
