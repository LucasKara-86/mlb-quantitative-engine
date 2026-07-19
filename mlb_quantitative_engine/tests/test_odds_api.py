from __future__ import annotations

from typing import Any, Dict

import requests
import tenacity.nap

from mlb_quantitative_engine.api.odds_api import OddsApiClient, OddsApiError


class _FakeResponse:
    def __init__(self, payload: Any, headers: Dict[str, str] | None = None) -> None:
        self._payload = payload
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


def _sample_odds_payload():
    return [
        {
            "id": "abc123",
            "home_team": "New York Yankees",
            "away_team": "Los Angeles Dodgers",
            "commence_time": "2026-07-17T23:05:00Z",
            "bookmakers": [],
        }
    ]


def test_get_mlb_odds_returns_raw_events(monkeypatch) -> None:
    monkeypatch.setattr(
        "mlb_quantitative_engine.api.odds_api.requests.get",
        lambda *args, **kwargs: _FakeResponse(_sample_odds_payload(), {"x-requests-used": "5", "x-requests-remaining": "495"}),
    )
    client = OddsApiClient(api_key="test-key")
    events = client.get_mlb_odds()

    assert len(events) == 1
    assert events[0]["home_team"] == "New York Yankees"


def test_quota_headers_are_tracked(monkeypatch) -> None:
    monkeypatch.setattr(
        "mlb_quantitative_engine.api.odds_api.requests.get",
        lambda *args, **kwargs: _FakeResponse(_sample_odds_payload(), {"x-requests-used": "7", "x-requests-remaining": "493"}),
    )
    client = OddsApiClient(api_key="test-key")
    client.get_mlb_odds()

    assert client.last_requests_used == 7
    assert client.last_requests_remaining == 493


def test_response_is_cached_within_ttl(monkeypatch) -> None:
    call_count = {"count": 0}

    def fake_get(*args: Any, **kwargs: Any) -> _FakeResponse:
        call_count["count"] += 1
        return _FakeResponse(_sample_odds_payload(), {"x-requests-used": "1", "x-requests-remaining": "499"})

    monkeypatch.setattr("mlb_quantitative_engine.api.odds_api.requests.get", fake_get)
    client = OddsApiClient(api_key="test-key", cache_ttl_seconds=60)

    client.get_mlb_odds()
    client.get_mlb_odds()

    assert call_count["count"] == 1


def test_api_key_is_sent_as_query_param(monkeypatch) -> None:
    captured_params = {}

    def fake_get(url: str, params=None, **kwargs) -> _FakeResponse:
        captured_params.update(params or {})
        return _FakeResponse(_sample_odds_payload(), {})

    monkeypatch.setattr("mlb_quantitative_engine.api.odds_api.requests.get", fake_get)
    client = OddsApiClient(api_key="my-secret-key")
    client.get_mlb_odds()

    assert captured_params["apiKey"] == "my-secret-key"


def test_request_retries_and_then_raises_odds_api_error(monkeypatch) -> None:
    monkeypatch.setattr(tenacity.nap, "sleep", lambda seconds: None)

    call_count = {"count": 0}

    def failing_get(*args: Any, **kwargs: Any) -> None:
        call_count["count"] += 1
        raise requests.ConnectionError("boom")

    monkeypatch.setattr("mlb_quantitative_engine.api.odds_api.requests.get", failing_get)
    client = OddsApiClient(api_key="test-key")

    try:
        client.get_mlb_odds()
        assert False, "esperava OddsApiError"
    except OddsApiError:
        pass

    assert call_count["count"] == 3


def test_get_mlb_events_returns_raw_event_list(monkeypatch) -> None:
    payload = [
        {"id": "evt1", "home_team": "New York Yankees", "away_team": "Los Angeles Dodgers", "commence_time": "2026-07-18T23:05:00Z"},
    ]
    monkeypatch.setattr(
        "mlb_quantitative_engine.api.odds_api.requests.get",
        lambda *args, **kwargs: _FakeResponse(payload),
    )
    client = OddsApiClient(api_key="test-key")
    events = client.get_mlb_events()

    assert len(events) == 1
    assert events[0]["id"] == "evt1"


def test_get_event_odds_requests_correct_endpoint_and_params(monkeypatch) -> None:
    captured = {}

    def fake_get(url: str, params=None, **kwargs) -> _FakeResponse:
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse({"id": "evt1", "bookmakers": []})

    monkeypatch.setattr("mlb_quantitative_engine.api.odds_api.requests.get", fake_get)
    client = OddsApiClient(api_key="test-key")
    result = client.get_event_odds("evt1", markets="team_totals")

    assert result == {"id": "evt1", "bookmakers": []}
    assert captured["url"].endswith("/events/evt1/odds")
    assert captured["params"]["markets"] == "team_totals"
    assert captured["params"]["apiKey"] == "test-key"
