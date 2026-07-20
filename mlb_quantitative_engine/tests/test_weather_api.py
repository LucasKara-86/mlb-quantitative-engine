from __future__ import annotations

from typing import Any, Dict

import requests
import tenacity.nap

from mlb_quantitative_engine.api.weather_api import WeatherApiClient, WeatherApiError


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


def _sample_forecast_payload() -> Dict[str, Any]:
    return {
        "hourly": {
            "time": ["2026-07-20T18:00", "2026-07-20T19:00"],
            "temperature_2m": [82.0, 80.0],
            "wind_speed_10m": [8.5, 9.0],
            "wind_direction_10m": [200.0, 210.0],
        }
    }


def test_get_hourly_forecast_returns_raw_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        "mlb_quantitative_engine.api.weather_api.requests.get",
        lambda *args, **kwargs: _FakeResponse(_sample_forecast_payload()),
    )
    client = WeatherApiClient()
    forecast = client.get_hourly_forecast(40.83, -73.93)

    assert forecast["hourly"]["temperature_2m"] == [82.0, 80.0]


def test_request_uses_fahrenheit_and_mph_units(monkeypatch) -> None:
    captured = {}

    def fake_get(url: str, params=None, **kwargs) -> _FakeResponse:
        captured["params"] = params
        return _FakeResponse(_sample_forecast_payload())

    monkeypatch.setattr("mlb_quantitative_engine.api.weather_api.requests.get", fake_get)
    client = WeatherApiClient()
    client.get_hourly_forecast(40.83, -73.93)

    assert captured["params"]["temperature_unit"] == "fahrenheit"
    assert captured["params"]["wind_speed_unit"] == "mph"
    assert captured["params"]["latitude"] == 40.83
    assert captured["params"]["longitude"] == -73.93


def test_response_is_cached_within_ttl(monkeypatch) -> None:
    call_count = {"count": 0}

    def fake_get(*args: Any, **kwargs: Any) -> _FakeResponse:
        call_count["count"] += 1
        return _FakeResponse(_sample_forecast_payload())

    monkeypatch.setattr("mlb_quantitative_engine.api.weather_api.requests.get", fake_get)
    client = WeatherApiClient(cache_ttl_seconds=60)

    client.get_hourly_forecast(40.83, -73.93)
    client.get_hourly_forecast(40.83, -73.93)

    assert call_count["count"] == 1


def test_different_coordinates_are_not_cached_together(monkeypatch) -> None:
    call_count = {"count": 0}

    def fake_get(*args: Any, **kwargs: Any) -> _FakeResponse:
        call_count["count"] += 1
        return _FakeResponse(_sample_forecast_payload())

    monkeypatch.setattr("mlb_quantitative_engine.api.weather_api.requests.get", fake_get)
    client = WeatherApiClient(cache_ttl_seconds=60)

    client.get_hourly_forecast(40.83, -73.93)
    client.get_hourly_forecast(34.05, -118.24)

    assert call_count["count"] == 2


def test_request_retries_and_then_raises_weather_api_error(monkeypatch) -> None:
    monkeypatch.setattr(tenacity.nap, "sleep", lambda seconds: None)

    call_count = {"count": 0}

    def failing_get(*args: Any, **kwargs: Any) -> None:
        call_count["count"] += 1
        raise requests.ConnectionError("boom")

    monkeypatch.setattr("mlb_quantitative_engine.api.weather_api.requests.get", failing_get)
    client = WeatherApiClient()

    try:
        client.get_hourly_forecast(40.83, -73.93)
        assert False, "esperava WeatherApiError"
    except WeatherApiError:
        pass

    assert call_count["count"] == 3
