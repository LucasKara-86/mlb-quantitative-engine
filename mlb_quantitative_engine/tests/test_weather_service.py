from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from mlb_quantitative_engine.api.weather_api import WeatherApiError
from mlb_quantitative_engine.services.weather_service import (
    MAX_WEATHER_FACTOR,
    MIN_WEATHER_FACTOR,
    TEMPERATURE_BASELINE_F,
    WeatherService,
    calculate_temperature_factor,
    calculate_wind_factor,
    outbound_wind_component,
)


def _forecast(temp: float, wind_speed: float = 5.0, wind_dir: float = 180.0) -> Dict[str, Any]:
    return {
        "hourly": {
            "time": ["2026-07-20T17:00", "2026-07-20T18:00", "2026-07-20T19:00"],
            "temperature_2m": [temp - 2, temp, temp + 2],
            "wind_speed_10m": [wind_speed, wind_speed, wind_speed],
            "wind_direction_10m": [wind_dir, wind_dir, wind_dir],
        }
    }


class _FakeWeatherApiClient:
    def __init__(self, forecast: Optional[Dict[str, Any]] = None, raise_error: bool = False) -> None:
        self.forecast = forecast
        self.raise_error = raise_error
        self.calls = 0

    def get_hourly_forecast(self, latitude: float, longitude: float) -> Dict[str, Any]:
        self.calls += 1
        if self.raise_error:
            raise WeatherApiError("simulado")
        return self.forecast


# --- calculate_temperature_factor (matemática pura) ---


def test_temperature_factor_is_neutral_at_baseline() -> None:
    assert calculate_temperature_factor(TEMPERATURE_BASELINE_F) == pytest.approx(1.0)


def test_temperature_factor_increases_with_heat() -> None:
    assert calculate_temperature_factor(90.0) > 1.0


def test_temperature_factor_decreases_with_cold() -> None:
    assert calculate_temperature_factor(50.0) < 1.0


def test_temperature_factor_is_clamped_to_bounds() -> None:
    assert calculate_temperature_factor(200.0) == pytest.approx(MAX_WEATHER_FACTOR)
    assert calculate_temperature_factor(-50.0) == pytest.approx(MIN_WEATHER_FACTOR)


# --- WeatherService.get_weather_conditions ---


def test_climate_controlled_venue_is_always_neutral() -> None:
    client = _FakeWeatherApiClient(_forecast(temp=95.0))  # calor extremo, mas domo fechado
    service = WeatherService(api_client=client)

    conditions = service.get_weather_conditions("Rogers Centre", "2026-07-20T18:00:00Z")

    assert conditions.factor == 1.0
    assert conditions.climate_controlled is True
    assert client.calls == 0  # nem precisa consultar previsão


def test_open_air_venue_uses_forecast_temperature() -> None:
    client = _FakeWeatherApiClient(_forecast(temp=90.0))
    service = WeatherService(api_client=client)

    conditions = service.get_weather_conditions("Yankee Stadium", "2026-07-20T18:00:00Z")

    assert conditions.temperature_f == pytest.approx(90.0)
    assert conditions.factor > 1.0
    assert conditions.climate_controlled is False


def test_unknown_venue_falls_back_to_neutral() -> None:
    client = _FakeWeatherApiClient(_forecast(temp=90.0))
    service = WeatherService(api_client=client)

    conditions = service.get_weather_conditions("Estádio Inexistente", "2026-07-20T18:00:00Z")

    assert conditions.factor == 1.0
    assert client.calls == 0


def test_missing_game_datetime_falls_back_to_neutral() -> None:
    client = _FakeWeatherApiClient(_forecast(temp=90.0))
    service = WeatherService(api_client=client)

    conditions = service.get_weather_conditions("Yankee Stadium", None)

    assert conditions.factor == 1.0
    assert client.calls == 0


def test_weather_api_failure_falls_back_to_neutral_without_raising() -> None:
    client = _FakeWeatherApiClient(raise_error=True)
    service = WeatherService(api_client=client)

    conditions = service.get_weather_conditions("Yankee Stadium", "2026-07-20T18:00:00Z")

    assert conditions.factor == 1.0


def test_picks_the_forecast_hour_closest_to_game_start() -> None:
    forecast = {
        "hourly": {
            "time": ["2026-07-20T17:00", "2026-07-20T18:00", "2026-07-20T19:00"],
            "temperature_2m": [70.0, 95.0, 70.0],
            "wind_speed_10m": [5.0, 5.0, 5.0],
            "wind_direction_10m": [180.0, 180.0, 180.0],
        }
    }
    client = _FakeWeatherApiClient(forecast)
    service = WeatherService(api_client=client)

    conditions = service.get_weather_conditions("Yankee Stadium", "2026-07-20T18:05:00Z")

    assert conditions.temperature_f == pytest.approx(95.0)


# --- vento: funções puras ---


def test_outbound_wind_component_is_positive_when_blowing_out_to_center() -> None:
    """Vento vindo de trás do home (origem = bearing_cf + 180) sopra campo afora -> positivo."""
    # cf_bearing 25 -> vento OUT vem de 205
    assert outbound_wind_component(20.0, wind_from_deg=205.0, cf_bearing_deg=25.0) == pytest.approx(20.0, abs=0.1)


def test_outbound_wind_component_is_negative_when_blowing_in_from_center() -> None:
    """Vento vindo do centro do campo (origem = bearing_cf) sopra campo adentro -> negativo."""
    assert outbound_wind_component(20.0, wind_from_deg=25.0, cf_bearing_deg=25.0) == pytest.approx(-20.0, abs=0.1)


def test_outbound_wind_component_is_near_zero_for_crosswind() -> None:
    """Vento perpendicular ao eixo home->centro quase não tem componente afora/adentro."""
    assert outbound_wind_component(20.0, wind_from_deg=115.0, cf_bearing_deg=25.0) == pytest.approx(0.0, abs=0.5)


def test_wind_factor_boosts_offense_for_out_blowing_wind() -> None:
    assert calculate_wind_factor(20.0, wind_from_deg=205.0, cf_bearing_deg=25.0) > 1.0


def test_wind_factor_suppresses_offense_for_in_blowing_wind() -> None:
    assert calculate_wind_factor(20.0, wind_from_deg=25.0, cf_bearing_deg=25.0) < 1.0


# --- vento: integrado no serviço ---


def test_out_blowing_wind_raises_the_factor_vs_in_blowing_wind() -> None:
    """Mesma temperatura neutra (70°F): vento campo afora deve dar fator maior que campo adentro."""
    out_wind = WeatherService(api_client=_FakeWeatherApiClient(_forecast(temp=70.0, wind_speed=20.0, wind_dir=205.0)))
    in_wind = WeatherService(api_client=_FakeWeatherApiClient(_forecast(temp=70.0, wind_speed=20.0, wind_dir=25.0)))

    out_conditions = out_wind.get_weather_conditions("Yankee Stadium", "2026-07-20T18:00:00Z")
    in_conditions = in_wind.get_weather_conditions("Yankee Stadium", "2026-07-20T18:00:00Z")

    assert out_conditions.factor > 1.0
    assert in_conditions.factor < 1.0
    assert out_conditions.factor > in_conditions.factor
    assert out_conditions.wind_speed_mph == pytest.approx(20.0)


def test_combined_factor_is_clamped_to_bounds() -> None:
    """Calor extremo + vento forte a favor não deve estourar o teto combinado."""
    extreme = WeatherService(
        api_client=_FakeWeatherApiClient(_forecast(temp=110.0, wind_speed=40.0, wind_dir=205.0))
    )
    conditions = extreme.get_weather_conditions("Yankee Stadium", "2026-07-20T18:00:00Z")
    assert conditions.factor <= MAX_WEATHER_FACTOR
