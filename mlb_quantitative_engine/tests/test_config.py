from __future__ import annotations

from pathlib import Path

from mlb_quantitative_engine.config import Settings, get_settings, settings


def test_settings_singleton_is_cached() -> None:
    assert get_settings() is settings


def test_default_mlb_api_base_url() -> None:
    assert settings.mlb_api_base_url == "https://statsapi.mlb.com/api/v1"


def test_database_path_is_inside_database_folder() -> None:
    assert settings.database_path.parent.name == "database"
    assert settings.database_path.name == "database.db"


def test_monte_carlo_simulations_default() -> None:
    assert settings.monte_carlo_simulations == 100_000


def test_value_bet_thresholds_match_specification() -> None:
    assert settings.min_expected_value == 0.05
    assert settings.min_edge == 0.04
    assert settings.min_confidence == 0.70


def test_env_var_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "test-key-123")
    overridden = Settings(_env_file=None)
    assert overridden.odds_api_key == "test-key-123"
