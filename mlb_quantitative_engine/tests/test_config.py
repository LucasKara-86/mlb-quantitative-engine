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
    """Bate com DEFAULT_N_SIMULATIONS (analytics/monte_carlo.py), o valor realmente usado
    em produção -- não os 100_000 antigos, que nunca chegavam a ser lidos pelo código."""
    assert settings.monte_carlo_simulations == 20_000


def test_value_bet_thresholds_match_specification() -> None:
    """min_confidence está na mesma escala 0-100 de ValueBet.confidence_score (não 0-1)."""
    assert settings.min_expected_value == 0.05
    assert settings.min_edge == 0.04
    assert settings.min_confidence == 70.0


def test_tunable_params_loaded_from_json() -> None:
    """Parâmetros ajustáveis pelo auto-tuning vêm de tunable_params.json, não hardcoded."""
    assert settings.overdispersion == 1.4
    assert settings.mean_uncertainty_pct == 0.12
    assert settings.kelly_fraction_multiplier == 0.25
    assert settings.max_stake_fraction == 0.02
    assert settings.price_tolerance == 0.05
    assert settings.starter_expected_innings == 5.33
    assert settings.low_winpct_over_threshold == 0.445
    assert settings.high_winpct_under_threshold == 0.555


def test_env_var_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "test-key-123")
    overridden = Settings(_env_file=None)
    assert overridden.odds_api_key == "test-key-123"
