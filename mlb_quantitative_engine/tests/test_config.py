from __future__ import annotations

import json
from pathlib import Path

from mlb_quantitative_engine.config import Settings, TUNABLE_PARAMS_PATH, get_settings, settings


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
    """Parâmetros ajustáveis vêm de tunable_params.json, não hardcoded -- valida que
    `settings` reflete o CONTEÚDO ATUAL do arquivo. Não fixa valores literais de propósito:
    `overdispersion`/`mean_uncertainty_pct` (e outros) são reescritos pelo auto-tuning
    (services/auto_tuning_service.py), então pinar um valor exato aqui faria o gate de
    testes rejeitar todo ajuste desses parâmetros -- justamente o que este teste deveria
    proteger. Compara-se contra o JSON, que é a fonte da verdade."""
    on_disk = json.loads(TUNABLE_PARAMS_PATH.read_text(encoding="utf-8"))
    tunable_attributes = (
        "overdispersion",
        "mean_uncertainty_pct",
        "kelly_fraction_multiplier",
        "max_stake_fraction",
        "price_tolerance",
        "starter_expected_innings",
        "low_winpct_over_threshold",
        "high_winpct_under_threshold",
    )
    for name in tunable_attributes:
        assert name in on_disk, f"{name} ausente em tunable_params.json"
        assert getattr(settings, name) == on_disk[name], (
            f"settings.{name}={getattr(settings, name)!r} não reflete "
            f"tunable_params.json[{name!r}]={on_disk[name]!r}"
        )


def test_env_var_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ODDS_API_KEY", "test-key-123")
    overridden = Settings(_env_file=None)
    assert overridden.odds_api_key == "test-key-123"
