from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BASE_DIR.parent

TUNABLE_PARAMS_PATH: Path = BASE_DIR / "tunable_params.json"


def _load_tunable_params() -> dict:
    """Lê `tunable_params.json` -- os parâmetros numéricos que
    `services/auto_tuning_service.py` pode ajustar sozinho (nunca segredos: aqueles
    continuam só no `.env`, fora do git). Arquivo ausente/vazio -> {} (cai nos defaults
    de literatura/especificação abaixo, mesmo comportamento de antes deste arquivo existir).
    Lido a cada chamada (sem cache): o custo é irrelevante e evita estado obsoleto entre
    testes ou entre uma mudança do auto-tuner e o próximo `get_settings()`."""
    if not TUNABLE_PARAMS_PATH.exists():
        return {}
    return json.loads(TUNABLE_PARAMS_PATH.read_text(encoding="utf-8"))


def _tunable(name: str, default: Any) -> Any:
    return _load_tunable_params().get(name, default)


class Settings(BaseSettings):
    """Configuração centralizada e validada da aplicação, carregada a partir de variáveis de ambiente (.env)."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MLB Stats API (https://statsapi.mlb.com/api/)
    mlb_api_base_url: str = "https://statsapi.mlb.com/api/v1"
    # A MLB serve o feed ao vivo do jogo em uma versão de API diferente (v1.1) do restante dos endpoints.
    mlb_api_live_base_url: str = "https://statsapi.mlb.com/api/v1.1"

    # The Odds API
    odds_api_key: str = Field(default="", description="Chave de acesso da The Odds API")
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"

    # Weather API (Open-Meteo -- gratuita, sem necessidade de chave)
    weather_api_base_url: str = "https://api.open-meteo.com/v1/forecast"
    weather_api_key: str = Field(default="", description="Não utilizado pelo Open-Meteo; mantido por compatibilidade")

    # Telegram (alertas de Value Bet)
    telegram_bot_token: str = Field(default="", description="Token do bot do Telegram")
    telegram_channel_id: str = Field(default="", description="Canal/chat de destino dos alertas (ex.: @canal)")

    # Banco de dados
    database_path: Path = BASE_DIR / "database" / "database.db"

    # Simulação Monte Carlo (nº de simulações -- ver analytics/monte_carlo.py). Valor real
    # usado hoje em produção é 20_000 (DEFAULT_N_SIMULATIONS); este campo existia antes
    # desalinhado (100_000, nunca lido de fato pelo código) -- corrigido para bater com o
    # comportamento atual e agora efetivamente threadado via reports/report_generator.py.
    monte_carlo_simulations: int = Field(default_factory=lambda: _tunable("monte_carlo_simulations", 20_000))

    # Sobre-dispersão da Binomial Negativa e incerteza da média projetada (ver
    # analytics/poisson.py e analytics/monte_carlo.py) -- valores iniciais de literatura,
    # candidatos centrais de recalibração pelo harness de calibração/auto-tuning.
    overdispersion: float = Field(default_factory=lambda: _tunable("overdispersion", 1.4))
    mean_uncertainty_pct: float = Field(default_factory=lambda: _tunable("mean_uncertainty_pct", 0.12))

    # Critérios mínimos para classificar uma Value Bet (conforme especificação do projeto).
    # `min_confidence` está na MESMA escala 0-100 de `ValueBet.confidence_score` (não 0-1) --
    # corrigido aqui: o campo existia antes com 0.70, nunca comparado de fato com o
    # confidence_score real (o código lia uma constante separada em value_bet_calculator.py).
    min_expected_value: float = Field(default_factory=lambda: _tunable("min_expected_value", 0.05))
    min_edge: float = Field(default_factory=lambda: _tunable("min_edge", 0.04))
    min_confidence: float = Field(default_factory=lambda: _tunable("min_confidence", 70.0))

    # Kelly fracionado, teto de stake e tolerância de odd mínima aceitável (ver
    # analytics/value_bet_calculator.py).
    kelly_fraction_multiplier: float = Field(default_factory=lambda: _tunable("kelly_fraction_multiplier", 0.25))
    max_stake_fraction: float = Field(default_factory=lambda: _tunable("max_stake_fraction", 0.02))
    price_tolerance: float = Field(default_factory=lambda: _tunable("price_tolerance", 0.05))

    # Filtros de envio por qualidade do time (validados empiricamente no histórico de
    # calibração; afetam SÓ Team Total, nunca Game Total). Motivo: o modelo é
    # sistematicamente superconfiante nas duas pontas abaixo —
    #  - Team-Total OVER de time fraco (pct < low): projeta ~58% e acerta ~40%.
    #  - Team-Total UNDER de time forte (pct >= high): projeta ~53% e acerta ~14%.
    # O lado oposto de cada time segue liberado (fraco -> só Under; forte -> só Over).
    low_winpct_over_threshold: float = Field(default_factory=lambda: _tunable("low_winpct_over_threshold", 0.445))
    high_winpct_under_threshold: float = Field(default_factory=lambda: _tunable("high_winpct_under_threshold", 0.555))

    # Constantes de projeção (ver analytics/projections.py: ProjectionEngine) -- inning
    # média coberta pelo titular e penalidades de fadiga/indisponibilidade do bullpen,
    # pontos de estabilização do encolhimento bayesiano de wRC+/FIP.
    starter_expected_innings: float = Field(default_factory=lambda: _tunable("starter_expected_innings", 5.33))
    bullpen_fatigue_impact: float = Field(default_factory=lambda: _tunable("bullpen_fatigue_impact", 0.15))
    bullpen_unavailable_impact: float = Field(default_factory=lambda: _tunable("bullpen_unavailable_impact", 0.03))
    pa_stabilization_point: float = Field(default_factory=lambda: _tunable("pa_stabilization_point", 250.0))
    ip_stabilization_point: float = Field(default_factory=lambda: _tunable("ip_stabilization_point", 70.0))

    # Logging
    log_level: str = "INFO"
    log_dir: Path = BASE_DIR / "logs"

    # Cache
    cache_ttl_seconds: int = 300


@lru_cache
def get_settings() -> Settings:
    """Retorna uma instância única (singleton) e cacheada de Settings."""
    return Settings()


settings: Settings = get_settings()
