from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BASE_DIR.parent


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

    # Weather API (provedor a definir na etapa de clima)
    weather_api_key: str = Field(default="", description="Chave de acesso do provedor meteorológico")

    # Telegram (alertas de Value Bet)
    telegram_bot_token: str = Field(default="", description="Token do bot do Telegram")
    telegram_channel_id: str = Field(default="", description="Canal/chat de destino dos alertas (ex.: @canal)")

    # Banco de dados
    database_path: Path = BASE_DIR / "database" / "database.db"

    # Simulação Monte Carlo
    monte_carlo_simulations: int = 100_000

    # Critérios mínimos para classificar uma Value Bet (conforme especificação do projeto)
    min_expected_value: float = 0.05
    min_edge: float = 0.04
    min_confidence: float = 0.70

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
