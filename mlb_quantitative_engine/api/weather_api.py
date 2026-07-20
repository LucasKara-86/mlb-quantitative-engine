from __future__ import annotations

"""Cliente para a API pública Open-Meteo (https://open-meteo.com/) -- previsão do tempo
horária por coordenadas geográficas. Gratuita e sem necessidade de chave de API, ao
contrário das demais integrações externas deste projeto (Odds API, Telegram).
"""

from typing import Any, Dict, Optional

import requests
from cachetools import TTLCache
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mlb_quantitative_engine.config import settings
from mlb_quantitative_engine.utils.logger import log


class WeatherApiError(RuntimeError):
    """Erro ao comunicar com a API de previsão do tempo após esgotar as tentativas de retry."""


class WeatherApiClient:
    """Cliente HTTP para a Open-Meteo. Mesmo padrão de retry/cache dos demais clientes
    (ver api/mlb_api.py) -- cache por coordenada evita rebuscar a previsão do mesmo
    estádio repetidamente dentro da janela de TTL."""

    def __init__(self, base_url: Optional[str] = None, cache_ttl_seconds: Optional[int] = None) -> None:
        self.base_url = base_url or settings.weather_api_base_url
        self._cache: TTLCache = TTLCache(maxsize=128, ttl=cache_ttl_seconds or settings.cache_ttl_seconds)

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.get(self.base_url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()

    def get_hourly_forecast(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Retorna a previsão horária bruta (temperatura, vento) para os próximos dias
        na coordenada informada."""
        cache_key = (round(latitude, 3), round(longitude, 3))
        if cache_key in self._cache:
            log.debug(f"Cache hit para previsão do tempo em {cache_key}")
            return self._cache[cache_key]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "UTC",
        }
        try:
            payload = self._request(params)
        except requests.RequestException as exc:
            log.error(f"Falha ao consultar previsão do tempo em {self.base_url}: {exc}")
            raise WeatherApiError(f"Falha ao consultar previsão do tempo: {exc}") from exc

        self._cache[cache_key] = payload
        return payload
