from __future__ import annotations

"""Cliente para a The Odds API (https://the-odds-api.com/), v4.

Cada chamada a /odds consome créditos da cota mensal da chave (custo =
nº de mercados x nº de regiões solicitadas, independente do número de jogos
retornados). O cliente registra a cota restante informada nos headers de
resposta (`x-requests-remaining`) para permitir monitorar o consumo.

Mercado `team_totals` (total por time) NÃO está disponível no endpoint bulk
`/sports/{sport}/odds` — exige o endpoint por evento (`/events/{id}/odds`),
que cobra 1 requisição por jogo em vez de 1 requisição para o dia inteiro
(ainda barato: testado em produção, custa 1 crédito por evento/mercado/região,
igual ao endpoint bulk — não confundir com o endpoint `/historical/`, esse
sim caro e indisponível no plano gratuito).
"""

from typing import Any, Dict, List, Optional

import requests
from cachetools import TTLCache
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mlb_quantitative_engine.config import settings
from mlb_quantitative_engine.utils.logger import log

MLB_SPORT_KEY = "baseball_mlb"


class OddsApiError(RuntimeError):
    """Erro ao comunicar com a The Odds API após esgotar as tentativas de retry."""


class OddsApiClient:
    """Cliente HTTP para a The Odds API, com retry, cache e monitoramento de cota."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
    ) -> None:
        self.base_url = base_url or settings.odds_api_base_url
        self.api_key = api_key or settings.odds_api_key
        self._cache: TTLCache = TTLCache(maxsize=64, ttl=cache_ttl_seconds or settings.cache_ttl_seconds)
        self.last_requests_used: Optional[int] = None
        self.last_requests_remaining: Optional[int] = None

        if not self.api_key:
            log.warning("ODDS_API_KEY não configurada — chamadas à The Odds API vão falhar")

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _request(self, url: str, params: Dict[str, Any]) -> requests.Response:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        return response

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        full_params = {"apiKey": self.api_key, **(params or {})}

        cache_key = (url, tuple(sorted(full_params.items())))
        if cache_key in self._cache:
            log.debug(f"Cache hit para {url}")
            return self._cache[cache_key]

        try:
            response = self._request(url, full_params)
        except requests.RequestException as exc:
            log.error(f"Falha ao consultar The Odds API em {url}: {exc}")
            raise OddsApiError(f"Falha ao consultar {url}: {exc}") from exc

        self._update_quota(response)
        payload = response.json()
        self._cache[cache_key] = payload
        return payload

    def _update_quota(self, response: requests.Response) -> None:
        used = response.headers.get("x-requests-used")
        remaining = response.headers.get("x-requests-remaining")
        if used is not None:
            self.last_requests_used = int(used)
        if remaining is not None:
            self.last_requests_remaining = int(remaining)
            if self.last_requests_remaining < 50:
                log.warning(f"Cota da The Odds API baixa: {self.last_requests_remaining} requisições restantes")

    def get_mlb_odds(
        self,
        regions: str = "us",
        markets: str = "h2h,totals",
        odds_format: str = "decimal",
    ) -> List[Dict[str, Any]]:
        """Retorna as odds brutas de todos os jogos de MLB agendados (mercados h2h e totals)."""
        return self._get(
            f"sports/{MLB_SPORT_KEY}/odds",
            params={"regions": regions, "markets": markets, "oddsFormat": odds_format},
        )

    def get_mlb_events(self) -> List[Dict[str, Any]]:
        """Retorna a lista de eventos (jogos) de MLB agendados, sem odds — usado para
        descobrir o event_id necessário para consultar mercados adicionais por evento."""
        return self._get(f"sports/{MLB_SPORT_KEY}/events")

    def get_event_odds(
        self,
        event_id: str,
        regions: str = "us",
        markets: str = "team_totals",
        odds_format: str = "decimal",
    ) -> Dict[str, Any]:
        """Retorna odds de mercados adicionais (ex.: team_totals) para um evento específico.

        Custa 1 requisição por evento consultado (diferente do endpoint bulk, que cobre
        o dia inteiro numa única chamada) — usar com moderação para não gastar cota à toa.
        """
        return self._get(
            f"sports/{MLB_SPORT_KEY}/events/{event_id}/odds",
            params={"regions": regions, "markets": markets, "oddsFormat": odds_format},
        )
