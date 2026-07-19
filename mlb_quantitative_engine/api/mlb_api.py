from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from cachetools import TTLCache
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mlb_quantitative_engine.config import settings
from mlb_quantitative_engine.utils.logger import log


class MLBApiError(RuntimeError):
    """Erro ao comunicar com a MLB Stats API após esgotar as tentativas de retry."""


@dataclass
class GameSummary:
    """Resumo normalizado de um jogo agendado, extraído do endpoint de schedule."""

    game_pk: int
    game_date: str
    game_datetime: Optional[str]
    home_team: str
    away_team: str
    venue: Optional[str]
    status: Optional[str]
    home_probable_pitcher: Optional[str]
    away_probable_pitcher: Optional[str]
    home_probable_pitcher_id: Optional[int]
    away_probable_pitcher_id: Optional[int]
    home_team_id: Optional[int]
    away_team_id: Optional[int]


class MLBApiClient:
    """Cliente HTTP para a MLB Stats API (https://statsapi.mlb.com/api/).

    Aplica retry automático com backoff exponencial (tenacity) para falhas de
    rede transitórias e cache local de curta duração (cachetools) para evitar
    chamadas redundantes ao mesmo endpoint dentro da janela de TTL.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        live_base_url: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
    ) -> None:
        self.base_url = base_url or settings.mlb_api_base_url
        self.live_base_url = live_base_url or settings.mlb_api_live_base_url
        self._cache: TTLCache = TTLCache(maxsize=256, ttl=cache_ttl_seconds or settings.cache_ttl_seconds)

    def _build_url(self, endpoint: str, base_url: Optional[str] = None) -> str:
        root = base_url or self.base_url
        return f"{root.rstrip('/')}/{endpoint.lstrip('/')}"

    @retry(
        retry=retry_if_exception_type(requests.RequestException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        return response.json()

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, base_url: Optional[str] = None) -> Dict[str, Any]:
        url = self._build_url(endpoint, base_url)
        cache_key = (url, tuple(sorted((params or {}).items())))
        if cache_key in self._cache:
            log.debug(f"Cache hit para {url}")
            return self._cache[cache_key]

        try:
            payload = self._request(url, params)
        except requests.RequestException as exc:
            log.error(f"Falha ao consultar MLB Stats API em {url}: {exc}")
            raise MLBApiError(f"Falha ao consultar {url}: {exc}") from exc

        self._cache[cache_key] = payload
        return payload

    def get_schedule_raw(self, date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retorna o payload bruto dos jogos do dia, hidratado com times e arremessadores prováveis."""
        target_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        payload = self._get(
            "schedule",
            params={
                "sportId": 1,
                "date": target_date,
                "hydrate": "probablePitcher,team,venue",
            },
        )
        dates = payload.get("dates", [])
        return dates[0].get("games", []) if dates else []

    def get_games_for_date(self, date: Optional[str] = None) -> List[GameSummary]:
        """Retorna os jogos do dia já normalizados em GameSummary."""
        raw_games = self.get_schedule_raw(date)
        return [self._parse_game(game) for game in raw_games]

    @staticmethod
    def _parse_game(game: Dict[str, Any]) -> GameSummary:
        teams = game.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})
        game_datetime = game.get("gameDate")
        return GameSummary(
            game_pk=game["gamePk"],
            game_date=game.get("officialDate") or (game_datetime[:10] if game_datetime else ""),
            game_datetime=game_datetime,
            home_team=home.get("team", {}).get("name", "Home"),
            away_team=away.get("team", {}).get("name", "Away"),
            venue=game.get("venue", {}).get("name"),
            status=game.get("status", {}).get("detailedState"),
            home_probable_pitcher=home.get("probablePitcher", {}).get("fullName"),
            away_probable_pitcher=away.get("probablePitcher", {}).get("fullName"),
            home_probable_pitcher_id=home.get("probablePitcher", {}).get("id"),
            away_probable_pitcher_id=away.get("probablePitcher", {}).get("id"),
            home_team_id=home.get("team", {}).get("id"),
            away_team_id=away.get("team", {}).get("id"),
        )

    def get_game_feed(self, game_pk: int) -> Dict[str, Any]:
        """Retorna o feed ao vivo completo de um jogo específico (play-by-play, boxscore embutido, clima)."""
        return self._get(f"game/{game_pk}/feed/live", base_url=self.live_base_url)

    def get_boxscore(self, game_pk: int) -> Dict[str, Any]:
        """Retorna o boxscore (estatísticas por jogador) de um jogo específico."""
        return self._get(f"game/{game_pk}/boxscore")

    def get_player_season_stats(
        self, person_id: int, group: str, season: Optional[int] = None
    ) -> Dict[str, Any]:
        """Retorna o objeto de estatísticas brutas de temporada de um jogador.

        `group` deve ser "hitting" ou "pitching". Retorna um dict vazio caso o
        jogador não tenha estatísticas registradas na temporada (ex.: recém promovido).
        """
        target_season = season or datetime.now(timezone.utc).year
        payload = self._get(
            f"people/{person_id}/stats",
            params={"stats": "season", "group": group, "season": target_season},
        )
        stats = payload.get("stats", [])
        if not stats:
            return {}
        splits = stats[0].get("splits", [])
        return splits[0].get("stat", {}) if splits else {}

    def get_team_roster(self, team_id: int, roster_type: str = "active") -> List[Dict[str, Any]]:
        """Retorna a lista de jogadores do roster de um time (padrão: roster ativo).

        Usado como fallback quando a lineup titular oficial ainda não foi divulgada.
        """
        payload = self._get(f"teams/{team_id}/roster", params={"rosterType": roster_type})
        return payload.get("roster", [])

    def get_player_game_log(
        self, person_id: int, group: str, season: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retorna o histórico de aparições (jogo a jogo) de um jogador na temporada.

        Cada item tem `date` (YYYY-MM-DD) e `stat` (mesmo formato de get_player_season_stats,
        mas referente àquele jogo específico). Usado para medir uso recente de bullpen.
        """
        target_season = season or datetime.now(timezone.utc).year
        payload = self._get(
            f"people/{person_id}/stats",
            params={"stats": "gameLog", "group": group, "season": target_season},
        )
        stats = payload.get("stats", [])
        if not stats:
            return []
        return stats[0].get("splits", [])
