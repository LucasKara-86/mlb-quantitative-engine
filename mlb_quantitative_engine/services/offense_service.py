from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from mlb_quantitative_engine.analytics.sabermetrics import (
    BattingMetrics,
    BattingStatLine,
    LeagueConstants,
    aggregate_batting_stat_lines,
    compute_batting_metrics,
)
from mlb_quantitative_engine.api.mlb_api import MLBApiClient
from mlb_quantitative_engine.utils.logger import log


class OffenseService:
    """Busca estatísticas brutas de rebatedores na MLB Stats API e calcula suas métricas sabermétricas."""

    def __init__(self, api_client: Optional[MLBApiClient] = None) -> None:
        self.api_client = api_client or MLBApiClient()

    @staticmethod
    def _parse_batting_stat_line(raw: Dict[str, Any]) -> Optional[BattingStatLine]:
        """Converte o dict bruto de estatísticas de rebatedor (mesmo formato usado pela MLB
        Stats API tanto em /people/{id}/stats quanto em seasonStats.batting do boxscore)."""
        if not raw or int(raw.get("atBats", 0)) == 0:
            return None
        return BattingStatLine(
            ab=int(raw.get("atBats", 0)),
            h=int(raw.get("hits", 0)),
            doubles=int(raw.get("doubles", 0)),
            triples=int(raw.get("triples", 0)),
            hr=int(raw.get("homeRuns", 0)),
            bb=int(raw.get("baseOnBalls", 0)),
            ibb=int(raw.get("intentionalWalks", 0)),
            hbp=int(raw.get("hitByPitch", 0)),
            sf=int(raw.get("sacFlies", 0)),
            sh=int(raw.get("sacBunts", 0)),
            k=int(raw.get("strikeOuts", 0)),
        )

    def get_batting_stat_line(self, person_id: int, season: Optional[int] = None) -> Optional[BattingStatLine]:
        """Retorna as estatísticas brutas de temporada de um rebatedor, ou None se indisponíveis."""
        raw = self.api_client.get_player_season_stats(person_id, group="hitting", season=season)
        stat_line = self._parse_batting_stat_line(raw)
        if stat_line is None:
            log.warning(f"Sem estatísticas de rebatedor para o jogador {person_id} (temporada={season})")
        return stat_line

    def get_batting_metrics(
        self,
        person_id: int,
        season: Optional[int] = None,
        constants: LeagueConstants = LeagueConstants(),
    ) -> Optional[BattingMetrics]:
        """Busca as estatísticas brutas e retorna as métricas sabermétricas já calculadas."""
        stat_line = self.get_batting_stat_line(person_id, season)
        if stat_line is None:
            return None
        return compute_batting_metrics(stat_line, constants)

    def get_team_offense_metrics(
        self,
        player_ids: Sequence[int],
        season: Optional[int] = None,
        constants: LeagueConstants = LeagueConstants(),
    ) -> Optional[BattingMetrics]:
        """Agrega as estatísticas de vários rebatedores (ex.: uma lineup) em métricas de time.

        Faz uma chamada de API por jogador — usado no caminho de fallback, quando não
        há estatísticas de temporada já embutidas em outra fonte (ex.: boxscore).
        """
        stat_lines = [
            stat_line
            for player_id in player_ids
            if (stat_line := self.get_batting_stat_line(player_id, season)) is not None
        ]
        if not stat_lines:
            log.warning(f"Nenhum rebatedor com estatísticas disponíveis entre {list(player_ids)}")
            return None
        aggregated = aggregate_batting_stat_lines(stat_lines)
        return compute_batting_metrics(aggregated, constants)

    def get_team_offense_metrics_from_raw_stats(
        self,
        raw_stats_list: List[Dict[str, Any]],
        constants: LeagueConstants = LeagueConstants(),
    ) -> Optional[BattingMetrics]:
        """Agrega estatísticas de rebatedores já obtidas por outra fonte (ex.: boxscore),
        evitando uma chamada de API extra por jogador quando os dados já estão em mãos."""
        stat_lines = [
            stat_line
            for raw in raw_stats_list
            if (stat_line := self._parse_batting_stat_line(raw)) is not None
        ]
        if not stat_lines:
            log.warning("Nenhum rebatedor com estatísticas válidas na lista fornecida")
            return None
        aggregated = aggregate_batting_stat_lines(stat_lines)
        return compute_batting_metrics(aggregated, constants)
