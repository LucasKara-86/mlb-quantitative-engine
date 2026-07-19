from __future__ import annotations

from typing import Any, Dict, Optional

from mlb_quantitative_engine.analytics.sabermetrics import (
    LeagueConstants,
    PitchingMetrics,
    PitchingStatLine,
    compute_pitching_metrics,
)
from mlb_quantitative_engine.api.mlb_api import MLBApiClient
from mlb_quantitative_engine.utils.logger import log


class PitchingService:
    """Busca estatísticas brutas de arremessadores na MLB Stats API e calcula suas métricas sabermétricas.

    Observação: a MLB Stats API não expõe a contagem de ground balls / fly balls /
    line drives rebatidos (apenas os outs resultantes deles, o que não é equivalente).
    Por isso GB%/FB% ficam indisponíveis (None) até que uma fonte de batted-ball data
    (ex.: Baseball Savant) seja integrada em uma etapa futura.
    """

    def __init__(self, api_client: Optional[MLBApiClient] = None) -> None:
        self.api_client = api_client or MLBApiClient()

    @staticmethod
    def _parse_pitching_stat_line(raw: Dict[str, Any]) -> Optional[PitchingStatLine]:
        """Converte o dict bruto de estatísticas de arremessador (mesmo formato usado pela MLB
        Stats API tanto em /people/{id}/stats quanto em cada split de gameLog)."""
        outs = PitchingService._innings_pitched_to_outs(str(raw.get("inningsPitched", "")))
        if not raw or outs == 0:
            return None
        return PitchingStatLine(
            outs=outs,
            h=int(raw.get("hits", 0)),
            er=int(raw.get("earnedRuns", 0)),
            r=int(raw.get("runs", 0)),
            hr=int(raw.get("homeRuns", 0)),
            bb=int(raw.get("baseOnBalls", 0)),
            ibb=int(raw.get("intentionalWalks", 0)),
            hbp=int(raw.get("hitBatsmen", 0)),
            k=int(raw.get("strikeOuts", 0)),
            batters_faced=int(raw.get("battersFaced", 0)),
        )

    def get_pitching_stat_line(self, person_id: int, season: Optional[int] = None) -> Optional[PitchingStatLine]:
        """Retorna as estatísticas brutas de temporada de um arremessador, ou None se indisponíveis."""
        raw = self.api_client.get_player_season_stats(person_id, group="pitching", season=season)
        stat_line = self._parse_pitching_stat_line(raw)
        if stat_line is None:
            log.warning(f"Sem estatísticas de arremessador para o jogador {person_id} (temporada={season})")
        return stat_line

    def get_pitching_metrics(
        self,
        person_id: int,
        season: Optional[int] = None,
        constants: LeagueConstants = LeagueConstants(),
    ) -> Optional[PitchingMetrics]:
        """Busca as estatísticas brutas e retorna as métricas sabermétricas já calculadas."""
        stat_line = self.get_pitching_stat_line(person_id, season)
        if stat_line is None:
            return None
        return compute_pitching_metrics(stat_line, constants)

    @staticmethod
    def _innings_pitched_to_outs(innings_pitched: str) -> int:
        """Converte o formato "180.1" da MLB Stats API (180 innings + 1 out) para outs totais."""
        if not innings_pitched:
            return 0
        whole, _, fraction = innings_pitched.partition(".")
        whole_outs = int(whole) * 3 if whole else 0
        fractional_outs = int(fraction) if fraction else 0
        return whole_outs + fractional_outs
