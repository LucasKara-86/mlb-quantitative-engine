from __future__ import annotations

"""Estado do bullpen de um time: fadiga recente, disponibilidade e qualidade agregada.

Raciocínio e decisões de design:

- Um arremessador é tratado como "bullpen" nesta etapa quando não teve nenhuma
  abertura na temporada (`gamesStarted == 0`). A MLB Stats API não rotula o
  "papel" de cada arremessador (closer/setup/long relief) diretamente, então:
    - Closer = reliever do time com mais saves na temporada (se saves > 0).
    - Setup = reliever com mais holds na temporada, excluindo o closer.
  São heurísticas razoáveis, não uma classificação oficial da MLB.

- Uso recente vem do game log de cada arremessador (`stats=gameLog`), olhando
  a data de cada aparição e as innings lançadas naquele jogo. Um reliever é
  considerado "provavelmente indisponível" quando lançou em cada um dos
  últimos 3 dias consecutivos, ou lançou >= 1.1 innings no dia anterior —
  regras de bullpen usage bem estabelecidas na prática ("não usar um
  arremessador 3 dias seguidos"), não uma regra oficial publicada pela MLB.

- Bullpen Fatigue Index (0-100): índice PRÓPRIO deste projeto (não é uma
  métrica sabermétrica publicada por FanGraphs/Baseball Savant). Pondera as
  innings lançadas nos últimos 4 dias por recência (peso maior para "ontem"
  que para "há 4 dias"), soma entre todos os relievers do time e normaliza
  contra FATIGUE_SCALE. É uma aproximação transparente e calibrável do
  cansaço agregado do bullpen — não deve ser lida como uma métrica com
  significado absoluto fora deste projeto.

- ERA/WHIP/FIP do bullpen são agregados a partir das linhas de estatística
  brutas de temporada de cada reliever (soma de contagens, nunca média das
  taxas individuais — ver aggregate_pitching_stat_lines).
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from mlb_quantitative_engine.analytics.sabermetrics import (
    LeagueConstants,
    PitchingMetrics,
    aggregate_pitching_stat_lines,
    compute_pitching_metrics,
)
from mlb_quantitative_engine.api.mlb_api import MLBApiClient
from mlb_quantitative_engine.services.pitching_service import PitchingService
from mlb_quantitative_engine.utils.logger import log


@dataclass(frozen=True)
class RelieverStatus:
    """Estado de um arremessador de bullpen específico."""

    player_id: int
    full_name: str
    season_saves: int
    season_holds: int
    innings_pitched_last_1d: float
    innings_pitched_last_2d: float
    innings_pitched_last_3d: float
    innings_pitched_last_4d: float
    is_likely_unavailable: bool


@dataclass(frozen=True)
class BullpenStatus:
    """Estado agregado do bullpen de um time para uma data de referência."""

    team_id: int
    reference_date: str
    relievers: List[RelieverStatus]
    closer_player_id: Optional[int]
    setup_player_id: Optional[int]
    unavailable_count: int
    fatigue_index: float
    metrics: Optional[PitchingMetrics]


class BullpenService:
    """Consulta o roster de um time e monta o estado do bullpen a partir do game log de cada reliever."""

    CONSECUTIVE_DAYS_THRESHOLD = 3
    HEAVY_OUTING_INNINGS_THRESHOLD = 1.1
    RECENCY_WEIGHTS: Dict[int, float] = {1: 3.0, 2: 2.0, 3: 1.0, 4: 0.5}
    FATIGUE_SCALE = 24.0  # soma ponderada acima da qual o índice satura em 100

    def __init__(self, api_client: Optional[MLBApiClient] = None) -> None:
        self.api_client = api_client or MLBApiClient()
        self.pitching_service = PitchingService(self.api_client)

    def get_bullpen_status(
        self,
        team_id: int,
        reference_date: Optional[str] = None,
        season: Optional[int] = None,
        constants: LeagueConstants = LeagueConstants(),
    ) -> BullpenStatus:
        ref_date = self._parse_date(reference_date)
        target_season = season or ref_date.year

        roster = self.api_client.get_team_roster(team_id, roster_type="active")
        pitcher_entries = [p for p in roster if p.get("position", {}).get("abbreviation") == "P"]

        relievers: List[RelieverStatus] = []
        raw_stat_lines = []
        closer_id, closer_saves = None, 0
        setup_id, setup_holds = None, 0

        for entry in pitcher_entries:
            player_id = entry["person"]["id"]
            full_name = entry["person"]["fullName"]

            season_stats = self.api_client.get_player_season_stats(player_id, group="pitching", season=target_season)
            if not season_stats or int(season_stats.get("gamesStarted", 0)) > 0:
                continue  # abriu ao menos um jogo na temporada -> não é bullpen puro nesta etapa

            stat_line = PitchingService._parse_pitching_stat_line(season_stats)
            if stat_line is not None:
                raw_stat_lines.append(stat_line)

            saves = int(season_stats.get("saves", 0))
            holds = int(season_stats.get("holds", 0))
            if saves > closer_saves:
                closer_id, closer_saves = player_id, saves
            if holds > setup_holds:
                setup_id, setup_holds = player_id, holds

            recent_innings = self._recent_innings_by_days_ago(player_id, target_season, ref_date)
            is_unavailable = (
                all(recent_innings.get(d, 0.0) > 0 for d in (1, 2, 3))
                or recent_innings.get(1, 0.0) >= self.HEAVY_OUTING_INNINGS_THRESHOLD
            )

            relievers.append(
                RelieverStatus(
                    player_id=player_id,
                    full_name=full_name,
                    season_saves=saves,
                    season_holds=holds,
                    innings_pitched_last_1d=recent_innings.get(1, 0.0),
                    innings_pitched_last_2d=recent_innings.get(2, 0.0),
                    innings_pitched_last_3d=recent_innings.get(3, 0.0),
                    innings_pitched_last_4d=recent_innings.get(4, 0.0),
                    is_likely_unavailable=is_unavailable,
                )
            )

        # setup não pode ser a mesma pessoa que o closer
        if setup_id == closer_id:
            setup_id = None

        metrics = None
        if raw_stat_lines:
            aggregated = aggregate_pitching_stat_lines(raw_stat_lines)
            metrics = compute_pitching_metrics(aggregated, constants)

        fatigue_index = self._compute_fatigue_index(relievers)
        unavailable_count = sum(1 for r in relievers if r.is_likely_unavailable)

        if not relievers:
            log.warning(f"Nenhum reliever identificado para o time {team_id}")

        return BullpenStatus(
            team_id=team_id,
            reference_date=ref_date.isoformat(),
            relievers=relievers,
            closer_player_id=closer_id if closer_saves > 0 else None,
            setup_player_id=setup_id if setup_holds > 0 else None,
            unavailable_count=unavailable_count,
            fatigue_index=fatigue_index,
            metrics=metrics,
        )

    def _recent_innings_by_days_ago(
        self, player_id: int, season: int, reference_date: date
    ) -> Dict[int, float]:
        """Soma innings pitched por 'dias atrás' (1-4) em relação à reference_date, a partir do game log."""
        game_log = self.api_client.get_player_game_log(player_id, group="pitching", season=season)
        innings_by_days_ago: Dict[int, float] = {}

        for split in game_log:
            split_date = self._safe_parse_date(split.get("date"))
            if split_date is None:
                continue
            days_ago = (reference_date - split_date).days
            if days_ago < 1 or days_ago > 4:
                continue
            outs = PitchingService._innings_pitched_to_outs(str(split.get("stat", {}).get("inningsPitched", "")))
            innings_by_days_ago[days_ago] = innings_by_days_ago.get(days_ago, 0.0) + outs / 3

        return innings_by_days_ago

    def _compute_fatigue_index(self, relievers: List[RelieverStatus]) -> float:
        raw_workload = sum(
            self.RECENCY_WEIGHTS[1] * r.innings_pitched_last_1d
            + self.RECENCY_WEIGHTS[2] * r.innings_pitched_last_2d
            + self.RECENCY_WEIGHTS[3] * r.innings_pitched_last_3d
            + self.RECENCY_WEIGHTS[4] * r.innings_pitched_last_4d
            for r in relievers
        )
        return round(min(100.0, (raw_workload / self.FATIGUE_SCALE) * 100.0), 1)

    @staticmethod
    def _parse_date(reference_date: Optional[str]) -> date:
        if reference_date is None:
            return datetime.now(timezone.utc).date()
        return datetime.strptime(reference_date, "%Y-%m-%d").date()

    @staticmethod
    def _safe_parse_date(value: Optional[str]) -> Optional[date]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return None
