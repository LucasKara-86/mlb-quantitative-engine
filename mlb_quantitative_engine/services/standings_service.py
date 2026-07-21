from __future__ import annotations

"""Serviço de classificação (standings) da temporada — usado apenas pelos filtros de
ENVIO por qualidade do time no ReportGenerator.

Motivação (validada empiricamente no histórico de calibração deste projeto):
- O modelo é sistematicamente superconfiante em duas pontas específicas de Team Total:
    * Team-Total OVER de time FRACO (aproveitamento < `low_winpct_over_threshold`):
      projeta ~58% de probabilidade e a taxa real de acerto foi ~40%.
    * Team-Total UNDER de time FORTE (aproveitamento >= `high_winpct_under_threshold`):
      projeta ~53% e a taxa real foi ~14%.
- A correção é um filtro de ENVIO (não mexe na matemática da projeção, que ainda está
  em validação): continua-se avaliando e persistindo TODAS as pontas para calibração;
  só se deixa de RECOMENDAR o lado viciado. O lado oposto de cada time segue liberado
  (time fraco -> só Under; time forte -> só Over). Game Total nunca é afetado.

O aproveitamento (winning percentage) vem da MLB Stats API (ver MLBApiClient.get_standings),
que já aplica cache/retry. Falha de rede é tratada como "nada bloqueado" (fail-open): um
problema de classificação nunca deve derrubar o envio de alertas — apenas, na pior das
hipóteses, deixar passar uma aposta que teríamos filtrado, o que é logado.
"""

from datetime import datetime, timezone
from typing import Dict, Optional

from mlb_quantitative_engine.api.mlb_api import MLBApiClient, MLBApiError
from mlb_quantitative_engine.config import settings
from mlb_quantitative_engine.utils.logger import log


class StandingsService:
    """Expõe, por time, qual ponta de Team Total deve ser suprimida no envio."""

    def __init__(
        self,
        api_client: Optional[MLBApiClient] = None,
        low_winpct_over_threshold: Optional[float] = None,
        high_winpct_under_threshold: Optional[float] = None,
    ) -> None:
        self.api_client = api_client or MLBApiClient()
        self.low_winpct_over_threshold = (
            low_winpct_over_threshold
            if low_winpct_over_threshold is not None
            else settings.low_winpct_over_threshold
        )
        self.high_winpct_under_threshold = (
            high_winpct_under_threshold
            if high_winpct_under_threshold is not None
            else settings.high_winpct_under_threshold
        )

    def get_win_pct_by_team(self, season: Optional[int] = None) -> Dict[int, float]:
        """{team_id: winning_percentage}. Fail-open: devolve {} se a API falhar."""
        target_season = season or datetime.now(timezone.utc).year
        try:
            return self.api_client.get_standings(target_season)
        except MLBApiError as exc:
            log.warning(f"Não foi possível buscar a classificação ({target_season}); "
                        f"filtros por qualidade do time desativados nesta passada: {exc}")
            return {}

    def get_blocked_team_market_sides(self, season: Optional[int] = None) -> Dict[int, str]:
        """{team_id: lado_suprimido}, onde lado_suprimido é "over" (time fraco) ou
        "under" (time forte). Um time nunca cai nos dois grupos (os limiares não se
        cruzam), então cada time bloqueado mapeia para exatamente um lado.

        Times "no meio" (entre os dois limiares) não aparecem — nada é suprimido para eles.
        """
        blocked: Dict[int, str] = {}
        for team_id, win_pct in self.get_win_pct_by_team(season).items():
            if win_pct < self.low_winpct_over_threshold:
                blocked[team_id] = "over"
            elif win_pct >= self.high_winpct_under_threshold:
                blocked[team_id] = "under"
        return blocked
