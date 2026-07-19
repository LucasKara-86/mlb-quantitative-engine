from __future__ import annotations

"""Busca o resultado real (placar final) de um jogo já encerrado.

Peça de dados necessária para qualquer backtest: comparar a projeção/aposta
recomendada contra o que de fato aconteceu. Usa o boxscore da MLB Stats API
(gratuito, sem custo de créditos — diferente das odds).
"""

from dataclasses import dataclass
from typing import Optional

from mlb_quantitative_engine.api.mlb_api import MLBApiClient
from mlb_quantitative_engine.utils.logger import log


@dataclass(frozen=True)
class GameResult:
    """Placar final real de um jogo encerrado."""

    game_pk: int
    home_runs: int
    away_runs: int

    @property
    def total_runs(self) -> int:
        return self.home_runs + self.away_runs


class GameResultService:
    """Consulta o placar final real de um jogo, quando encerrado."""

    def __init__(self, api_client: Optional[MLBApiClient] = None) -> None:
        self.api_client = api_client or MLBApiClient()

    def get_final_score(self, game_pk: int) -> Optional[GameResult]:
        """Retorna o placar final, ou None se o jogo ainda não terminou (ou dado ausente)."""
        boxscore = self.api_client.get_boxscore(game_pk)
        home_stats = boxscore.get("teams", {}).get("home", {}).get("teamStats", {}).get("batting", {})
        away_stats = boxscore.get("teams", {}).get("away", {}).get("teamStats", {}).get("batting", {})
        home_runs = home_stats.get("runs")
        away_runs = away_stats.get("runs")

        if home_runs is None or away_runs is None:
            log.warning(f"Placar final indisponível para o jogo {game_pk} (ainda não encerrado?)")
            return None

        return GameResult(game_pk=game_pk, home_runs=int(home_runs), away_runs=int(away_runs))

    def determine_totals_outcome(self, game_pk: int, point: float) -> Optional[str]:
        """Determina o resultado de uma aposta de Game Total ("over"/"under"/"push") para uma linha."""
        result = self.get_final_score(game_pk)
        if result is None:
            return None
        return self.classify_total(result.total_runs, point)

    def determine_team_total_outcome(self, game_pk: int, team_side: str, point: float) -> Optional[str]:
        """Determina o resultado de uma aposta de Team Total ("over"/"under"/"push") para
        um time específico (`team_side`: "home" ou "away") numa linha."""
        result = self.get_final_score(game_pk)
        if result is None:
            return None
        actual_runs = result.home_runs if team_side == "home" else result.away_runs
        return self.classify_total(actual_runs, point)

    @staticmethod
    def classify_total(actual_runs: float, point: float) -> str:
        """Compara um total real (do jogo ou de um time) contra uma linha de mercado."""
        if actual_runs > point:
            return "over"
        if actual_runs < point:
            return "under"
        return "push"
