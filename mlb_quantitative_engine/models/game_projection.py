from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameProjection:
    """Projeção de corridas esperadas para um jogo específico, produzida pelo motor analítico.

    Estrutura de dados pura — a probabilidade de over/under e a distribuição de
    placares são derivadas em analytics/poisson.py a partir de projected_total_runs.
    """

    home_team: str
    away_team: str
    projected_home_runs: float
    projected_away_runs: float
    projected_total_runs: float
    park_factor: float = 1.0
    weather_factor: float = 1.0
