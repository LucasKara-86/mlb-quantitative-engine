from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValueBet:
    """Avaliação de uma aposta específica: probabilidade projetada pelo modelo vs. mercado.

    Estrutura de dados pura — o cálculo (de-vig, edge, EV, Kelly) acontece em
    analytics/value_bet_calculator.py. Sempre representa UMA avaliação (não
    necessariamente recomendada); `meets_criteria` indica se ela passou nos
    limiares mínimos da especificação (EV > 5%, Edge > 4%, Confiança > 70%).

    `bookmaker` e `price` registram a MELHOR odd encontrada, mantidos para
    auditoria/CLV — mas não devem ser comunicados como parte da recomendação
    ao usuário final (qualquer casa serve, desde que ofereça pelo menos
    `minimum_acceptable_price`). `suggested_stake_fraction` e
    `minimum_acceptable_price` são os valores realmente recomendáveis.
    """

    game_pk: int
    home_team: str
    away_team: str
    market: str  # ex.: "game_total_over", "game_total_under"
    bookmaker: str
    price: float
    point: float
    projected_probability: float
    implied_probability_raw: float
    implied_probability_fair: float
    edge: float
    expected_value: float
    kelly_fraction: float
    kelly_fraction_quarter: float
    suggested_stake_fraction: float
    minimum_acceptable_price: float
    confidence_score: float
    meets_criteria: bool


def describe_market(market: str, home_team: str, away_team: str) -> str:
    """Traduz o identificador de mercado (ex.: "home_team_total_over") para um rótulo
    legível, substituindo "home_team_total"/"away_team_total" pelo nome real do time
    e "game_total" por "Jogo". Reaproveitado pelos alertas do Telegram."""
    label = market.replace("home_team_total", home_team).replace("away_team_total", away_team)
    label = label.replace("game_total", "Jogo")
    return label.replace("_over", " Over").replace("_under", " Under")
