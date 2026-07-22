from __future__ import annotations

"""Calculadora de Value Bets: compara a probabilidade projetada pelo motor quantitativo
com a probabilidade implícita "justa" do mercado (após remover o vig da casa),
calculando Edge, Expected Value e Kelly Criterion.

Raciocínio estatístico:
- A probabilidade projetada de cada lado (`projected_prob_over`/`under`) vem da
  camada de Monte Carlo (analytics/monte_carlo.py), não de uma Poisson/Binomial
  Negativa "pura" — o Monte Carlo já embute tanto a sobre-dispersão real de
  corridas por jogo quanto a incerteza da própria média projetada, produzindo
  probabilidades menos extremas (mais realistas) do que tratar a média como um
  número exato. Ver monte_carlo.py para o raciocínio completo.
- A probabilidade implícita bruta de uma odd decimal é 1/odd. Mas a soma das
  probabilidades implícitas de Over+Under numa casa sempre passa de 100% — a
  diferença é o vig/juice, a margem da casa. Comparar nossa projeção contra a
  probabilidade BRUTA subestimaria sistematicamente nosso edge real. Por isso
  removemos o vig pelo método proporcional (de-vig básico):
      fair_p = raw_p / (raw_p_over + raw_p_under)
  Métodos mais sofisticados (ex.: Shin's method, que corrige o
  "favorite-longshot bias") ficam para uma etapa futura de calibração.
- Edge = probabilidade projetada - probabilidade implícita justa.
- Expected Value (EV) por unidade apostada: EV = p*D - 1, onde p é nossa
  probabilidade projetada e D é a odd decimal oferecida pelo mercado.
- Kelly Criterion: fração ótima do bankroll para maximizar crescimento
  geométrico de longo prazo: f* = (p*D - 1) / (D - 1). Kelly cheio é
  agressivo demais na prática, já que nossa estimativa de probabilidade tem
  incerteza — por isso também calculamos Kelly Fracionado (25% do Kelly
  cheio por padrão), prática usual entre apostadores quantitativos.
- Confiança: quem chama esta calculadora informa um confidence_score já
  calculado (nesta etapa do projeto, a média dos confidence_score das
  lineups dos dois times — ver services/lineup_service.py). Um sistema de
  Score mais completo (combinando ataque, pitching, bullpen, clima etc.,
  conforme a especificação) é uma etapa futura separada.
- Critérios de exibição (conforme especificação): EV > 5%, Edge > 4%,
  Confiança > 70%. O ValueBet é sempre retornado, mesmo quando não qualifica
  (`meets_criteria=False`) — para permitir registrar o histórico completo de
  avaliações, não só das apostas recomendadas. Havia também um piso de
  probabilidade projetada (>= 64%), removido depois que a camada de Monte
  Carlo (item D da correção estatística) passou a produzir probabilidades já
  mais conservadoras por padrão — o piso fixo estava barrando bets com
  edge/EV genuinamente bons só por a probabilidade final (agora mais realista)
  ficar um pouco abaixo de um corte arbitrário. Se o harness de calibração
  (analytics/calibration.py) mostrar no futuro que faixas de probabilidade
  baixa têm taxa de acerto pior que o previsto, um piso pode voltar — mas
  calibrado com dado real, não um número escolhido a priori.
- Stake sugerida: além do Kelly Fracionado "puro" (kelly_fraction_quarter),
  expomos `suggested_stake_fraction`, que aplica um teto de gestão de risco
  (2% do bankroll por aposta, por padrão) — Kelly puro pode recomendar
  frações grandes em cenários de edge muito alto, o que é imprudente na
  prática dada a incerteza do modelo. O teto é só de exibição/recomendação;
  o Kelly "puro" continua registrado para análise.
- Odd mínima aceitável: a melhor odd encontrada entre casas pode não estar
  disponível para todo mundo no momento de apostar. Em vez de exigir
  exatamente a melhor odd, expomos `minimum_acceptable_price` — um piso até
  `PRICE_TOLERANCE` (5%) abaixo da melhor odd encontrada — para que a
  recomendação funcione em qualquer casa que ofereça pelo menos esse preço,
  sem depender de uma casa específica. Por isso o ValueBet também não expõe
  mais o nome da casa como parte da recomendação (o campo `bookmaker` segue
  registrado internamente, só para auditoria/CLV, mas não deve ser
  comunicado como parte da recomendação ao usuário final).

Este módulo é puramente matemático: não depende de services/ nem de api/,
seguindo a mesma separação de analytics/ já usada em sabermetrics.py e
projections.py.
"""

from typing import Tuple

from mlb_quantitative_engine.analytics.monte_carlo import (
    DEFAULT_MEAN_UNCERTAINTY_PCT,
    DEFAULT_N_SIMULATIONS,
    simulate_total_probability,
)
from mlb_quantitative_engine.analytics.poisson import DEFAULT_OVERDISPERSION
from mlb_quantitative_engine.models.value_bet import ValueBet

MIN_EXPECTED_VALUE: float = 0.05
MIN_EDGE: float = 0.04
MIN_CONFIDENCE: float = 70.0
DEFAULT_KELLY_FRACTION_MULTIPLIER: float = 0.25
MAX_STAKE_FRACTION: float = 0.02
PRICE_TOLERANCE: float = 0.05


def remove_vig(over_price: float, under_price: float) -> Tuple[float, float]:
    """Remove o vig de um par Over/Under via de-vig proporcional, devolvendo (fair_over, fair_under)."""
    raw_over = 1.0 / over_price if over_price > 0 else 0.0
    raw_under = 1.0 / under_price if under_price > 0 else 0.0
    overround = raw_over + raw_under
    if overround <= 0:
        return 0.0, 0.0
    return raw_over / overround, raw_under / overround


def expected_value(probability: float, decimal_odds: float) -> float:
    """EV por unidade apostada: p*D - 1."""
    return probability * decimal_odds - 1.0


def calculate_edge(projected_probability: float, fair_implied_probability: float) -> float:
    """Diferença entre a probabilidade projetada pelo modelo e a probabilidade justa do mercado."""
    return projected_probability - fair_implied_probability


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    """Fração ótima do bankroll (Kelly Criterion), pelo menos 0 (nunca aposta com edge negativo)."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    f = (probability * decimal_odds - 1.0) / b
    return max(0.0, min(f, 1.0))


def cap_stake_fraction(stake_fraction: float, max_stake_fraction: float = MAX_STAKE_FRACTION) -> float:
    """Aplica o teto de gestão de risco sobre uma fração de stake já calculada."""
    return min(stake_fraction, max_stake_fraction)


def minimum_acceptable_price(best_price: float, tolerance: float = PRICE_TOLERANCE) -> float:
    """Piso de odd aceitável: até `tolerance` (5%) abaixo da melhor odd encontrada,
    para a recomendação valer em qualquer casa, não só na que ofereceu o melhor preço."""
    return best_price * (1.0 - tolerance)


def _evaluate_side(
    game_pk: int,
    home_team: str,
    away_team: str,
    side: str,
    bookmaker: str,
    price: float,
    point: float,
    projected_probability: float,
    raw_implied_probability: float,
    fair_implied_probability: float,
    confidence_score: float,
    kelly_fraction_multiplier: float,
    market_prefix: str = "game_total",
    max_stake_fraction: float = MAX_STAKE_FRACTION,
    price_tolerance: float = PRICE_TOLERANCE,
    min_expected_value: float = MIN_EXPECTED_VALUE,
    min_edge: float = MIN_EDGE,
    min_confidence: float = MIN_CONFIDENCE,
) -> ValueBet:
    edge = calculate_edge(projected_probability, fair_implied_probability)
    ev = expected_value(projected_probability, price)
    kelly = kelly_fraction(projected_probability, price)
    kelly_quarter = kelly * kelly_fraction_multiplier
    suggested_stake = cap_stake_fraction(kelly_quarter, max_stake_fraction)
    min_price = minimum_acceptable_price(price, price_tolerance)

    meets_criteria = ev > min_expected_value and edge > min_edge and confidence_score > min_confidence

    return ValueBet(
        game_pk=game_pk,
        home_team=home_team,
        away_team=away_team,
        market=f"{market_prefix}_{side}",
        bookmaker=bookmaker,
        price=price,
        point=point,
        projected_probability=round(projected_probability, 4),
        implied_probability_raw=round(raw_implied_probability, 4),
        implied_probability_fair=round(fair_implied_probability, 4),
        edge=round(edge, 4),
        expected_value=round(ev, 4),
        kelly_fraction=round(kelly, 4),
        kelly_fraction_quarter=round(kelly_quarter, 4),
        suggested_stake_fraction=round(suggested_stake, 4),
        minimum_acceptable_price=round(min_price, 4),
        confidence_score=round(confidence_score, 1),
        meets_criteria=meets_criteria,
    )


def evaluate_game_total_value_bets(
    game_pk: int,
    home_team: str,
    away_team: str,
    projected_total_runs: float,
    point: float,
    over_price: float,
    over_bookmaker: str,
    under_price: float,
    under_bookmaker: str,
    confidence_score: float,
    kelly_fraction_multiplier: float = DEFAULT_KELLY_FRACTION_MULTIPLIER,
    market_prefix: str = "game_total",
    max_stake_fraction: float = MAX_STAKE_FRACTION,
    price_tolerance: float = PRICE_TOLERANCE,
    min_expected_value: float = MIN_EXPECTED_VALUE,
    min_edge: float = MIN_EDGE,
    min_confidence: float = MIN_CONFIDENCE,
    mean_uncertainty_pct: float = DEFAULT_MEAN_UNCERTAINTY_PCT,
    overdispersion: float = DEFAULT_OVERDISPERSION,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
) -> Tuple[ValueBet, ValueBet]:
    """Avalia as duas pontas (Over e Under) de um mercado de total para um jogo.

    `projected_total_runs` é a média projetada pelo motor (analytics/projections.py) —
    pode ser o total do jogo inteiro ou o total projetado de UM time específico
    (ver evaluate_team_total_value_bets); a probabilidade em cada ponta é derivada
    via Monte Carlo (analytics/monte_carlo.py) na MESMA linha (`point`) que o mercado
    está oferecendo, para uma comparação justa. `market_prefix` rotula o mercado
    resultante (ex.: "game_total", "home_team_total", "away_team_total").

    Todos os limiares/parâmetros de cálculo têm defaults de módulo (os valores de
    especificação/literatura), mas quem orquestra a avaliação em produção
    (reports/report_generator.py) deve passá-los explicitamente a partir de
    `config.settings` -- este módulo permanece puramente matemático, sem importar
    `config` diretamente (mesma convenção do restante de analytics/).
    """
    fair_over, fair_under = remove_vig(over_price, under_price)
    raw_over = 1.0 / over_price if over_price > 0 else 0.0
    raw_under = 1.0 / under_price if under_price > 0 else 0.0

    simulation = simulate_total_probability(
        projected_total_runs, point,
        mean_uncertainty_pct=mean_uncertainty_pct, overdispersion=overdispersion, n_simulations=n_simulations,
    )
    projected_prob_over = simulation.probability_over
    projected_prob_under = simulation.probability_under

    over_bet = _evaluate_side(
        game_pk, home_team, away_team, "over", over_bookmaker, over_price, point,
        projected_prob_over, raw_over, fair_over, confidence_score, kelly_fraction_multiplier, market_prefix,
        max_stake_fraction, price_tolerance, min_expected_value, min_edge, min_confidence,
    )
    under_bet = _evaluate_side(
        game_pk, home_team, away_team, "under", under_bookmaker, under_price, point,
        projected_prob_under, raw_under, fair_under, confidence_score, kelly_fraction_multiplier, market_prefix,
        max_stake_fraction, price_tolerance, min_expected_value, min_edge, min_confidence,
    )
    return over_bet, under_bet


def evaluate_team_total_value_bets(
    game_pk: int,
    home_team: str,
    away_team: str,
    team_label: str,
    projected_team_runs: float,
    point: float,
    over_price: float,
    over_bookmaker: str,
    under_price: float,
    under_bookmaker: str,
    confidence_score: float,
    kelly_fraction_multiplier: float = DEFAULT_KELLY_FRACTION_MULTIPLIER,
    max_stake_fraction: float = MAX_STAKE_FRACTION,
    price_tolerance: float = PRICE_TOLERANCE,
    min_expected_value: float = MIN_EXPECTED_VALUE,
    min_edge: float = MIN_EDGE,
    min_confidence: float = MIN_CONFIDENCE,
    mean_uncertainty_pct: float = DEFAULT_MEAN_UNCERTAINTY_PCT,
    overdispersion: float = DEFAULT_OVERDISPERSION,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
) -> Tuple[ValueBet, ValueBet]:
    """Avalia as duas pontas (Over/Under) do mercado de Team Total para UM time específico.

    `team_label` deve ser "home_team_total" ou "away_team_total" — vira o prefixo
    do campo `market` no ValueBet resultante, para diferenciar do Game Total.
    """
    return evaluate_game_total_value_bets(
        game_pk=game_pk,
        home_team=home_team,
        away_team=away_team,
        projected_total_runs=projected_team_runs,
        point=point,
        over_price=over_price,
        over_bookmaker=over_bookmaker,
        under_price=under_price,
        under_bookmaker=under_bookmaker,
        confidence_score=confidence_score,
        kelly_fraction_multiplier=kelly_fraction_multiplier,
        market_prefix=team_label,
        max_stake_fraction=max_stake_fraction,
        price_tolerance=price_tolerance,
        min_expected_value=min_expected_value,
        min_edge=min_edge,
        min_confidence=min_confidence,
        mean_uncertainty_pct=mean_uncertainty_pct,
        overdispersion=overdispersion,
        n_simulations=n_simulations,
    )
