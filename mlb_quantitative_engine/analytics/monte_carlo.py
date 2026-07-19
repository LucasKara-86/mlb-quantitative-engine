from __future__ import annotations

"""Simulação de Monte Carlo: propaga a incerteza da PRÓPRIA média projetada, além da
variância de jogo já capturada pela Binomial Negativa (analytics/poisson.py).

Raciocínio estatístico:
- `poisson.probability_over/under` respondem "dado que a média projetada está
  EXATAMENTE correta, qual a chance do total passar da linha?" — mas a média em
  si é uma estimativa (produzida por wRC+/FIP já encolhidos, blend de bullpen,
  park factor), com erro de estimativa próprio que não desaparece mesmo depois do
  encolhimento bayesiano (analytics/sabermetrics.shrink_toward_league_average):
  ainda restam fontes de incerteza não modeladas (forma do titular no dia, lineup
  de última hora, clima/umpire — ainda não implementados). Tratar a média como um
  número exato faz o modelo parecer mais confiante do que deveria.
- Esta camada simula N jogos: em cada simulação, primeiro sorteia uma "média
  verdadeira" de uma Normal centrada na média projetada (desvio proporcional a
  `mean_uncertainty_pct`), depois sorteia o total de corridas daquele jogo de uma
  Binomial Negativa com essa média sorteada (mesma overdispersão de poisson.py).
  A fração de simulações acima/abaixo da linha de mercado é a probabilidade final
  — ela já embute tanto a variância de jogo (Binomial Negativa) quanto a incerteza
  do parâmetro (a Normal em torno da média), motivo de dar probabilidades ainda
  menos extremas que só a Binomial Negativa sozinha.
- `DEFAULT_MEAN_UNCERTAINTY_PCT = 0.12` é um valor inicial conservador (12% de
  desvio padrão em torno da média projetada) — não calibrado com dados reais
  deste projeto ainda. Junto com DEFAULT_OVERDISPERSION (poisson.py), deve ser
  recalibrado assim que o harness de calibração (analytics/calibration.py)
  acumular amostra suficiente de apostas resolvidas.
- Semente fixa por padrão (`DEFAULT_RANDOM_SEED`): a mesma entrada deve sempre
  produzir a mesma probabilidade — importante para reprodutibilidade em testes e
  para que reprocessar o mesmo jogo (ex.: retentativa de lineup) sem mudar a
  projeção não gere uma recomendação "diferente" por puro ruído de simulação.
  20.000 simulações mantêm o erro de simulação (`simulation_error`, o desvio
  padrão da estimativa binomial da própria probabilidade) tipicamente abaixo de
  meio ponto percentual — preciso o suficiente para os limiares de Value Bet
  (Edge > 4%, EV > 5%) sem custar mais que alguns milissegundos por avaliação.

Este módulo é puramente matemático: não depende de services/ nem de api/,
seguindo a mesma separação das demais camadas de analytics/.
"""

import math
from dataclasses import dataclass

import numpy as np

from mlb_quantitative_engine.analytics.poisson import DEFAULT_OVERDISPERSION

DEFAULT_MEAN_UNCERTAINTY_PCT: float = 0.12
DEFAULT_N_SIMULATIONS: int = 20_000
DEFAULT_RANDOM_SEED: int = 42


@dataclass(frozen=True)
class MonteCarloResult:
    """Resultado de uma simulação: probabilidades de Over/Under/Push e o erro de
    simulação (desvio padrão estimado da própria probabilidade de Over, pela
    variância de uma proporção binomial — mede o quão preciso é o resultado dado
    `n_simulations`, não incerteza do modelo em si)."""

    probability_over: float
    probability_under: float
    probability_push: float
    simulation_error: float


def simulate_total_probability(
    projected_mean: float,
    total_line: float,
    mean_uncertainty_pct: float = DEFAULT_MEAN_UNCERTAINTY_PCT,
    overdispersion: float = DEFAULT_OVERDISPERSION,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> MonteCarloResult:
    """Simula `n_simulations` jogos e devolve a fração que fica acima/abaixo/exatamente
    na linha de mercado. Ver docstring do módulo para o raciocínio estatístico completo.
    """
    if projected_mean <= 0:
        return MonteCarloResult(probability_over=0.0, probability_under=1.0, probability_push=0.0, simulation_error=0.0)

    rng = np.random.default_rng(random_seed)
    mean_std = max(projected_mean * mean_uncertainty_pct, 1e-6)
    simulated_means = np.clip(rng.normal(projected_mean, mean_std, size=n_simulations), 0.05, None)

    if overdispersion <= 1.0:
        simulated_runs = rng.poisson(simulated_means)
    else:
        p = 1.0 / overdispersion
        r = simulated_means / (overdispersion - 1.0)
        simulated_runs = rng.negative_binomial(r, p)

    threshold = math.floor(total_line)
    if total_line == threshold:
        # linha inteira: push é um resultado possível
        over_mask = simulated_runs > threshold
        under_mask = simulated_runs < threshold
        push_mask = simulated_runs == threshold
    else:
        over_mask = simulated_runs > threshold
        under_mask = ~over_mask
        push_mask = np.zeros_like(simulated_runs, dtype=bool)

    prob_over = float(over_mask.mean())
    prob_under = float(under_mask.mean())
    prob_push = float(push_mask.mean())
    simulation_error = float(math.sqrt(max(prob_over * (1.0 - prob_over), 0.0) / n_simulations))

    return MonteCarloResult(
        probability_over=prob_over,
        probability_under=prob_under,
        probability_push=prob_push,
        simulation_error=simulation_error,
    )
