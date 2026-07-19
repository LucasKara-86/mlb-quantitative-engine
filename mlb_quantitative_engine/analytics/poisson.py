from __future__ import annotations

"""Camada de probabilidade de corridas totais, a partir de uma média projetada.

Distribuição usada: Binomial Negativa (NB2), não Poisson pura.

Raciocínio estatístico (achado real de produção, não hipotético):
- Um lote de 9 apostas recomendadas pelo modelo (todas com probabilidade
  projetada entre 64% e 89%) perdeu 8 das 9 no mesmo dia — com erros de
  projeção enormes E NAS DUAS DIREÇÕES no mesmo lote de jogos (ex.: projetei
  7.76 corridas pra um time que foi shutout; projetei 2.63 pra outro que
  fez 7). Investigar caso a caso descartou "bug de dado errado" (titulares e
  placares batiam) — o padrão é consistente com o modelo tratando a média
  projetada como mais confiável do que ela realmente é.
- A contagem de corridas por jogo de MLB tem variância historicamente MAIOR
  que a média (overdispersion) — o efeito de "innings explosivas" (um time
  fizer 5 corridas numa única entrada) quebra a premissa central da Poisson
  de que Var(X) = E[X]. Ignorar isso faz o modelo relatar probabilidades
  mais extremas (mais perto de 0% ou 100%) do que a realidade sustenta, ou
  seja, confiança sistematicamente inflada — exatamente o padrão observado.
- A Binomial Negativa generaliza a Poisson permitindo Var(X) = média *
  overdispersão (overdispersão >= 1.0; overdispersão = 1.0 recupera a
  Poisson exatamente, no limite). Parametrização usada (ver `_nb_params`):
  dado média µ e overdispersão φ, p = 1/φ e r = µ/(φ-1).
- `DEFAULT_OVERDISPERSION = 1.4` é um valor inicial de literatura pública de
  sabermetria (variância de corridas por jogo historicamente ~1.3-1.6x a
  média) — não foi calibrado com os dados reais deste projeto ainda. Assim
  que o harness de calibração (analytics/calibration.py) acumular amostra
  suficiente de apostas resolvidas (GREEN/RED/PUSH), este valor deve ser
  recalibrado para bater a taxa de acerto real por faixa de probabilidade.
- Esta camada ainda trata a média projetada como um número exato — a
  incerteza sobre a PRÓPRIA média (erro de estimativa de wRC+/FIP) é tratada
  em analytics/monte_carlo.py, que empilha em cima desta distribuição.
"""

import math
from typing import Dict, Tuple

from scipy.stats import nbinom
from scipy.stats import poisson as _poisson

DEFAULT_OVERDISPERSION: float = 1.4


def _nb_params(mean: float, overdispersion: float) -> Tuple[float, float]:
    """Converte (média, overdispersão) nos parâmetros (r, p) da Binomial Negativa
    usados por scipy.stats.nbinom, de forma que Var(X) = média * overdispersão.

    Requer overdispersion > 1.0 (o limite overdispersion -> 1.0 é a Poisson pura,
    tratado separadamente pelos chamadores para evitar r -> infinito)."""
    p = 1.0 / overdispersion
    r = mean / (overdispersion - 1.0)
    return r, p


def probability_over(
    projected_mean: float, total_line: float, overdispersion: float = DEFAULT_OVERDISPERSION
) -> float:
    """P(corridas totais > total_line) para uma variável ~ NB(média=projected_mean,
    overdispersão=overdispersion). overdispersion<=1.0 cai de volta para Poisson pura.

    Linhas de mercado tipicamente terminam em .5 (ex.: 8.5) para evitar push;
    P(X > 8.5) = P(X >= 9) = 1 - P(X <= 8) = 1 - CDF(8). Usar floor(total_line)
    generaliza corretamente também para linhas inteiras.
    """
    if projected_mean <= 0:
        return 0.0
    threshold = math.floor(total_line)
    if overdispersion <= 1.0:
        return float(1.0 - _poisson.cdf(threshold, projected_mean))
    r, p = _nb_params(projected_mean, overdispersion)
    return float(1.0 - nbinom.cdf(threshold, r, p))


def probability_under(
    projected_mean: float, total_line: float, overdispersion: float = DEFAULT_OVERDISPERSION
) -> float:
    """P(corridas totais < total_line). Ignora a possibilidade de push em linhas inteiras."""
    if projected_mean <= 0:
        return 1.0
    threshold = math.floor(total_line)
    is_integer_line = total_line == threshold
    lookup = threshold - 1 if is_integer_line else threshold
    if overdispersion <= 1.0:
        return float(_poisson.cdf(lookup, projected_mean))
    r, p = _nb_params(projected_mean, overdispersion)
    return float(nbinom.cdf(lookup, r, p))


def score_distribution(
    projected_mean: float, max_runs: int = 20, overdispersion: float = DEFAULT_OVERDISPERSION
) -> Dict[int, float]:
    """Distribuição de probabilidade P(corridas totais = k) para k em [0, max_runs]."""
    if projected_mean <= 0:
        return {0: 1.0}
    if overdispersion <= 1.0:
        return {runs: float(_poisson.pmf(runs, projected_mean)) for runs in range(max_runs + 1)}
    r, p = _nb_params(projected_mean, overdispersion)
    return {runs: float(nbinom.pmf(runs, r, p)) for runs in range(max_runs + 1)}


def confidence_interval(
    projected_mean: float, confidence: float = 0.95, overdispersion: float = DEFAULT_OVERDISPERSION
) -> Tuple[int, int]:
    """Intervalo de corridas totais (inteiros) que cobre `confidence` de probabilidade acumulada."""
    if projected_mean <= 0:
        return (0, 0)
    lower_tail = (1 - confidence) / 2
    if overdispersion <= 1.0:
        lower = int(_poisson.ppf(lower_tail, projected_mean))
        upper = int(_poisson.ppf(1 - lower_tail, projected_mean))
        return lower, upper
    r, p = _nb_params(projected_mean, overdispersion)
    lower = int(nbinom.ppf(lower_tail, r, p))
    upper = int(nbinom.ppf(1 - lower_tail, r, p))
    return lower, upper
