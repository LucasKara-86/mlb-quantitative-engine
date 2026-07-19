from __future__ import annotations

"""Mede se as probabilidades que o modelo diz que tem batem com o que realmente acontece —
o teste definitivo de qualquer modelo de probabilidade, e o que faltava neste projeto até
agora: o modelo emitia probabilidades (64%, 75%, 90%...) sem NUNCA verificar, com dados
reais, se apostas marcadas "75% de probabilidade" realmente ganham perto de 75% das vezes.

Raciocínio estatístico:
- Brier Score: MSE entre a probabilidade prevista e o resultado real (1=ganhou, 0=perdeu).
  Quanto menor, melhor calibrado (e mais "resolvido"/confiante corretamente) é o modelo.
  0.0 = previsões perfeitas; 0.25 = equivalente a sempre prever 50% (o "chute neutro");
  valores acima de 0.25 indicam um modelo pior que simplesmente não ter opinião.
- Tabela de confiabilidade (reliability table): agrupa as apostas resolvidas em faixas de
  probabilidade prevista (ex.: 70-75%) e compara a média prevista na faixa contra a taxa de
  acerto REAL na faixa. Um modelo bem calibrado tem essas duas colunas próximas em toda
  faixa; se a taxa real for sistematicamente menor que a prevista, o modelo está confiante
  demais (overconfident) — foi exatamente esse o problema flagrado num caso real de
  produção (8 de 9 apostas com probabilidade prevista entre 64-89% perderam no mesmo dia).
- Pushes são excluídos do cálculo (nem ganharam nem perderam — não informam calibração),
  mesma convenção já usada em analytics/backtesting.py (Hit Rate ignora pushes).
- Esta ferramenta não SUBSTITUI julgamento estatístico — com poucas dezenas de apostas
  resolvidas, o intervalo de confiança de qualquer taxa de acerto observada é enorme. Ela
  serve para ACUMULAR e vigiar a calibração ao longo do tempo, e eventualmente recalibrar
  DEFAULT_OVERDISPERSION (poisson.py) e DEFAULT_MEAN_UNCERTAINTY_PCT (monte_carlo.py) com
  dados reais em vez dos valores iniciais de literatura pública usados hoje.

Este módulo é puramente matemático: recebe uma lista de (probabilidade_prevista, ganhou),
não depende de services/ nem de api/ — a extração desses pares a partir do banco de dados
fica em services/calibration_report_service.py.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

Prediction = Tuple[float, bool]

DEFAULT_BUCKET_EDGES: Tuple[float, ...] = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01)


@dataclass(frozen=True)
class ReliabilityBucket:
    """Uma faixa de probabilidade prevista, comparando a média prevista contra a taxa de
    acerto real das apostas resolvidas que caíram nessa faixa."""

    bucket_label: str
    predicted_probability_mean: float
    actual_win_rate: float
    count: int

    @property
    def overconfidence(self) -> float:
        """Prevista - real. Positivo = modelo mais confiante do que deveria (RUIM);
        negativo = modelo mais conservador do que o necessário (deixa EV na mesa, mas
        seguro)."""
        return round(self.predicted_probability_mean - self.actual_win_rate, 4)


def brier_score(predictions: Sequence[Prediction]) -> Optional[float]:
    """Erro quadrático médio entre probabilidade prevista e resultado real (0 ou 1).
    None quando não há nenhuma previsão resolvida ainda (amostra vazia)."""
    if not predictions:
        return None
    total = sum((probability - (1.0 if won else 0.0)) ** 2 for probability, won in predictions)
    return round(total / len(predictions), 4)


def reliability_table(
    predictions: Sequence[Prediction], bucket_edges: Sequence[float] = DEFAULT_BUCKET_EDGES
) -> List[ReliabilityBucket]:
    """Agrupa as previsões em faixas de probabilidade e compara previsto x real em cada
    uma. Faixas sem nenhuma previsão são omitidas (não há o que comparar)."""
    edges = sorted(bucket_edges)
    buckets: List[ReliabilityBucket] = []

    for lower, upper in zip(edges, edges[1:]):
        bucket_predictions = [(p, won) for p, won in predictions if lower <= p < upper]
        if not bucket_predictions:
            continue
        mean_predicted = sum(p for p, _ in bucket_predictions) / len(bucket_predictions)
        win_rate = sum(1 for _, won in bucket_predictions if won) / len(bucket_predictions)
        upper_label = f"{upper:.0%}" if upper <= 1.0 else "100%+"
        buckets.append(
            ReliabilityBucket(
                bucket_label=f"{lower:.0%}-{upper_label}",
                predicted_probability_mean=round(mean_predicted, 4),
                actual_win_rate=round(win_rate, 4),
                count=len(bucket_predictions),
            )
        )
    return buckets


def overall_hit_rate(predictions: Sequence[Prediction]) -> Optional[float]:
    """Taxa de acerto geral entre todas as previsões resolvidas (ignora faixas)."""
    if not predictions:
        return None
    wins = sum(1 for _, won in predictions if won)
    return round(wins / len(predictions), 4)
