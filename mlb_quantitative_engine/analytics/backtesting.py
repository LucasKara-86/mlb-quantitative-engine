from __future__ import annotations

"""Motor de métricas de backtest: mede a performance histórica de uma estratégia
de apostas a partir de uma lista de apostas já resolvidas (jogo terminado).

Raciocínio estatístico:
- ROI e Yield, em apostas esportivas, são convencionalmente a MESMA fórmula
  (Lucro Total / Total Apostado) — diferente de finanças tradicionais, onde
  podem divergir. Expomos os dois nomes no resultado por clareza semântica
  com a especificação, mas o valor numérico é idêntico.
- Profit Factor = Lucro Bruto (soma dos ganhos) / Perda Bruta (soma das
  perdas, em módulo). Métrica emprestada de trading; None quando não há
  nenhuma perda registrada (divisão por zero não tem significado útil aqui).
- Max Drawdown = maior queda percentual entre um pico e um vale subsequente
  na curva de capital acumulada — mede o pior "tombo" que o bankroll sofreu.
- Sharpe Ratio ADAPTADO = média dos retornos por aposta / desvio padrão dos
  retornos por aposta. É uma adaptação (não o Sharpe financeiro clássico,
  que assume uma taxa livre de risco e composição temporal) — mede
  consistência do retorno por aposta, não risco de portfólio.
- CLV (Closing Line Value) = quanto a odd que conseguimos foi melhor que a
  odd de fechamento do mercado, em % da odd de fechamento. CLV positivo
  consistente é considerado, na literatura de apostas quantitativas, um
  indicador de habilidade preditiva mais confiável que o resultado dos jogos
  no curto prazo (que tem muito ruído). Requer a odd de fechamento
  (closing_price); quando ausente, a aposta é ignorada no cálculo do CLV
  médio sem afetar as demais métricas.
- Hit Rate = vitórias / (vitórias + derrotas), ignorando pushes (empate na
  aposta, stake devolvido).

Este módulo é puramente matemático — não depende de services/ nem de api/.
A orquestração (buscar resultados reais, decidir quais apostas "aconteceram")
fica em uma camada de mais alto nível.
"""

import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class SettledBet:
    """Uma aposta já resolvida (o jogo terminou), pronta para entrar no cálculo de backtest."""

    date: str
    game_pk: int
    market: str
    price: float
    stake: float  # em unidades de bankroll (ex.: fração do bankroll ou valor absoluto)
    outcome: str  # "win", "loss" ou "push"
    closing_price: Optional[float] = None  # odd de fechamento do mercado, para CLV


@dataclass(frozen=True)
class BacktestResult:
    """Métricas de performance agregadas de um conjunto de apostas resolvidas."""

    total_bets: int
    wins: int
    losses: int
    pushes: int
    hit_rate: float
    total_staked: float
    total_profit: float
    roi: float
    yield_pct: float
    profit_factor: Optional[float]
    max_drawdown: float
    sharpe_ratio: Optional[float]
    average_clv: Optional[float]
    equity_curve: List[float] = field(default_factory=list)


def settle_bet_profit(bet: SettledBet) -> float:
    """Lucro (ou prejuízo) de uma aposta resolvida, em unidades de stake."""
    if bet.outcome == "win":
        return bet.stake * (bet.price - 1.0)
    if bet.outcome == "loss":
        return -bet.stake
    return 0.0  # push: stake devolvido, sem lucro nem prejuízo


def calculate_clv(price_taken: float, closing_price: float) -> float:
    """CLV: quanto a odd que conseguimos foi melhor que a odd de fechamento.

    Positivo = conseguimos uma odd melhor (mais alta) do que o mercado fechou.
    """
    if closing_price <= 0:
        return 0.0
    return (price_taken - closing_price) / closing_price


def _max_drawdown(equity_curve: Sequence[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)
    return max_dd


def _sharpe_ratio(returns: Sequence[float]) -> Optional[float]:
    if len(returns) < 2:
        return None
    stdev = statistics.pstdev(returns)
    if stdev == 0:
        return None
    return statistics.mean(returns) / stdev


def compute_backtest_metrics(bets: Sequence[SettledBet], initial_bankroll: float = 1.0) -> BacktestResult:
    """Calcula todas as métricas de performance a partir de uma lista de apostas resolvidas."""
    if not bets:
        return BacktestResult(
            total_bets=0, wins=0, losses=0, pushes=0, hit_rate=0.0,
            total_staked=0.0, total_profit=0.0, roi=0.0, yield_pct=0.0,
            profit_factor=None, max_drawdown=0.0, sharpe_ratio=None,
            average_clv=None, equity_curve=[initial_bankroll],
        )

    profits = [settle_bet_profit(bet) for bet in bets]
    wins = sum(1 for bet in bets if bet.outcome == "win")
    losses = sum(1 for bet in bets if bet.outcome == "loss")
    pushes = sum(1 for bet in bets if bet.outcome == "push")
    decided = wins + losses
    hit_rate = wins / decided if decided else 0.0

    total_staked = sum(bet.stake for bet in bets)
    total_profit = sum(profits)
    roi = total_profit / total_staked if total_staked else 0.0

    gross_profit = sum(p for p in profits if p > 0)
    gross_loss = -sum(p for p in profits if p < 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    equity_curve = [initial_bankroll]
    running = initial_bankroll
    for profit in profits:
        running += profit
        equity_curve.append(running)

    returns = [profit / bet.stake for profit, bet in zip(profits, bets) if bet.stake > 0]

    clv_values = [
        calculate_clv(bet.price, bet.closing_price) for bet in bets if bet.closing_price is not None
    ]
    average_clv = sum(clv_values) / len(clv_values) if clv_values else None

    return BacktestResult(
        total_bets=len(bets),
        wins=wins,
        losses=losses,
        pushes=pushes,
        hit_rate=round(hit_rate, 4),
        total_staked=round(total_staked, 4),
        total_profit=round(total_profit, 4),
        roi=round(roi, 4),
        yield_pct=round(roi, 4),
        profit_factor=round(profit_factor, 4) if profit_factor is not None else None,
        max_drawdown=round(_max_drawdown(equity_curve), 4),
        sharpe_ratio=round(_sharpe_ratio(returns), 4) if _sharpe_ratio(returns) is not None else None,
        average_clv=round(average_clv, 4) if average_clv is not None else None,
        equity_curve=[round(v, 4) for v in equity_curve],
    )
