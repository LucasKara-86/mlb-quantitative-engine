from __future__ import annotations

"""Constrói o relatório de backtest (ROI, hit rate, profit factor, drawdown, Sharpe
adaptado, CLV -- ver analytics/backtesting.py) a partir das apostas resolvidas no banco.

Ponte equivalente à de services/calibration_report_service.py, mas para
analytics/backtesting.py (que existia matematicamente completo, mas nunca tinha sido
conectado ao banco). Usa a mesma fonte de dados (`outcome` preenchido por
services/bet_result_checker.py) e, além do resultado geral, quebra por:
- família de mercado (game_total / home_team_total / away_team_total, ignorando o lado
  Over/Under -- é a família que importa para achar um segmento estruturalmente ruim, não
  o lado específico de uma aposta);
- faixa de confidence_score (mesmos cortes de 5 em 5 pontos usados no critério de Value
  Bet, de 70 em diante -- abaixo de 70 a aposta nunca teria sido enviada).

CLV (`closing_price`) fica sempre None: este projeto ainda não persiste a odd de
fechamento do mercado, só a odd tomada no momento do alerta -- `average_clv` do resultado
sai None em todos os relatórios até essa captura existir (analytics/backtesting.py já lida
com isso sem quebrar as demais métricas).
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from mlb_quantitative_engine.analytics.backtesting import BacktestResult, SettledBet, compute_backtest_metrics
from mlb_quantitative_engine.database.models import Game, ValueBet
from mlb_quantitative_engine.database.repository import Repository

_CONFIDENCE_BUCKET_EDGES: Tuple[float, ...] = (70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.01)
_MARKET_PREFIXES: Tuple[str, ...] = ("game_total", "home_team_total", "away_team_total")


@dataclass(frozen=True)
class BacktestReport:
    """Resultado de backtest geral e segmentado a partir de um conjunto de apostas resolvidas."""

    overall: BacktestResult
    by_market: Dict[str, BacktestResult]
    by_confidence_bucket: Dict[str, BacktestResult]


def _market_group(market: str) -> str:
    for prefix in _MARKET_PREFIXES:
        if market.startswith(prefix):
            return prefix
    return market


def _confidence_bucket(confidence_score: float) -> str:
    edges = _CONFIDENCE_BUCKET_EDGES
    for lower, upper in zip(edges, edges[1:]):
        if lower <= confidence_score < upper:
            return f"{lower:.0f}-{min(upper, 100.0):.0f}"
    return f"{edges[0]:.0f}+"


def _settled_bet_from_row(value_bet: ValueBet, game: Game) -> SettledBet:
    return SettledBet(
        date=game.game_date,
        game_pk=game.game_pk,
        market=value_bet.market,
        price=value_bet.price,
        stake=value_bet.suggested_stake_fraction,
        outcome=value_bet.outcome,
        closing_price=None,
    )


def build_backtest_report(
    repository: Optional[Repository] = None,
    resolved_rows: Optional[Sequence[Tuple[ValueBet, Game]]] = None,
) -> BacktestReport:
    """Monta o relatório a partir de `resolved_rows` (pares ValueBet+Game já resolvidos,
    outcome preenchido) quando informado -- usado por reports/daily_analysis_runner.py para
    recortar só um dia específico -- ou, por padrão, de todo o histórico do banco."""
    if resolved_rows is None:
        repo = repository or Repository()
        resolved_rows = repo.list_resolved_value_bets_with_game()

    pairs: List[Tuple[ValueBet, SettledBet]] = [
        (value_bet, _settled_bet_from_row(value_bet, game)) for value_bet, game in resolved_rows
    ]

    overall = compute_backtest_metrics([settled for _, settled in pairs])

    market_groups: Dict[str, List[SettledBet]] = defaultdict(list)
    confidence_groups: Dict[str, List[SettledBet]] = defaultdict(list)
    for value_bet, settled in pairs:
        market_groups[_market_group(value_bet.market)].append(settled)
        confidence_groups[_confidence_bucket(value_bet.confidence_score)].append(settled)

    by_market = {group: compute_backtest_metrics(bets) for group, bets in market_groups.items()}
    by_confidence_bucket = {
        bucket: compute_backtest_metrics(bets) for bucket, bets in confidence_groups.items()
    }

    return BacktestReport(overall=overall, by_market=by_market, by_confidence_bucket=by_confidence_bucket)


def format_backtest_report(report: BacktestReport) -> str:
    """Formata o relatório em texto simples para console/log, no mesmo estilo de
    services/calibration_report_service.format_calibration_report."""
    lines = ["=== Relatório de Backtest ==="]
    overall = report.overall
    if overall.total_bets == 0:
        lines.append("Sem apostas resolvidas ainda -- nada para calcular.")
        return "\n".join(lines)

    lines.append(
        f"Geral: {overall.total_bets} apostas (V={overall.wins} D={overall.losses} P={overall.pushes}) | "
        f"Hit rate: {overall.hit_rate:.1%} | ROI: {overall.roi:+.1%} | "
        f"Profit Factor: {overall.profit_factor if overall.profit_factor is not None else 'N/D'} | "
        f"Max Drawdown: {overall.max_drawdown:.1%}"
    )
    lines.append("")
    lines.append("Por mercado:")
    for group, result in sorted(report.by_market.items()):
        lines.append(f"  {group:<20} n={result.total_bets:<4} hit={result.hit_rate:.1%}  roi={result.roi:+.1%}")
    lines.append("")
    lines.append("Por faixa de confiança:")
    for bucket, result in sorted(report.by_confidence_bucket.items()):
        lines.append(f"  {bucket:<10} n={result.total_bets:<4} hit={result.hit_rate:.1%}  roi={result.roi:+.1%}")

    return "\n".join(lines)
