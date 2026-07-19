from __future__ import annotations

"""Constrói o relatório de calibração a partir das apostas resolvidas no banco (GREEN/RED/PUSH
já verificados por bet_result_checker.py) — a ponte entre os dados reais persistidos e a
matemática pura de analytics/calibration.py.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from mlb_quantitative_engine.analytics.calibration import (
    Prediction,
    ReliabilityBucket,
    brier_score,
    overall_hit_rate,
    reliability_table,
)
from mlb_quantitative_engine.database.repository import Repository


@dataclass(frozen=True)
class CalibrationReport:
    """Resumo de calibração do modelo a partir de todas as apostas já resolvidas."""

    total_resolved: int
    total_wins: int
    total_losses: int
    total_pushes: int
    brier_score: Optional[float]
    overall_hit_rate: Optional[float]
    reliability: List[ReliabilityBucket]


def _predictions_from_resolved_bets(resolved_bets: Sequence) -> List[Prediction]:
    """Converte ValueBets resolvidos em pares (probabilidade_prevista, ganhou), ignorando
    pushes (não informam calibração — ver analytics/calibration.py)."""
    return [
        (bet.projection_probability, bet.outcome == "win")
        for bet in resolved_bets
        if bet.outcome in ("win", "loss")
    ]


def build_calibration_report(repository: Optional[Repository] = None) -> CalibrationReport:
    repo = repository or Repository()
    resolved_bets = repo.list_resolved_value_bets()

    total_wins = sum(1 for bet in resolved_bets if bet.outcome == "win")
    total_losses = sum(1 for bet in resolved_bets if bet.outcome == "loss")
    total_pushes = sum(1 for bet in resolved_bets if bet.outcome == "push")

    predictions = _predictions_from_resolved_bets(resolved_bets)

    return CalibrationReport(
        total_resolved=len(resolved_bets),
        total_wins=total_wins,
        total_losses=total_losses,
        total_pushes=total_pushes,
        brier_score=brier_score(predictions),
        overall_hit_rate=overall_hit_rate(predictions),
        reliability=reliability_table(predictions),
    )


def format_calibration_report(report: CalibrationReport) -> str:
    """Formata o relatório em texto simples para console/log."""
    lines = [
        "=== Relatório de Calibração ===",
        f"Apostas resolvidas: {report.total_resolved} "
        f"(vitórias={report.total_wins}, derrotas={report.total_losses}, pushes={report.total_pushes})",
    ]

    if report.total_resolved == 0:
        lines.append("Sem apostas resolvidas ainda -- nada para calibrar.")
        return "\n".join(lines)

    lines.append(f"Taxa de acerto geral: {report.overall_hit_rate:.1%}" if report.overall_hit_rate is not None else "Taxa de acerto: N/D")
    lines.append(f"Brier Score: {report.brier_score:.4f} (0=perfeito, 0.25=equivalente a 'sempre 50%')" if report.brier_score is not None else "Brier Score: N/D")
    lines.append("")

    if not report.reliability:
        lines.append("Amostra insuficiente para tabela de confiabilidade por faixa.")
        return "\n".join(lines)

    lines.append(f"{'Faixa prevista':<14}{'Média prevista':>16}{'Taxa real':>12}{'N':>6}{'Overconf.':>12}")
    for bucket in report.reliability:
        lines.append(
            f"{bucket.bucket_label:<14}{bucket.predicted_probability_mean:>15.1%} "
            f"{bucket.actual_win_rate:>11.1%}{bucket.count:>6}{bucket.overconfidence:>+12.1%}"
        )

    return "\n".join(lines)
