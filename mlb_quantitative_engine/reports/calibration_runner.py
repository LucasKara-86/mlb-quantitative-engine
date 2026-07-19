from __future__ import annotations

"""Imprime o relatório de calibração atual (Brier Score + tabela de confiabilidade por
faixa de probabilidade) a partir das apostas já resolvidas no banco (GREEN/RED/PUSH
verificados por bet_result_checker.py).

Uso sob demanda, não faz parte do pipeline automático — rode quando quiser conferir se
as probabilidades do modelo estão batendo com a realidade:
`python -m mlb_quantitative_engine.reports.calibration_runner`
"""

from mlb_quantitative_engine.services.calibration_report_service import (
    build_calibration_report,
    format_calibration_report,
)

if __name__ == "__main__":
    report = build_calibration_report()
    print(format_calibration_report(report))
