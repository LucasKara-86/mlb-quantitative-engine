from __future__ import annotations

"""Ponto de entrada da tarefa diária de análise + auto-tuning: reúne o backtest de ontem,
a calibração acumulada, roda o ciclo completo de auto-tuning
(services/auto_tuning_service.py) e envia o relatório único ao Telegram
(services/telegram_notifier.format_daily_analysis_message).

Pensado para rodar uma vez por dia, de manhã, depois que os jogos de ontem já
terminaram e tiveram seus resultados verificados (ver
reports/result_checker_runner.py) -- roda antes de reports/daily_planner.py
para não competir por I/O no mesmo minuto.

O auto-tuning aplicado aqui pode commitar E dar `git push origin main`
sozinho (ver services/auto_tuning_service.py) quando uma proposta de ajuste
sobrevive ao gate de testes -- é o comportamento padrão (`push=True`) desta
tarefa em produção.

Uso: `python -m mlb_quantitative_engine.reports.daily_analysis_runner`
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from mlb_quantitative_engine.database.models import ValueBet
from mlb_quantitative_engine.database.repository import Repository
from mlb_quantitative_engine.services.auto_tuning_service import AutoTuningRunResult, run_auto_tuning
from mlb_quantitative_engine.services.backtest_report_service import BacktestReport, build_backtest_report
from mlb_quantitative_engine.services.calibration_report_service import build_calibration_report
from mlb_quantitative_engine.services.telegram_notifier import RecentChangeSummary, TelegramNotifier
from mlb_quantitative_engine.utils.logger import log

# Janela de acompanhamento das "mudanças recentes em observação" no relatório -- bem
# maior que o cooldown do auto-tuning (5 dias, ver COOLDOWN_DAYS em
# services/auto_tuning_service.py) para dar tempo de acumular amostra observável.
RECENT_CHANGES_WINDOW_DAYS = 14


def _yesterday_date(now: datetime) -> str:
    """Data LOCAL, não UTC (mesmo cuidado de incremental_runner.py/daily_planner.py):
    "ontem" é sempre relativo ao fuso de quem opera o sistema, não ao servidor em UTC."""
    return (now.astimezone() - timedelta(days=1)).strftime("%Y-%m-%d")


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _build_yesterday_backtest(repository: Repository, yesterday: str) -> BacktestReport:
    all_resolved = repository.list_resolved_value_bets_with_game()
    scoped = [(value_bet, game) for value_bet, game in all_resolved if game.game_date == yesterday]
    return build_backtest_report(resolved_rows=scoped)


def _recent_change_summaries(
    repository: Repository, now: datetime, window_days: int = RECENT_CHANGES_WINDOW_DAYS
) -> List[RecentChangeSummary]:
    """Mudanças de parâmetro APLICADAS (não revertidas) nos últimos `window_days` dias,
    com o hit rate das apostas avaliadas desde então (`ValueBet.created_at >=
    change.created_at`, já resolvidas) -- só um sinal de acompanhamento, não causal (ver
    docstring de RecentChangeSummary em services/telegram_notifier.py)."""
    since = now - timedelta(days=window_days)
    changes = [change for change in repository.list_recent_parameter_changes(since) if change.applied]
    if not changes:
        return []

    resolved_bets: List[ValueBet] = list(repository.list_resolved_value_bets())
    summaries: List[RecentChangeSummary] = []
    for change in changes:
        change_time = _as_utc(change.created_at)
        bets_since = [bet for bet in resolved_bets if _as_utc(bet.created_at) >= change_time]
        decided = [bet for bet in bets_since if bet.outcome in ("win", "loss")]
        wins = sum(1 for bet in decided if bet.outcome == "win")
        hit_rate = (wins / len(decided)) if decided else None
        summaries.append(
            RecentChangeSummary(
                parameter_name=change.parameter_name,
                old_value=change.old_value,
                new_value=change.new_value,
                days_ago=max((now - change_time).days, 0),
                bets_since=len(bets_since),
                hit_rate_since=hit_rate,
            )
        )
    return summaries


def run_daily_analysis(
    repository: Optional[Repository] = None,
    telegram_notifier: Optional[TelegramNotifier] = None,
    now: Optional[datetime] = None,
) -> AutoTuningRunResult:
    """Executa o ciclo diário completo (backtest de ontem + calibração acumulada +
    auto-tuning) e envia o relatório ao Telegram. Retorna o resultado do auto-tuning --
    o que services/auto_tuning_service.py decidiu e aplicou.

    O envio ao Telegram é best-effort: uma falha (canal não configurado, rede) é logada
    e engolida, nunca desfaz o auto-tuning já persistido (mesma convenção de
    reports/report_generator.py para alertas de Value Bet)."""
    repo = repository or Repository()
    now = now or datetime.now(timezone.utc)
    notifier = telegram_notifier or TelegramNotifier()

    yesterday = _yesterday_date(now)
    yesterday_backtest = _build_yesterday_backtest(repo, yesterday)
    calibration = build_calibration_report(repo)
    auto_tuning = run_auto_tuning(repository=repo, now=now)
    recent_changes = _recent_change_summaries(repo, now)

    try:
        notifier.send_daily_analysis_report(
            date=yesterday,
            yesterday_backtest=yesterday_backtest.overall,
            calibration=calibration,
            auto_tuning=auto_tuning,
            recent_changes=recent_changes,
        )
    except Exception as exc:  # noqa: BLE001 - notificação é best-effort, nunca deve mascarar o auto-tuning já feito
        log.error(f"daily_analysis_runner: falha ao enviar relatório diário ao Telegram: {exc}")

    return auto_tuning


if __name__ == "__main__":
    result = run_daily_analysis()
    applied = sum(1 for change in result.changes if change.applied)
    reverted = sum(1 for change in result.changes if not change.applied)
    print(f"Auto-tuning: {applied} mudança(s) aplicada(s), {reverted} revertida(s), {len(result.deferred)} represada(s)")
