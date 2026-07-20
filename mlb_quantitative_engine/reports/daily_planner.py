from __future__ import annotations

"""Planejador diário: roda UMA vez de manhã, busca o schedule do dia e reescreve os
gatilhos das tarefas agendadas do Windows para que elas disparem só quando há algo a
fazer — em vez do polling cego a cada 10 min o dia inteiro.

Motivação: o horário dos jogos muda a cada dia. Ficar acordando o processo a cada 10 min
das 11:00 à meia-noite enquanto o primeiro jogo é só às 19:40 é inútil. Como o sistema já
conhece o schedule, ele mesmo calcula (ver analytics/schedule_planning.py) os horários
exatos e configura o Windows Task Scheduler de acordo.

O que este módulo faz a cada execução:
- MLB_Incremental_Report -> gatilhos ÚNICOS em (início do jogo - 30 min) e (- 15 min),
  para cada jogo. O de -30 é o envio principal; o de -15 pega escalações que saíram
  tarde (via retentativa de lineup). Fora desses instantes, a tarefa não roda.
- MLB_Result_Checker -> um gatilho repetindo a cada 15 min, mas só na janela em que há
  jogo acontecendo ([primeiro jogo, último jogo + 4h]). Verificar resultado precisa de
  polling (o fim do jogo é imprevisível), mas não o dia inteiro.
- Sem jogos hoje (ou janela já passou): desabilita a tarefa correspondente até o próximo
  planejamento.

Robustez:
- O catch-up (PC dormindo no horário exato do gatilho) é preservado pela configuração
  StartWhenAvailable das tarefas (feita na instalação), não por polling.
- Se este planner falhar (rede/SSL), as tarefas mantêm os gatilhos do último
  planejamento bem-sucedido — nada para silenciosamente.

Específico do Windows: a reescrita dos gatilhos usa o módulo ScheduledTasks via
PowerShell. A parte de cálculo (analytics/schedule_planning.py) é pura e testável; a
parte de I/O (aplicar no SO) é injetável (`apply_fn`) para permitir teste sem tocar no
agendador real.

Uso: `python -m mlb_quantitative_engine.reports.daily_planner`
"""

import subprocess
from datetime import datetime, timezone
from typing import Callable, List, Optional, Sequence, Tuple

from mlb_quantitative_engine.analytics.schedule_planning import (
    compute_incremental_trigger_times,
    compute_result_check_window,
)
from mlb_quantitative_engine.api.mlb_api import MLBApiClient
from mlb_quantitative_engine.utils.logger import log

INCREMENTAL_TASK_NAME = "MLB_Incremental_Report"
RESULT_TASK_NAME = "MLB_Result_Checker"
RESULT_CHECK_INTERVAL_MINUTES = 15

ApplyFn = Callable[[List[datetime], Optional[Tuple[datetime, datetime]]], None]


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _to_local_string(dt: datetime) -> str:
    """Converte um datetime UTC para o horário LOCAL da máquina (o Task Scheduler
    interpreta -At no fuso local) no formato que o PowerShell entende."""
    return dt.astimezone().strftime("%Y-%m-%dT%H:%M:%S")


def _run_powershell(script: str) -> None:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Falha ao atualizar tarefa agendada: {result.stderr.strip() or result.stdout.strip()}")


def apply_windows_schedule(
    triggers: List[datetime], window: Optional[Tuple[datetime, datetime]]
) -> None:
    """Reescreve os gatilhos das tarefas no Windows Task Scheduler (implementação real)."""
    _apply_incremental_triggers(triggers)
    _apply_result_window(window)


def _apply_incremental_triggers(triggers: List[datetime]) -> None:
    if not triggers:
        log.info(f"Sem gatilhos para {INCREMENTAL_TASK_NAME} hoje; desabilitando a tarefa")
        _run_powershell(f'Disable-ScheduledTask -TaskName "{INCREMENTAL_TASK_NAME}" | Out-Null')
        return

    lines = ["$triggers = @()"]
    for trigger in triggers:
        lines.append(f'$triggers += New-ScheduledTaskTrigger -Once -At "{_to_local_string(trigger)}"')
    lines.append(f'Set-ScheduledTask -TaskName "{INCREMENTAL_TASK_NAME}" -Trigger $triggers | Out-Null')
    lines.append(f'Enable-ScheduledTask -TaskName "{INCREMENTAL_TASK_NAME}" | Out-Null')
    _run_powershell("\n".join(lines))
    log.info(f"{INCREMENTAL_TASK_NAME}: {len(triggers)} gatilho(s) configurado(s)")


def _apply_result_window(window: Optional[Tuple[datetime, datetime]]) -> None:
    if window is None:
        log.info(f"Sem janela de jogos hoje; desabilitando {RESULT_TASK_NAME}")
        _run_powershell(f'Disable-ScheduledTask -TaskName "{RESULT_TASK_NAME}" | Out-Null')
        return

    start, end = window
    duration_minutes = max(int((end - start).total_seconds() // 60), RESULT_CHECK_INTERVAL_MINUTES)
    script = "\n".join(
        [
            f'$trigger = New-ScheduledTaskTrigger -Once -At "{_to_local_string(start)}" '
            f"-RepetitionInterval (New-TimeSpan -Minutes {RESULT_CHECK_INTERVAL_MINUTES}) "
            f"-RepetitionDuration (New-TimeSpan -Minutes {duration_minutes})",
            f'Set-ScheduledTask -TaskName "{RESULT_TASK_NAME}" -Trigger $trigger | Out-Null',
            f'Enable-ScheduledTask -TaskName "{RESULT_TASK_NAME}" | Out-Null',
        ]
    )
    _run_powershell(script)
    log.info(
        f"{RESULT_TASK_NAME}: verificando a cada {RESULT_CHECK_INTERVAL_MINUTES} min "
        f"de {start.isoformat()} até {end.isoformat()}"
    )


def plan_today(
    date: Optional[str] = None,
    api_client: Optional[MLBApiClient] = None,
    now: Optional[datetime] = None,
    apply_fn: Optional[ApplyFn] = None,
) -> Tuple[List[datetime], Optional[Tuple[datetime, datetime]]]:
    """Calcula e aplica o agendamento do dia. Retorna (gatilhos, janela) para inspeção.

    `apply_fn` é injetável para testes (padrão: reescreve as tarefas reais do Windows)."""
    now = now or datetime.now(timezone.utc)
    target_date = date or now.strftime("%Y-%m-%d")
    client = api_client or MLBApiClient()
    apply = apply_fn or apply_windows_schedule

    games = client.get_games_for_date(target_date)
    start_times: Sequence[datetime] = [
        _parse_datetime(game.game_datetime) for game in games if game.game_datetime
    ]

    triggers = compute_incremental_trigger_times(start_times, now)
    window = compute_result_check_window(start_times, now)

    log.info(
        f"Planejamento {target_date}: {len(start_times)} jogo(s) -> "
        f"{len(triggers)} gatilho(s) de relatório, janela de resultados={window}"
    )
    apply(triggers, window)
    return triggers, window


if __name__ == "__main__":
    computed_triggers, computed_window = plan_today()
    print(f"Gatilhos do relatório hoje: {len(computed_triggers)}")
    print(f"Janela de verificação de resultados: {computed_window}")
