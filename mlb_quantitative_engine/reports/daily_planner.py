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

A reescrita dos gatilhos no SO depende da plataforma: no Windows usa o módulo
ScheduledTasks via PowerShell; no Linux usa o crontab do usuário (`apply_linux_schedule`),
com a mesma semântica — um gatilho único por horário exato, já que o cron aceita dia+mês
específicos num campo (equivalente a `-Once -At`). A janela do verificador de resultados
(que no Windows usa RepetitionInterval/RepetitionDuration) vira uma linha de cron por
instante de 15 em 15 min dentro da janela — mesmo efeito prático. A parte de cálculo
(analytics/schedule_planning.py) é pura e testável; a parte de I/O (aplicar no SO) é
injetável (`apply_fn`) para permitir teste sem tocar no agendador real.

Uso: `python -m mlb_quantitative_engine.reports.daily_planner`
"""

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
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

# raiz do projeto (mlb_quantitative_engine/reports/daily_planner.py -> sobe 2 níveis)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "bin" / "python"
_LOG_DIR = _PROJECT_ROOT / "mlb_quantitative_engine" / "logs"

_CRON_BEGIN_INCREMENTAL = "# MLB_QUANTITATIVE_ENGINE BEGIN incremental_report"
_CRON_END_INCREMENTAL = "# MLB_QUANTITATIVE_ENGINE END incremental_report"
_CRON_BEGIN_RESULT = "# MLB_QUANTITATIVE_ENGINE BEGIN result_checker"
_CRON_END_RESULT = "# MLB_QUANTITATIVE_ENGINE END result_checker"

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


def _python_executable() -> str:
    return str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable


def _cron_command(module: str, log_name: str) -> str:
    log_path = _LOG_DIR / log_name
    return f"cd {_PROJECT_ROOT} && {_python_executable()} -m {module} >> {log_path} 2>&1"


def _cron_line(trigger: datetime, command: str) -> str:
    """Uma linha de cron que dispara só nesse instante: dia+mês específicos (em vez de
    `*`) fazem o campo funcionar como gatilho único, o equivalente cron de `-Once -At`
    do Windows Task Scheduler. `daily_planner` reescreve o bloco a cada manhã, então
    linhas de dias passados são naturalmente substituídas antes de poderem repetir no
    ano seguinte."""
    local = trigger.astimezone()
    return f"{local.minute} {local.hour} {local.day} {local.month} * {command}"


def _read_crontab() -> str:
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode != 0:
        return ""  # ainda não existe crontab para o usuário
    return result.stdout


def _write_crontab(content: str) -> None:
    result = subprocess.run(["crontab", "-"], input=content, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Falha ao atualizar crontab: {result.stderr.strip() or result.stdout.strip()}")


def _replace_block(crontab_text: str, begin_marker: str, end_marker: str, new_lines: List[str]) -> str:
    """Substitui o conteúdo entre `begin_marker`/`end_marker` por `new_lines`, preservando
    o resto do crontab do usuário intocado (mesmo papel do nome único de task no Windows)."""
    lines = crontab_text.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == begin_marker:
            i += 1
            while i < len(lines) and lines[i].strip() != end_marker:
                i += 1
            i += 1  # pula a linha do end_marker
            continue
        out.append(lines[i])
        i += 1
    while out and not out[-1].strip():
        out.pop()
    if new_lines:
        out.append(begin_marker)
        out.extend(new_lines)
        out.append(end_marker)
    return "\n".join(out) + "\n" if out else ""


def apply_linux_schedule(
    triggers: List[datetime], window: Optional[Tuple[datetime, datetime]]
) -> None:
    """Reescreve os gatilhos equivalentes no crontab do usuário (implementação real para
    Linux — mesmo papel de `apply_windows_schedule`, ver docstring do módulo)."""
    current = _read_crontab()

    incremental_lines = [
        _cron_line(
            trigger,
            _cron_command("mlb_quantitative_engine.reports.incremental_runner", "cron_incremental.log"),
        )
        for trigger in triggers
    ]
    current = _replace_block(current, _CRON_BEGIN_INCREMENTAL, _CRON_END_INCREMENTAL, incremental_lines)

    result_lines: List[str] = []
    if window is not None:
        start, end = window
        command = _cron_command("mlb_quantitative_engine.reports.result_checker_runner", "cron_result_checker.log")
        step = timedelta(minutes=RESULT_CHECK_INTERVAL_MINUTES)
        t = start
        while t <= end:
            result_lines.append(_cron_line(t, command))
            t += step
    current = _replace_block(current, _CRON_BEGIN_RESULT, _CRON_END_RESULT, result_lines)

    _write_crontab(current)
    log.info(
        f"crontab atualizado: {len(incremental_lines)} gatilho(s) de relatório, "
        f"{len(result_lines)} verificação(ões) de resultado"
    )


def _default_apply_fn() -> ApplyFn:
    return apply_windows_schedule if sys.platform.startswith("win") else apply_linux_schedule


def plan_today(
    date: Optional[str] = None,
    api_client: Optional[MLBApiClient] = None,
    now: Optional[datetime] = None,
    apply_fn: Optional[ApplyFn] = None,
) -> Tuple[List[datetime], Optional[Tuple[datetime, datetime]]]:
    """Calcula e aplica o agendamento do dia. Retorna (gatilhos, janela) para inspeção.

    `apply_fn` é injetável para testes (padrão: reescreve as tarefas reais do SO —
    Windows Task Scheduler ou crontab, conforme a plataforma)."""
    now = now or datetime.now(timezone.utc)
    target_date = date or now.strftime("%Y-%m-%d")
    client = api_client or MLBApiClient()
    apply = apply_fn or _default_apply_fn()

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
