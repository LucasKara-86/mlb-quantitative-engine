from __future__ import annotations

"""Lógica pura de planejamento diário de agendamento: a partir dos horários de início
dos jogos do dia, calcula QUANDO as tarefas de trabalho precisam rodar.

Substitui o polling cego de intervalo fixo (a cada 10 min das 11:00 à meia-noite, roubo
de horário arbitrário) por gatilhos calculados a partir do schedule real — que muda a
cada dia. A camada que efetivamente reescreve as tarefas do Windows fica em
reports/daily_planner.py; este módulo é só matemática de horários (sem I/O), no mesmo
espírito de analytics/batch_scheduling.py.

Dois produtos:
- Gatilhos do relatório incremental: para cada jogo, um gatilho em (início - offset)
  para cada offset em `offsets`. Por padrão dois offsets — 30 min (envio principal,
  alinhado ao lead_time de batch_scheduling) e 15 min (segunda passada, para pegar
  escalações que só saíram mais perto da hora, via retentativa de lineup). Gatilhos que
  já passaram (relativo a `now`) são descartados; o Windows rejeita gatilho no passado.
- Janela do verificador de resultados: [primeiro início, último início + `tail`]. Ao
  contrário do relatório, checar resultado PRECISA de polling — o fim de um jogo é
  imprevisível (innings extras, atrasos) —, então aqui devolvemos só a JANELA em que
  vale a pena ficar verificando, não gatilhos exatos.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Tuple

DEFAULT_INCREMENTAL_OFFSETS: Tuple[timedelta, ...] = (timedelta(minutes=30), timedelta(minutes=15))
DEFAULT_RESULT_TAIL: timedelta = timedelta(hours=4)


def compute_incremental_trigger_times(
    game_start_times: Sequence[datetime],
    now: datetime,
    offsets: Sequence[timedelta] = DEFAULT_INCREMENTAL_OFFSETS,
) -> List[datetime]:
    """Gatilhos (únicos, ordenados, todos >= now) do relatório incremental para hoje.

    Para cada horário de início e cada offset, gera um gatilho em (início - offset).
    Jogos próximos no tempo naturalmente compartilham/deduplicam gatilhos coincidentes.
    """
    triggers = set()
    for start in game_start_times:
        for offset in offsets:
            trigger = start - offset
            if trigger >= now:
                triggers.add(trigger)
    return sorted(triggers)


def compute_result_check_window(
    game_start_times: Sequence[datetime],
    now: datetime,
    tail: timedelta = DEFAULT_RESULT_TAIL,
) -> Optional[Tuple[datetime, datetime]]:
    """Janela [início, fim] em que vale a pena verificar resultados hoje, ou None quando
    não há jogos ou a janela inteira já passou.

    - início = primeiro jogo (nunca antes de `now` — não adianta olhar o passado).
    - fim = último jogo + `tail` (folga para innings extras/atrasos; o fim real de um
      jogo é desconhecido de antemão).
    """
    if not game_start_times:
        return None
    end = max(game_start_times) + tail
    if end <= now:
        return None
    start = max(min(game_start_times), now)
    return start, end
