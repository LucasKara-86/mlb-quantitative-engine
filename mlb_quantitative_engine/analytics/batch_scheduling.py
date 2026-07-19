from __future__ import annotations

"""Agrupa os jogos do dia em lotes para atualização incremental do relatório.

Raciocínio (regra atual, redefinida pelo usuário depois de testar o agrupamento
por janela de 2 horas): cada jogo deve receber seu PRÓPRIO gatilho, `lead_time`
(30 minutos, por padrão) antes do SEU horário de início — não mais um gatilho
único compartilhado por vários jogos agrupados numa janela de 2 horas. A
mensagem no Telegram precisa estar no canal 30 minutos antes de CADA partida,
não 30 minutos antes só do primeiro jogo de um grupo.

Isso é obtido reaproveitando o mesmo algoritmo de agrupamento por âncora, só
que com `window=timedelta(0)` como padrão: um jogo só entra no lote de outro
se começar EXATAMENTE no mesmo horário (ex.: doubleheader com os dois jogos
marcados para o mesmo instante) — na prática, cada jogo vira seu próprio lote
de tamanho 1, disparado `lead_time` antes do seu próprio horário. O parâmetro
`window` continua existindo e funcional (útil para agrupar jogos próximos no
tempo caso essa política volte a fazer sentido no futuro), só não é mais o
comportamento padrão do app.

Este módulo é puramente funcional (sem I/O) — trabalha só com horários já
conhecidos. A camada de agendamento (cron) e busca de dados fica em outra
camada, que usa `compute_batches` e `due_batches` para decidir o que fazer.
Como `compute_batches` é chamado do zero a cada execução (busca o schedule do
dia via API e recalcula), não existe uma etapa separada de "montar o
cronograma no início do dia" — cada execução já está sempre olhando o
schedule atual.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, List, Sequence

DEFAULT_WINDOW = timedelta(0)
DEFAULT_LEAD_TIME = timedelta(minutes=30)


@dataclass(frozen=True)
class GameSchedule:
    """Um jogo e seu horário de início (timezone-aware, idealmente UTC)."""

    game_pk: int
    start_time: datetime


@dataclass(frozen=True)
class Batch:
    """Um lote de jogos a serem processados juntos, disparado `lead_time` antes da âncora."""

    anchor_time: datetime
    trigger_time: datetime
    games: List[GameSchedule] = field(default_factory=list)

    @property
    def game_pks(self) -> List[int]:
        return [g.game_pk for g in self.games]


def compute_batches(
    games: Sequence[GameSchedule],
    window: timedelta = DEFAULT_WINDOW,
    lead_time: timedelta = DEFAULT_LEAD_TIME,
) -> List[Batch]:
    """Agrupa os jogos em lotes, conforme a regra descrita no docstring do módulo."""
    sorted_games = sorted(games, key=lambda g: g.start_time)

    batches: List[Batch] = []
    current_anchor: datetime | None = None
    current_games: List[GameSchedule] = []

    for game in sorted_games:
        if current_anchor is None:
            current_anchor = game.start_time
            current_games = [game]
        elif game.start_time <= current_anchor + window:
            current_games.append(game)
        else:
            batches.append(
                Batch(anchor_time=current_anchor, trigger_time=current_anchor - lead_time, games=current_games)
            )
            current_anchor = game.start_time
            current_games = [game]

    if current_games:
        batches.append(
            Batch(anchor_time=current_anchor, trigger_time=current_anchor - lead_time, games=current_games)
        )

    return batches


def due_batches(
    batches: Sequence[Batch],
    now: datetime,
    is_processed: Callable[[datetime], bool],
) -> List[Batch]:
    """Retorna os lotes cujo horário de disparo já passou e que ainda não foram processados.

    `is_processed` recebe o `anchor_time` do lote e deve dizer se ele já foi
    processado (permite rodar o cron com uma cadência mais grosseira que o
    ideal sem reprocessar o mesmo lote nem pular um lote perdido).
    """
    return [batch for batch in batches if batch.trigger_time <= now and not is_processed(batch.anchor_time)]
