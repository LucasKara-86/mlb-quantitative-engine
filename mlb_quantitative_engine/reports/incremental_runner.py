from __future__ import annotations

"""Processa apenas os lotes de jogos cujo horário de disparo já chegou (ver
analytics/batch_scheduling.py), evitando reprocessar o dia inteiro a cada
execução — só busca dados novos para os jogos que estão prestes a começar.

Cada jogo tem seu próprio gatilho (30 minutos antes do seu horário de início
— `DEFAULT_LEAD_TIME` em batch_scheduling.py), não mais um gatilho
compartilhado por vários jogos agrupados numa janela de 2 horas. Isso exige
que a tarefa agendada rode com cadência fina o bastante (ver Windows Task
Scheduler "MLB_Incremental_Report") para não perder o alvo de "30 minutos
antes" por muita margem quando os jogos começam em horários não múltiplos do
intervalo do cron (ex.: jogo às 13:15 -> gatilho às 12:45).

Pensado para ser chamado por uma tarefa agendada do Windows a cada poucos
minutos: cada chamada verifica o que está pendente (usando o registro de
lotes já processados no banco) e processa só isso. O banco de dados é a
única saída persistida — não há mais exportação para xlsx/csv; quem quiser
consultar o estado do relatório usa o banco diretamente (ver Repository).

Quando nenhum `report_generator` é injetado explicitamente, este módulo
constrói um `ReportGenerator` com um `TelegramNotifier` real — ou seja, por
padrão, o pipeline automático ENVIA alertas ao canal configurado
(TELEGRAM_CHANNEL_ID) para toda avaliação que atingir os critérios de Value
Bet. Para rodar sem enviar nada ao Telegram (ex.: testes manuais), injete um
`report_generator` próprio sem notifier.

Uso: `python -m mlb_quantitative_engine.reports.incremental_runner`
"""

from datetime import datetime, timezone
from typing import List, Optional

from mlb_quantitative_engine.analytics.batch_scheduling import Batch, GameSchedule, compute_batches, due_batches
from mlb_quantitative_engine.api.mlb_api import MLBApiClient
from mlb_quantitative_engine.database.repository import Repository
from mlb_quantitative_engine.reports.report_generator import ReportGenerator
from mlb_quantitative_engine.services.telegram_notifier import TelegramNotifier
from mlb_quantitative_engine.utils.logger import log


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def run_due_batches(
    date: Optional[str] = None,
    api_client: Optional[MLBApiClient] = None,
    repository: Optional[Repository] = None,
    report_generator: Optional[ReportGenerator] = None,
    now: Optional[datetime] = None,
) -> List[Batch]:
    """Processa todos os lotes pendentes para a data informada (hoje, por padrão).

    `now` é injetável para testes determinísticos; em produção usa o horário atual.
    Retorna a lista de lotes efetivamente processados nesta chamada (vazia se
    nenhum lote estava com o horário de disparo já alcançado).
    """
    target_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now = now or datetime.now(timezone.utc)
    client = api_client or MLBApiClient()
    repo = repository or Repository()
    generator = report_generator or ReportGenerator(
        api_client=client, repository=repo, telegram_notifier=TelegramNotifier()
    )

    games = client.get_games_for_date(target_date)
    schedule = [
        GameSchedule(game_pk=game.game_pk, start_time=_parse_datetime(game.game_datetime))
        for game in games
        if game.game_datetime
    ]
    batches = compute_batches(schedule)

    pending = due_batches(
        batches, now=now, is_processed=lambda anchor: repo.is_batch_processed(target_date, anchor)
    )

    if not pending:
        log.info(f"Nenhum lote pendente para {target_date} (verificado às {now.isoformat()})")
        return []

    games_by_pk = {game.game_pk: game for game in games}
    all_odds = generator.fetch_all_odds()

    for batch in pending:
        log.info(
            f"Processando lote ancorado em {batch.anchor_time.isoformat()} "
            f"({len(batch.games)} jogo(s), disparo em {batch.trigger_time.isoformat()})"
        )
        for game_schedule in batch.games:
            game = games_by_pk.get(game_schedule.game_pk)
            if game is None:
                continue
            generator.build_row(game, all_odds)
        repo.mark_batch_processed(target_date, batch.anchor_time)

    return pending


def retry_pending_lineups(
    repository: Optional[Repository] = None,
    report_generator: Optional[ReportGenerator] = None,
    now: Optional[datetime] = None,
) -> int:
    """Reprocessa jogos cuja lineup ainda não estava oficial numa passada anterior
    (ver `ReportGenerator.retry_game` e `PendingLineupRetry`), sem gastar créditos
    novos de odds — só a lineup (grátis) é reconsultada, reaproveitando as odds já
    persistidas. Retorna quantos jogos foram efetivamente reavaliados nesta chamada.
    """
    now = now or datetime.now(timezone.utc)
    repo = repository or Repository()
    client = MLBApiClient()
    generator = report_generator or ReportGenerator(
        api_client=client, repository=repo, telegram_notifier=TelegramNotifier()
    )

    due_retries = repo.list_due_lineup_retries(now)
    if not due_retries:
        log.info(f"Nenhuma retentativa de lineup pendente (verificado às {now.isoformat()})")
        return 0

    for retry in due_retries:
        log.info(f"Reavaliando lineup do jogo {retry.game_pk} (retentativa agendada para {retry.retry_at.isoformat()})")
        generator.retry_game(retry.game_pk, now=now)

    return len(due_retries)


if __name__ == "__main__":
    processed = run_due_batches()
    print(f"Lotes processados nesta execução: {len(processed)}")
    retried = retry_pending_lineups()
    print(f"Retentativas de lineup reavaliadas nesta execução: {retried}")
