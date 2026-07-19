from __future__ import annotations

"""Verifica se as sugestões já enviadas ao Telegram deram GREEN, RED ou PUSH, e avisa
no canal assim que o jogo termina.

Raciocínio:
- Só vale a pena checar apostas que realmente chegaram ao canal (`alert_sent=True`,
  marcado em ReportGenerator._maybe_send_telegram_alert só após um envio bem-sucedido)
  e cujo resultado ainda não foi anunciado (`result_notified=False`). Uma vez enviado
  o aviso de resultado, `mark_bet_outcome` grava o resultado ("win"/"loss"/"push",
  usado pelo harness de calibração em analytics/calibration.py) e marca
  `result_notified=True`, tirando a aposta desta consulta para sempre — é assim que
  se evita reenviar a mesma mensagem.
- O placar (via boxscore, GameResultService) já existe durante o jogo, então NÃO
  basta ter runs disponíveis: é preciso confirmar que o jogo já terminou
  (status "Final"/"Game Over"/"Completed Early..." no schedule da MLB Stats API)
  antes de classificar over/under, senão o resultado seria prematuro/errado.
- Quando não há nenhuma aposta pendente (nenhum alerta enviado ainda hoje, ou tudo já
  resolvido), a função retorna sem fazer nenhuma chamada de API — rodar isso a cada
  poucos minutos fora do horário de jogos não desperdiça nada além de uma consulta
  gratuita ao schedule por data distinta entre as apostas pendentes.
"""

from typing import Dict, Optional, Set

from mlb_quantitative_engine.api.mlb_api import MLBApiClient
from mlb_quantitative_engine.database.repository import Repository
from mlb_quantitative_engine.services.game_result_service import GameResultService
from mlb_quantitative_engine.services.telegram_notifier import TelegramNotifier
from mlb_quantitative_engine.utils.logger import log

_FINAL_STATUS_PREFIXES = ("Final", "Game Over", "Completed Early")


def _is_final_status(status: Optional[str]) -> bool:
    if not status:
        return False
    return any(status.startswith(prefix) for prefix in _FINAL_STATUS_PREFIXES)


def _split_market(market: str) -> tuple:
    """"home_team_total_over" -> ("home_team_total", "over")."""
    side = market.rsplit("_", 1)[-1]
    prefix = market[: -(len(side) + 1)]
    return prefix, side


def check_pending_bet_results(
    api_client: Optional[MLBApiClient] = None,
    repository: Optional[Repository] = None,
    game_result_service: Optional[GameResultService] = None,
    telegram_notifier: Optional[TelegramNotifier] = None,
) -> int:
    """Para cada ValueBet já anunciado no Telegram mas sem resultado notificado, checa
    se o jogo terminou e, se sim, envia GREEN/RED/PUSH e marca como notificado.

    Retorna quantas apostas foram notificadas nesta chamada.
    """
    client = api_client or MLBApiClient()
    repo = repository or Repository()
    result_service = game_result_service or GameResultService(client)
    notifier = telegram_notifier or TelegramNotifier()

    pending = repo.list_bets_pending_result_check()
    if not pending:
        log.info("Nenhuma aposta pendente de verificação de resultado")
        return 0

    status_by_pk: Dict[int, str] = {}
    checked_dates: Set[str] = set()
    notified = 0

    for bet, game in pending:
        if game.game_date not in checked_dates:
            checked_dates.add(game.game_date)
            for summary in client.get_games_for_date(game.game_date):
                status_by_pk[summary.game_pk] = summary.status or ""

        if not _is_final_status(status_by_pk.get(game.game_pk)):
            continue

        result = result_service.get_final_score(game.game_pk)
        if result is None:
            log.warning(f"Jogo {game.game_pk} marcado como encerrado mas sem placar disponível ainda")
            continue

        market_prefix, side = _split_market(bet.market)
        if market_prefix == "game_total":
            actual_runs = result.total_runs
        elif market_prefix == "home_team_total":
            actual_runs = result.home_runs
        elif market_prefix == "away_team_total":
            actual_runs = result.away_runs
        else:
            log.warning(f"Mercado desconhecido para checagem de resultado: {bet.market}")
            continue

        outcome = GameResultService.classify_total(actual_runs, bet.point)
        if outcome == "push":
            label, bet_outcome = "PUSH", "push"
        elif outcome == side:
            label, bet_outcome = "GREEN", "win"
        else:
            label, bet_outcome = "RED", "loss"

        try:
            notifier.send_bet_result_alert(
                market=bet.market,
                home_team=game.home_team,
                away_team=game.away_team,
                point=bet.point,
                outcome_label=label,
                home_runs=result.home_runs,
                away_runs=result.away_runs,
            )
        except Exception as exc:  # noqa: BLE001 - falha de envio não deve travar o lote; tenta de novo no próximo run
            log.error(f"Falha ao enviar resultado ({label}) do jogo {game.game_pk}: {exc}")
            continue

        repo.mark_bet_outcome(bet.id, bet_outcome)
        notified += 1
        log.info(f"Resultado {label} notificado para {bet.market} do jogo {game.game_pk}")

    return notified
