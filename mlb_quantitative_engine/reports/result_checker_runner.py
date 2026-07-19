from __future__ import annotations

"""Ponto de entrada para a tarefa agendada que verifica, a cada poucos minutos, se as
sugestões já enviadas ao Telegram terminaram em GREEN, RED ou PUSH.

Pensado para ser chamado por uma tarefa agendada do Windows a cada 10 minutos,
apenas durante a janela em que jogos costumam estar acontecendo (mesma lógica de
"não gastar à toa" do incremental_runner: quando não há nenhuma aposta pendente
de verificação, a chamada não faz nenhum request além do necessário e retorna
imediatamente).

Uso: `python -m mlb_quantitative_engine.reports.result_checker_runner`
"""

from mlb_quantitative_engine.services.bet_result_checker import check_pending_bet_results

if __name__ == "__main__":
    notified = check_pending_bet_results()
    print(f"Resultados notificados nesta execução: {notified}")
