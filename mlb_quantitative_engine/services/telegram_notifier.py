from __future__ import annotations

"""Envia alertas de Value Bet para um canal/grupo do Telegram.

Mensagem pensada para ser limpa e direta: qual jogo, qual lado (Over/Under —
e o time, quando for Team Total), a odd mínima aceitável, a probabilidade
projetada pelo modelo e o tamanho de entrada sugerido (Kelly fracionado,
limitado a no máximo 2% da banca por aposta). Não inclui edge, EV bruto,
casa de apostas ou outros detalhes técnicos — isso fica no relatório
completo (xlsx/banco); o alerta é para consumo rápido em um canal. A odd
mínima (não a melhor encontrada) e a ausência da casa de apostas são
propositais: a recomendação deve valer em qualquer casa que ofereça pelo
menos essa odd, não travada a uma única banca específica.
"""

from typing import Optional

from mlb_quantitative_engine.api.telegram_api import TelegramApiClient
from mlb_quantitative_engine.config import settings
from mlb_quantitative_engine.models.value_bet import ValueBet, describe_market
from mlb_quantitative_engine.utils.logger import log


def format_value_bet_message(bet: ValueBet) -> str:
    """Formata uma avaliação de Value Bet numa mensagem limpa para o Telegram (HTML)."""
    matchup = f"{bet.away_team} @ {bet.home_team}"
    side = describe_market(bet.market, bet.home_team, bet.away_team)

    return (
        f"⚾ <b>{matchup}</b>\n\n"
        f"🎯 {side} {bet.point}\n"
        f"🎲 Odd mínima: {bet.minimum_acceptable_price:.2f}\n"
        f"📊 Probabilidade: {bet.projected_probability * 100:.1f}%\n"
        f"💰 Entrada sugerida: {bet.suggested_stake_fraction * 100:.1f}% da banca"
    )


_RESULT_ICON = {"GREEN": "✅", "RED": "❌", "PUSH": "⚪"}
_RESULT_LABEL = {"GREEN": "GREEN", "RED": "RED", "PUSH": "PUSH (anulada)"}


def format_bet_result_message(
    market: str,
    home_team: str,
    away_team: str,
    point: float,
    outcome_label: str,
    home_runs: int,
    away_runs: int,
) -> str:
    """Formata o resultado (GREEN/RED/PUSH) de uma sugestão já enviada, no mesmo estilo
    limpo do alerta original -- só o essencial: o lado apostado e o placar final."""
    matchup = f"{away_team} @ {home_team}"
    side = describe_market(market, home_team, away_team)

    return (
        f"{_RESULT_ICON[outcome_label]} <b>{_RESULT_LABEL[outcome_label]}</b>\n\n"
        f"⚾ {matchup}\n"
        f"🎯 {side} {point}\n"
        f"📌 Placar final: {away_team} {away_runs} x {home_runs} {home_team}"
    )


class TelegramNotifier:
    """Envia alertas de Value Bet para um canal do Telegram."""

    def __init__(self, api_client: Optional[TelegramApiClient] = None, channel_id: Optional[str] = None) -> None:
        self.api_client = api_client or TelegramApiClient()
        # None (não informado) -> usa o padrão das settings; "" explícito fica vazio de propósito
        # (ex.: testes que verificam o guard de "nenhum canal configurado").
        self.channel_id = settings.telegram_channel_id if channel_id is None else channel_id

    def send_value_bet_alert(self, bet: ValueBet) -> dict:
        """Formata e envia um alerta de Value Bet. Retorna a resposta bruta da API."""
        if not self.channel_id:
            raise ValueError("Nenhum canal configurado (TELEGRAM_CHANNEL_ID ausente)")

        message = format_value_bet_message(bet)
        log.info(f"Enviando alerta de Value Bet ao Telegram: {bet.away_team} @ {bet.home_team} ({bet.market})")
        return self.api_client.send_message(self.channel_id, message)

    def send_bet_result_alert(
        self,
        market: str,
        home_team: str,
        away_team: str,
        point: float,
        outcome_label: str,
        home_runs: int,
        away_runs: int,
    ) -> dict:
        """Formata e envia o resultado (GREEN/RED/PUSH) de uma sugestão já anunciada."""
        if not self.channel_id:
            raise ValueError("Nenhum canal configurado (TELEGRAM_CHANNEL_ID ausente)")

        message = format_bet_result_message(market, home_team, away_team, point, outcome_label, home_runs, away_runs)
        log.info(f"Enviando resultado ({outcome_label}) ao Telegram: {away_team} @ {home_team} ({market})")
        return self.api_client.send_message(self.channel_id, message)
