from __future__ import annotations

"""Envia alertas de Value Bet para um canal/grupo do Telegram.

Mensagem pensada para ser limpa e direta: qual jogo, qual lado (Over/Under —
e o time, quando for Team Total), a odd mínima aceitável, a probabilidade
projetada pelo modelo e o tamanho de entrada sugerido (Kelly fracionado,
limitado a no máximo 2% da banca por aposta). Não inclui edge, EV bruto,
casa de apostas ou outros detalhes técnicos — isso fica registrado só no
banco (ver Repository); o alerta é para consumo rápido em um canal. A odd
mínima (não a melhor encontrada) e a ausência da casa de apostas são
propositais: a recomendação deve valer em qualquer casa que ofereça pelo
menos essa odd, não travada a uma única banca específica.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

from mlb_quantitative_engine.analytics.backtesting import BacktestResult
from mlb_quantitative_engine.api.telegram_api import TelegramApiClient
from mlb_quantitative_engine.config import settings
from mlb_quantitative_engine.models.value_bet import ValueBet, describe_market
from mlb_quantitative_engine.services.auto_tuning_service import AutoTuningRunResult
from mlb_quantitative_engine.services.calibration_report_service import CalibrationReport
from mlb_quantitative_engine.utils.logger import log

_BRASILIA_TZ = ZoneInfo("America/Sao_Paulo")


def format_game_datetime_brasilia(game_datetime: Optional[str]) -> Optional[str]:
    """Converte o horário do jogo (ISO em UTC, ex.: '2026-07-20T23:05:00Z') para o
    horário de Brasília, formatado como 'dd/mm/AAAA HH:MM'. Retorna None quando o
    horário não é conhecido ou não pode ser interpretado (a mensagem simplesmente
    omite a linha nesse caso)."""
    if not game_datetime:
        return None
    try:
        utc_dt = datetime.fromisoformat(game_datetime.replace("Z", "+00:00"))
    except ValueError:
        return None
    return utc_dt.astimezone(_BRASILIA_TZ).strftime("%d/%m/%Y %H:%M")


def format_value_bet_message(bet: ValueBet, game_datetime: Optional[str] = None) -> str:
    """Formata uma avaliação de Value Bet numa mensagem limpa para o Telegram (HTML).

    Quando o horário do jogo é conhecido, inclui a data/hora em horário de Brasília."""
    matchup = f"{bet.away_team} @ {bet.home_team}"
    side = describe_market(bet.market, bet.home_team, bet.away_team)

    lines = [f"⚾ <b>{matchup}</b>", ""]
    local_time = format_game_datetime_brasilia(game_datetime)
    if local_time:
        lines.append(f"🗓️ {local_time} (Brasília)")
    lines.extend(
        [
            f"🎯 {side} {bet.point}",
            f"🎲 Odd mínima: {bet.minimum_acceptable_price:.2f}",
            f"📊 Probabilidade: {bet.projected_probability * 100:.1f}%",
            f"💰 Entrada sugerida: {bet.suggested_stake_fraction * 100:.1f}% da banca",
        ]
    )
    return "\n".join(lines)


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


@dataclass(frozen=True)
class RecentChangeSummary:
    """Uma mudança de parâmetro aplicada nos últimos dias, com o efeito observado desde
    então (hit rate geral do período, não causal por parâmetro -- só um sinal de
    acompanhamento) -- monta a seção "mudanças recentes em observação" do relatório."""

    parameter_name: str
    old_value: float
    new_value: float
    days_ago: int
    bets_since: int
    hit_rate_since: Optional[float]


def _format_reliability_highlight(reliability) -> Optional[str]:
    if not reliability:
        return None
    worst = max(reliability, key=lambda bucket: abs(bucket.overconfidence))
    if abs(worst.overconfidence) < 0.05:
        return None
    return (
        f"faixa {worst.bucket_label} -- previsto {worst.predicted_probability_mean:.1%}, "
        f"real {worst.actual_win_rate:.1%} (n={worst.count}, overconf. {worst.overconfidence:+.1%})"
    )


def format_daily_analysis_message(
    date: str,
    yesterday_backtest: BacktestResult,
    calibration: CalibrationReport,
    auto_tuning: AutoTuningRunResult,
    recent_changes: Sequence[RecentChangeSummary] = (),
) -> str:
    """Formata o relatório diário de análise + auto-tuning (ver reports/daily_analysis_runner.py)."""
    lines = [f"📊 <b>Análise diária -- {date}</b>", "", "<b>Resultado de ontem</b>"]

    if yesterday_backtest.total_bets == 0:
        lines.append("Nenhuma aposta resolvida ontem.")
    else:
        lines.append(
            f"{yesterday_backtest.total_bets} apostas (✅{yesterday_backtest.wins} "
            f"❌{yesterday_backtest.losses} ⚪{yesterday_backtest.pushes}) | "
            f"Hit rate: {yesterday_backtest.hit_rate:.1%} | ROI: {yesterday_backtest.roi:+.1%}"
        )

    lines.extend(["", "<b>Calibração acumulada</b>"])
    if calibration.total_resolved == 0:
        lines.append("Sem histórico suficiente ainda.")
    else:
        lines.append(
            f"{calibration.total_resolved} apostas resolvidas no total | "
            f"Hit rate: {calibration.overall_hit_rate:.1%} | Brier: {calibration.brier_score:.4f}"
        )
        highlight = _format_reliability_highlight(calibration.reliability)
        if highlight:
            lines.append(f"⚠️ Pior faixa: {highlight}")

    lines.extend(["", "<b>Mudanças hoje</b>"])
    if auto_tuning.skipped_reason:
        lines.append(f"⏸️ Nada aplicado: {auto_tuning.skipped_reason}")
    elif not auto_tuning.changes:
        lines.append("Nenhuma mudança -- amostra insuficiente ou tudo dentro do esperado.")
    else:
        for change in auto_tuning.changes:
            adjustment = change.adjustment
            if change.applied:
                lines.append(
                    f"✅ <b>{adjustment.parameter_name}</b>: {adjustment.old_value} → {adjustment.new_value}\n"
                    f"   {adjustment.rationale}"
                )
            else:
                lines.append(
                    f"❌ <b>{adjustment.parameter_name}</b>: tentativa revertida ({change.test_failure_summary})"
                )

    if auto_tuning.deferred:
        lines.extend(["", "<b>Candidatas represadas</b> (cooldown ou teto do dia)"])
        for candidate in auto_tuning.deferred:
            lines.append(f"• {candidate.parameter_name}: {candidate.old_value} → {candidate.new_value}")

    if auto_tuning.negative_roi_findings:
        lines.extend(["", "<b>Achados</b> (sem ação automática)"])
        for finding in auto_tuning.negative_roi_findings:
            lines.append(f"• {finding}")

    if recent_changes:
        lines.extend(["", "<b>Mudanças recentes em observação</b>"])
        for recent in recent_changes:
            hit_rate_text = f"{recent.hit_rate_since:.1%}" if recent.hit_rate_since is not None else "N/D"
            lines.append(
                f"• {recent.parameter_name} ({recent.old_value}→{recent.new_value}, há {recent.days_ago}d): "
                f"{recent.bets_since} apostas desde então, hit rate {hit_rate_text}"
            )

    return "\n".join(lines)


class TelegramNotifier:
    """Envia alertas de Value Bet para um canal do Telegram."""

    def __init__(self, api_client: Optional[TelegramApiClient] = None, channel_id: Optional[str] = None) -> None:
        self.api_client = api_client or TelegramApiClient()
        # None (não informado) -> usa o padrão das settings; "" explícito fica vazio de propósito
        # (ex.: testes que verificam o guard de "nenhum canal configurado").
        self.channel_id = settings.telegram_channel_id if channel_id is None else channel_id

    def send_value_bet_alert(self, bet: ValueBet, game_datetime: Optional[str] = None) -> dict:
        """Formata e envia um alerta de Value Bet. Retorna a resposta bruta da API.

        `game_datetime` (ISO em UTC) é opcional; quando informado, a mensagem inclui a
        data/hora do jogo em horário de Brasília."""
        if not self.channel_id:
            raise ValueError("Nenhum canal configurado (TELEGRAM_CHANNEL_ID ausente)")

        message = format_value_bet_message(bet, game_datetime)
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

    def send_daily_analysis_report(
        self,
        date: str,
        yesterday_backtest: BacktestResult,
        calibration: CalibrationReport,
        auto_tuning: AutoTuningRunResult,
        recent_changes: Sequence[RecentChangeSummary] = (),
    ) -> dict:
        """Formata e envia o relatório diário de análise + auto-tuning (ver
        reports/daily_analysis_runner.py)."""
        if not self.channel_id:
            raise ValueError("Nenhum canal configurado (TELEGRAM_CHANNEL_ID ausente)")

        message = format_daily_analysis_message(date, yesterday_backtest, calibration, auto_tuning, recent_changes)
        log.info(f"Enviando relatório diário de análise ao Telegram ({date})")
        return self.api_client.send_message(self.channel_id, message)
