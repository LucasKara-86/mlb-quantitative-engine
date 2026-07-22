from __future__ import annotations

import pytest

from mlb_quantitative_engine.analytics.auto_tuning import ParameterAdjustment
from mlb_quantitative_engine.analytics.backtesting import BacktestResult
from mlb_quantitative_engine.analytics.calibration import ReliabilityBucket
from mlb_quantitative_engine.models.value_bet import ValueBet
from mlb_quantitative_engine.services.auto_tuning_service import AppliedChange, AutoTuningRunResult
from mlb_quantitative_engine.services.calibration_report_service import CalibrationReport
from mlb_quantitative_engine.services.telegram_notifier import (
    RecentChangeSummary,
    TelegramNotifier,
    format_bet_result_message,
    format_daily_analysis_message,
    format_game_datetime_brasilia,
    format_value_bet_message,
)


def _bet(**overrides) -> ValueBet:
    base = dict(
        game_pk=1,
        home_team="Toronto Blue Jays",
        away_team="Chicago White Sox",
        market="away_team_total_over",
        bookmaker="FanDuel",
        price=1.90,
        point=4.5,
        projected_probability=0.623,
        implied_probability_raw=0.526,
        implied_probability_fair=0.50,
        edge=0.123,
        expected_value=0.184,
        kelly_fraction=0.34,
        kelly_fraction_quarter=0.085,
        suggested_stake_fraction=0.02,  # já limitado a 2%, mesmo com kelly_fraction_quarter maior
        minimum_acceptable_price=1.71,  # 1.90 * 0.90
        confidence_score=90.0,
        meets_criteria=True,
    )
    base.update(overrides)
    return ValueBet(**base)


class _FakeTelegramApiClient:
    def __init__(self) -> None:
        self.sent = []

    def send_message(self, chat_id: str, text: str, parse_mode: str = "HTML") -> dict:
        self.sent.append((chat_id, text, parse_mode))
        return {"ok": True, "result": {"message_id": 1}}


def test_format_value_bet_message_team_total() -> None:
    message = format_value_bet_message(_bet())

    assert "Chicago White Sox @ Toronto Blue Jays" in message
    assert "Chicago White Sox Over 4.5" in message
    assert "1.71" in message
    assert "62.3%" in message
    assert "2.0%" in message  # suggested_stake_fraction, não o kelly_fraction_quarter bruto (8.5%)


def test_format_value_bet_message_never_mentions_a_bookmaker() -> None:
    message = format_value_bet_message(_bet())
    assert "FanDuel" not in message


def test_format_game_datetime_brasilia_converts_utc_to_utc_minus_3() -> None:
    # 23:05 UTC -> 20:05 em Brasília (UTC-3)
    assert format_game_datetime_brasilia("2026-07-20T23:05:00Z") == "20/07/2026 20:05"


def test_format_game_datetime_brasilia_handles_day_rollover_backwards() -> None:
    # 02:10 UTC do dia 21 -> 23:10 do dia 20 em Brasília
    assert format_game_datetime_brasilia("2026-07-21T02:10:00Z") == "20/07/2026 23:10"


def test_format_game_datetime_brasilia_returns_none_for_missing_or_invalid() -> None:
    assert format_game_datetime_brasilia(None) is None
    assert format_game_datetime_brasilia("") is None
    assert format_game_datetime_brasilia("nao-e-data") is None


def test_format_value_bet_message_includes_game_time_in_brasilia_when_provided() -> None:
    message = format_value_bet_message(_bet(), game_datetime="2026-07-20T23:05:00Z")
    assert "20/07/2026 20:05" in message
    assert "Brasília" in message


def test_format_value_bet_message_omits_time_line_when_datetime_absent() -> None:
    message = format_value_bet_message(_bet())
    assert "Brasília" not in message


def test_format_value_bet_message_game_total() -> None:
    bet = _bet(
        market="game_total_under", point=8.5, projected_probability=0.55,
        suggested_stake_fraction=0.015, minimum_acceptable_price=1.80,
    )
    message = format_value_bet_message(bet)

    assert "Jogo Under 8.5" in message
    assert "1.80" in message
    assert "55.0%" in message
    assert "1.5%" in message


def test_notifier_sends_formatted_message_to_configured_channel() -> None:
    client = _FakeTelegramApiClient()
    notifier = TelegramNotifier(api_client=client, channel_id="@voobarato")

    result = notifier.send_value_bet_alert(_bet())

    assert result["ok"] is True
    assert len(client.sent) == 1
    chat_id, text, parse_mode = client.sent[0]
    assert chat_id == "@voobarato"
    assert "Chicago White Sox Over 4.5" in text
    assert parse_mode == "HTML"


def test_notifier_includes_game_time_when_datetime_passed() -> None:
    client = _FakeTelegramApiClient()
    notifier = TelegramNotifier(api_client=client, channel_id="@voobarato")

    notifier.send_value_bet_alert(_bet(), game_datetime="2026-07-20T23:05:00Z")

    _, text, _ = client.sent[0]
    assert "20/07/2026 20:05 (Brasília)" in text


def test_notifier_raises_when_no_channel_configured() -> None:
    client = _FakeTelegramApiClient()
    notifier = TelegramNotifier(api_client=client, channel_id="")

    with pytest.raises(ValueError):
        notifier.send_value_bet_alert(_bet())


def test_format_bet_result_message_green() -> None:
    message = format_bet_result_message(
        market="away_team_total_over", home_team="Toronto Blue Jays", away_team="Chicago White Sox",
        point=4.5, outcome_label="GREEN", home_runs=3, away_runs=6,
    )

    assert "GREEN" in message
    assert "✅" in message
    assert "Chicago White Sox Over 4.5" in message
    assert "Chicago White Sox 6 x 3 Toronto Blue Jays" in message


def test_format_bet_result_message_red() -> None:
    message = format_bet_result_message(
        market="game_total_over", home_team="Toronto Blue Jays", away_team="Chicago White Sox",
        point=8.5, outcome_label="RED", home_runs=2, away_runs=3,
    )

    assert "RED" in message
    assert "❌" in message
    assert "Jogo Over 8.5" in message


def test_format_bet_result_message_push() -> None:
    message = format_bet_result_message(
        market="game_total_under", home_team="Toronto Blue Jays", away_team="Chicago White Sox",
        point=9.0, outcome_label="PUSH", home_runs=5, away_runs=4,
    )

    assert "PUSH" in message


def test_notifier_sends_result_alert_to_configured_channel() -> None:
    client = _FakeTelegramApiClient()
    notifier = TelegramNotifier(api_client=client, channel_id="@voobarato")

    result = notifier.send_bet_result_alert(
        market="away_team_total_over", home_team="Toronto Blue Jays", away_team="Chicago White Sox",
        point=4.5, outcome_label="GREEN", home_runs=3, away_runs=6,
    )

    assert result["ok"] is True
    assert len(client.sent) == 1
    chat_id, text, parse_mode = client.sent[0]
    assert chat_id == "@voobarato"
    assert "GREEN" in text


def test_notifier_raises_when_no_channel_configured_for_result_alert() -> None:
    client = _FakeTelegramApiClient()
    notifier = TelegramNotifier(api_client=client, channel_id="")

    with pytest.raises(ValueError):
        notifier.send_bet_result_alert(
            market="game_total_over", home_team="A", away_team="B",
            point=8.5, outcome_label="RED", home_runs=1, away_runs=1,
        )


def _backtest(total_bets: int = 0, wins: int = 0, losses: int = 0, pushes: int = 0, roi: float = 0.0) -> BacktestResult:
    return BacktestResult(
        total_bets=total_bets, wins=wins, losses=losses, pushes=pushes,
        hit_rate=wins / (wins + losses) if (wins + losses) else 0.0,
        total_staked=total_bets * 0.02, total_profit=roi * total_bets * 0.02, roi=roi, yield_pct=roi,
        profit_factor=None, max_drawdown=0.0, sharpe_ratio=None, average_clv=None, equity_curve=[],
    )


def _calibration(total_resolved: int = 0, hit_rate: float = 0.0, brier: float = 0.2, reliability=()) -> CalibrationReport:
    return CalibrationReport(
        total_resolved=total_resolved, total_wins=0, total_losses=0, total_pushes=0,
        brier_score=brier if total_resolved else None,
        overall_hit_rate=hit_rate if total_resolved else None,
        reliability=list(reliability),
    )


def _empty_auto_tuning(skipped_reason=None) -> AutoTuningRunResult:
    return AutoTuningRunResult(
        changes=[], deferred=[], negative_roi_findings=[],
        calibration=_calibration(), backtest=_backtest(), skipped_reason=skipped_reason,
    )


def test_format_daily_analysis_message_reports_no_bets_yesterday() -> None:
    message = format_daily_analysis_message("2026-07-21", _backtest(), _calibration(), _empty_auto_tuning())
    assert "Nenhuma aposta resolvida ontem" in message


def test_format_daily_analysis_message_reports_yesterday_results() -> None:
    message = format_daily_analysis_message(
        "2026-07-21", _backtest(total_bets=7, wins=3, losses=4, roi=-0.15), _calibration(), _empty_auto_tuning(),
    )
    assert "7 apostas" in message
    assert "✅3" in message and "❌4" in message
    assert "-15.0%" in message


def test_format_daily_analysis_message_reports_no_changes_by_default() -> None:
    message = format_daily_analysis_message("2026-07-21", _backtest(), _calibration(), _empty_auto_tuning())
    assert "Nenhuma mudança" in message


def test_format_daily_analysis_message_reports_skipped_reason() -> None:
    message = format_daily_analysis_message(
        "2026-07-21", _backtest(), _calibration(), _empty_auto_tuning(skipped_reason="árvore git suja"),
    )
    assert "árvore git suja" in message


def test_format_daily_analysis_message_reports_applied_change() -> None:
    adjustment = ParameterAdjustment("min_edge", 0.04, 0.045, "ROI geral negativo", 25)
    auto_tuning = AutoTuningRunResult(
        changes=[AppliedChange(adjustment=adjustment, applied=True, git_commit_sha="deadbeef")],
        deferred=[], negative_roi_findings=[], calibration=_calibration(), backtest=_backtest(),
    )
    message = format_daily_analysis_message("2026-07-21", _backtest(), _calibration(), auto_tuning)

    assert "min_edge" in message
    assert "0.04 → 0.045" in message
    assert "ROI geral negativo" in message
    assert "✅" in message


def test_format_daily_analysis_message_reports_reverted_change() -> None:
    adjustment = ParameterAdjustment("min_edge", 0.04, 0.045, "ROI geral negativo", 25)
    auto_tuning = AutoTuningRunResult(
        changes=[AppliedChange(
            adjustment=adjustment, applied=False, git_commit_sha=None,
            test_failure_summary="suíte de testes falhou",
        )],
        deferred=[], negative_roi_findings=[], calibration=_calibration(), backtest=_backtest(),
    )
    message = format_daily_analysis_message("2026-07-21", _backtest(), _calibration(), auto_tuning)

    assert "❌" in message
    assert "revertida" in message
    assert "suíte de testes falhou" in message


def test_format_daily_analysis_message_reports_deferred_candidates() -> None:
    adjustment = ParameterAdjustment("overdispersion", 1.4, 1.45, "overconfidence alto", 30)
    auto_tuning = AutoTuningRunResult(
        changes=[], deferred=[adjustment], negative_roi_findings=[],
        calibration=_calibration(), backtest=_backtest(),
    )
    message = format_daily_analysis_message("2026-07-21", _backtest(), _calibration(), auto_tuning)

    assert "Candidatas represadas" in message
    assert "overdispersion" in message


def test_format_daily_analysis_message_reports_negative_roi_findings() -> None:
    auto_tuning = AutoTuningRunResult(
        changes=[], deferred=[], negative_roi_findings=["home_team_total: ROI -12.0% em 25 apostas"],
        calibration=_calibration(), backtest=_backtest(),
    )
    message = format_daily_analysis_message("2026-07-21", _backtest(), _calibration(), auto_tuning)

    assert "home_team_total" in message
    assert "Achados" in message


def test_format_daily_analysis_message_reports_calibration_snapshot() -> None:
    calibration = _calibration(total_resolved=130, hit_rate=0.55, brier=0.21)
    message = format_daily_analysis_message("2026-07-21", _backtest(), calibration, _empty_auto_tuning())

    assert "130 apostas resolvidas" in message
    assert "55.0%" in message
    assert "0.2100" in message


def test_format_daily_analysis_message_highlights_worst_reliability_bucket() -> None:
    reliability = [
        ReliabilityBucket("60%-65%", 0.62, 0.60, 25),
        ReliabilityBucket("75%-80%", 0.77, 0.40, 30),  # overconfidence bem maior
    ]
    calibration = _calibration(total_resolved=55, hit_rate=0.5, brier=0.22, reliability=reliability)
    message = format_daily_analysis_message("2026-07-21", _backtest(), calibration, _empty_auto_tuning())

    assert "75%-80%" in message


def test_format_daily_analysis_message_reports_recent_changes_in_observation() -> None:
    recent = [
        RecentChangeSummary(
            parameter_name="min_edge", old_value=0.04, new_value=0.045,
            days_ago=3, bets_since=12, hit_rate_since=0.58,
        )
    ]
    message = format_daily_analysis_message(
        "2026-07-21", _backtest(), _calibration(), _empty_auto_tuning(), recent_changes=recent,
    )

    assert "Mudanças recentes em observação" in message
    assert "min_edge" in message
    assert "58.0%" in message


def test_notifier_sends_daily_analysis_report_to_configured_channel() -> None:
    client = _FakeTelegramApiClient()
    notifier = TelegramNotifier(api_client=client, channel_id="@voobarato")

    result = notifier.send_daily_analysis_report("2026-07-21", _backtest(), _calibration(), _empty_auto_tuning())

    assert result["ok"] is True
    chat_id, text, parse_mode = client.sent[0]
    assert chat_id == "@voobarato"
    assert "Análise diária" in text


def test_notifier_raises_when_no_channel_configured_for_daily_analysis_report() -> None:
    client = _FakeTelegramApiClient()
    notifier = TelegramNotifier(api_client=client, channel_id="")

    with pytest.raises(ValueError):
        notifier.send_daily_analysis_report("2026-07-21", _backtest(), _calibration(), _empty_auto_tuning())
