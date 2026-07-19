from __future__ import annotations

import pytest

from mlb_quantitative_engine.models.value_bet import ValueBet
from mlb_quantitative_engine.services.telegram_notifier import (
    TelegramNotifier,
    format_bet_result_message,
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
