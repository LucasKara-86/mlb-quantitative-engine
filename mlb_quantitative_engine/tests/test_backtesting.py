from __future__ import annotations

import statistics

import pytest

from mlb_quantitative_engine.analytics.backtesting import (
    BacktestResult,
    SettledBet,
    calculate_clv,
    compute_backtest_metrics,
    settle_bet_profit,
)


def _bet(price: float, stake: float, outcome: str, closing_price=None, game_pk: int = 1) -> SettledBet:
    return SettledBet(
        date="2026-07-17", game_pk=game_pk, market="game_total_over",
        price=price, stake=stake, outcome=outcome, closing_price=closing_price,
    )


def test_settle_bet_profit_win() -> None:
    assert settle_bet_profit(_bet(price=2.0, stake=1.0, outcome="win")) == pytest.approx(1.0)


def test_settle_bet_profit_loss() -> None:
    assert settle_bet_profit(_bet(price=2.0, stake=1.0, outcome="loss")) == pytest.approx(-1.0)


def test_settle_bet_profit_push() -> None:
    assert settle_bet_profit(_bet(price=2.0, stake=1.0, outcome="push")) == 0.0


def test_calculate_clv_positive_when_better_than_closing() -> None:
    assert calculate_clv(price_taken=2.0, closing_price=1.8) == pytest.approx((2.0 - 1.8) / 1.8)


def test_calculate_clv_negative_when_worse_than_closing() -> None:
    assert calculate_clv(price_taken=1.8, closing_price=2.0) == pytest.approx((1.8 - 2.0) / 2.0)


def test_compute_backtest_metrics_empty_bets_returns_degenerate_result() -> None:
    result = compute_backtest_metrics([], initial_bankroll=10.0)
    assert result.total_bets == 0
    assert result.roi == 0.0
    assert result.equity_curve == [10.0]


def test_compute_backtest_metrics_all_wins() -> None:
    bets = [_bet(price=2.0, stake=1.0, outcome="win") for _ in range(3)]
    result = compute_backtest_metrics(bets, initial_bankroll=1.0)

    assert result.total_bets == 3
    assert result.wins == 3
    assert result.hit_rate == 1.0
    assert result.total_staked == pytest.approx(3.0)
    assert result.total_profit == pytest.approx(3.0)
    assert result.roi == pytest.approx(1.0)
    assert result.yield_pct == pytest.approx(result.roi)
    assert result.profit_factor is None  # sem nenhuma perda
    assert result.max_drawdown == 0.0
    assert result.equity_curve == [1.0, 2.0, 3.0, 4.0]


def test_compute_backtest_metrics_all_losses() -> None:
    bets = [_bet(price=2.0, stake=1.0, outcome="loss") for _ in range(3)]
    result = compute_backtest_metrics(bets, initial_bankroll=10.0)

    assert result.wins == 0
    assert result.hit_rate == 0.0
    assert result.total_profit == pytest.approx(-3.0)
    assert result.roi == pytest.approx(-1.0)
    assert result.profit_factor == pytest.approx(0.0)
    assert result.equity_curve == [10.0, 9.0, 8.0, 7.0]
    assert result.max_drawdown == pytest.approx(0.3)


def test_compute_backtest_metrics_pushes_are_excluded_from_hit_rate() -> None:
    bets = [
        _bet(price=2.0, stake=1.0, outcome="win"),
        _bet(price=2.0, stake=1.0, outcome="loss"),
        _bet(price=2.0, stake=1.0, outcome="push"),
    ]
    result = compute_backtest_metrics(bets, initial_bankroll=10.0)

    assert result.total_bets == 3
    assert result.pushes == 1
    assert result.hit_rate == pytest.approx(0.5)  # 1 vitória / (1 vitória + 1 derrota), push ignorado
    assert result.total_profit == pytest.approx(0.0)  # +1 (win) - 1 (loss) + 0 (push)


def test_max_drawdown_measures_the_worst_peak_to_trough_decline() -> None:
    # bankroll: 10 -> 12 (+2) -> 9 (-3) -> 10 (+1)
    bets = [
        _bet(price=3.0, stake=1.0, outcome="win"),  # profit = 1*(3-1) = 2
        _bet(price=1.0, stake=3.0, outcome="loss"),  # profit = -3
        _bet(price=2.0, stake=1.0, outcome="win"),  # profit = 1
    ]
    result = compute_backtest_metrics(bets, initial_bankroll=10.0)

    assert result.equity_curve == [10.0, 12.0, 9.0, 10.0]
    assert result.max_drawdown == pytest.approx((12.0 - 9.0) / 12.0)


def test_sharpe_ratio_matches_manual_calculation() -> None:
    bets = [
        _bet(price=2.5, stake=1.0, outcome="win"),  # retorno = 1.5
        _bet(price=1.0, stake=1.0, outcome="loss"),  # retorno = -1.0
        _bet(price=2.0, stake=1.0, outcome="win"),  # retorno = 1.0
    ]
    result = compute_backtest_metrics(bets, initial_bankroll=10.0)

    returns = [1.5, -1.0, 1.0]
    expected_sharpe = statistics.mean(returns) / statistics.pstdev(returns)
    assert result.sharpe_ratio == pytest.approx(expected_sharpe, abs=1e-3)


def test_sharpe_ratio_is_none_with_fewer_than_two_bets() -> None:
    result = compute_backtest_metrics([_bet(price=2.0, stake=1.0, outcome="win")], initial_bankroll=10.0)
    assert result.sharpe_ratio is None


def test_average_clv_ignores_bets_without_closing_price() -> None:
    bets = [
        _bet(price=2.0, stake=1.0, outcome="win", closing_price=1.8),  # CLV = +0.1111
        _bet(price=1.8, stake=1.0, outcome="loss", closing_price=2.0),  # CLV = -0.1
        _bet(price=2.0, stake=1.0, outcome="win", closing_price=None),  # ignorado
    ]
    result = compute_backtest_metrics(bets, initial_bankroll=10.0)

    expected = ((2.0 - 1.8) / 1.8 + (1.8 - 2.0) / 2.0) / 2
    assert result.average_clv == pytest.approx(expected, abs=1e-3)


def test_average_clv_is_none_when_no_bet_has_closing_price() -> None:
    bets = [_bet(price=2.0, stake=1.0, outcome="win")]
    result = compute_backtest_metrics(bets, initial_bankroll=10.0)
    assert result.average_clv is None


def test_profit_factor_is_ratio_of_gross_profit_to_gross_loss() -> None:
    bets = [
        _bet(price=3.0, stake=1.0, outcome="win"),  # +2
        _bet(price=2.0, stake=1.0, outcome="win"),  # +1
        _bet(price=2.0, stake=2.0, outcome="loss"),  # -2
    ]
    result = compute_backtest_metrics(bets, initial_bankroll=10.0)
    assert result.profit_factor == pytest.approx(3.0 / 2.0)
