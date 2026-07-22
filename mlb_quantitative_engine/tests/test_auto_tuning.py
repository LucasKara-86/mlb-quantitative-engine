from __future__ import annotations

from mlb_quantitative_engine.analytics.auto_tuning import (
    MIN_SAMPLE_THRESHOLD,
    PARAMETER_BOUNDS,
    find_negative_roi_segments,
    propose_adjustments,
)
from mlb_quantitative_engine.analytics.backtesting import BacktestResult
from mlb_quantitative_engine.analytics.calibration import ReliabilityBucket

_BASE_VALUES = {
    "overdispersion": 1.4,
    "mean_uncertainty_pct": 0.12,
    "min_edge": 0.04,
    "min_confidence": 70.0,
}


def _bucket(predicted: float, actual: float, count: int) -> ReliabilityBucket:
    return ReliabilityBucket(
        bucket_label=f"{predicted:.0%}", predicted_probability_mean=predicted, actual_win_rate=actual, count=count
    )


def _backtest_result(total_bets: int, roi: float) -> BacktestResult:
    wins = total_bets // 2
    return BacktestResult(
        total_bets=total_bets, wins=wins, losses=total_bets - wins, pushes=0,
        hit_rate=wins / total_bets if total_bets else 0.0, total_staked=total_bets * 0.02,
        total_profit=roi * total_bets * 0.02, roi=roi, yield_pct=roi, profit_factor=None,
        max_drawdown=0.0, sharpe_ratio=None, average_clv=None, equity_curve=[],
    )


def test_no_adjustment_when_everything_within_tolerance() -> None:
    reliability = [_bucket(0.75, 0.72, MIN_SAMPLE_THRESHOLD)]
    outcome = propose_adjustments(
        reliability=reliability, overall_roi=0.02, overall_n=50, current_values=_BASE_VALUES,
    )
    assert outcome.accepted == []
    assert outcome.deferred == []


def test_no_adjustment_when_sample_too_small() -> None:
    reliability = [_bucket(0.80, 0.40, MIN_SAMPLE_THRESHOLD - 1)]  # overconfidence enorme, amostra pequena demais
    outcome = propose_adjustments(
        reliability=reliability, overall_roi=None, overall_n=0, current_values=_BASE_VALUES,
    )
    assert outcome.accepted == []


def test_overconfidence_increases_overdispersion() -> None:
    """Reproduz o padrão do incidente documentado: previu ~75%, acertou ~40% -- modelo
    confiante demais -- deve alargar a distribuição (aumentar overdispersão)."""
    reliability = [_bucket(0.75, 0.40, 30)]
    outcome = propose_adjustments(
        reliability=reliability, overall_roi=0.0, overall_n=0, current_values=_BASE_VALUES,
    )
    assert len(outcome.accepted) == 1
    adjustment = outcome.accepted[0]
    assert adjustment.parameter_name == "overdispersion"
    assert adjustment.new_value > adjustment.old_value
    assert adjustment.sample_size == 30


def test_underconfidence_decreases_overdispersion() -> None:
    """Modelo previu 70% e acertou 95% -- conservador demais -- deve estreitar a
    distribuição (reduzir overdispersão) para capturar o EV que fica na mesa."""
    reliability = [_bucket(0.70, 0.95, 30)]
    outcome = propose_adjustments(
        reliability=reliability, overall_roi=0.0, overall_n=0, current_values=_BASE_VALUES,
    )
    assert len(outcome.accepted) == 1
    adjustment = outcome.accepted[0]
    assert adjustment.parameter_name == "overdispersion"
    assert adjustment.new_value < adjustment.old_value


def test_reliability_buckets_below_60_percent_are_ignored() -> None:
    """O critério de Value Bet só admite apostas com probabilidade projetada relevante;
    uma faixa baixa (ex.: 40%) mal calibrada não deve mexer no parâmetro estrutural."""
    reliability = [_bucket(0.45, 0.10, 30)]
    outcome = propose_adjustments(
        reliability=reliability, overall_roi=0.0, overall_n=0, current_values=_BASE_VALUES,
    )
    assert outcome.accepted == []


def test_overdispersion_falls_back_to_mean_uncertainty_when_at_bound() -> None:
    values = dict(_BASE_VALUES, overdispersion=PARAMETER_BOUNDS["overdispersion"].maximum)
    reliability = [_bucket(0.75, 0.40, 30)]
    outcome = propose_adjustments(reliability=reliability, overall_roi=0.0, overall_n=0, current_values=values)
    assert len(outcome.accepted) == 1
    assert outcome.accepted[0].parameter_name == "mean_uncertainty_pct"


def test_overdispersion_never_exceeds_bounds() -> None:
    bounds = PARAMETER_BOUNDS["overdispersion"]
    values = dict(_BASE_VALUES, overdispersion=bounds.maximum - bounds.step / 2)
    reliability = [_bucket(0.75, 0.40, 30)]
    outcome = propose_adjustments(reliability=reliability, overall_roi=0.0, overall_n=0, current_values=values)
    assert outcome.accepted[0].new_value <= bounds.maximum


def test_negative_overall_roi_raises_min_edge() -> None:
    outcome = propose_adjustments(
        reliability=[], overall_roi=-0.10, overall_n=40, current_values=_BASE_VALUES,
    )
    assert len(outcome.accepted) == 1
    adjustment = outcome.accepted[0]
    assert adjustment.parameter_name == "min_edge"
    assert adjustment.new_value > adjustment.old_value


def test_negative_roi_ignored_below_min_sample() -> None:
    outcome = propose_adjustments(
        reliability=[], overall_roi=-0.10, overall_n=MIN_SAMPLE_THRESHOLD - 1, current_values=_BASE_VALUES,
    )
    assert outcome.accepted == []


def test_min_edge_falls_back_to_min_confidence_when_at_bound() -> None:
    values = dict(_BASE_VALUES, min_edge=PARAMETER_BOUNDS["min_edge"].maximum)
    outcome = propose_adjustments(reliability=[], overall_roi=-0.10, overall_n=40, current_values=values)
    assert len(outcome.accepted) == 1
    assert outcome.accepted[0].parameter_name == "min_confidence"


def test_cooldown_defers_a_candidate() -> None:
    reliability = [_bucket(0.75, 0.40, 30)]
    outcome = propose_adjustments(
        reliability=reliability, overall_roi=0.0, overall_n=0, current_values=_BASE_VALUES,
        parameters_in_cooldown=["overdispersion"],
    )
    assert outcome.accepted == []
    assert len(outcome.deferred) == 1
    assert outcome.deferred[0].parameter_name == "overdispersion"


def test_max_changes_per_run_defers_extra_candidates() -> None:
    reliability = [_bucket(0.75, 0.40, 30)]
    outcome = propose_adjustments(
        reliability=reliability, overall_roi=-0.10, overall_n=40, current_values=_BASE_VALUES,
        max_changes_per_run=1,
    )
    assert len(outcome.accepted) == 1
    assert len(outcome.deferred) == 1


def test_find_negative_roi_segments_flags_bad_segment_with_enough_sample() -> None:
    by_market = {
        "game_total": _backtest_result(total_bets=30, roi=0.05),
        "home_team_total": _backtest_result(total_bets=25, roi=-0.10),
        "away_team_total": _backtest_result(total_bets=5, roi=-0.20),  # amostra pequena demais
    }
    findings = find_negative_roi_segments(by_market)
    assert len(findings) == 1
    assert "home_team_total" in findings[0]


def test_find_negative_roi_segments_empty_when_all_healthy() -> None:
    by_market = {"game_total": _backtest_result(total_bets=30, roi=0.05)}
    assert find_negative_roi_segments(by_market) == []
