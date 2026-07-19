from __future__ import annotations

import pytest

from mlb_quantitative_engine.analytics.calibration import (
    brier_score,
    overall_hit_rate,
    reliability_table,
)


def test_brier_score_is_none_for_empty_predictions() -> None:
    assert brier_score([]) is None


def test_brier_score_is_zero_for_perfect_predictions() -> None:
    predictions = [(1.0, True), (0.0, False), (1.0, True)]
    assert brier_score(predictions) == 0.0


def test_brier_score_is_quarter_for_always_predicting_fifty_percent() -> None:
    predictions = [(0.5, True), (0.5, False), (0.5, True), (0.5, False)]
    assert brier_score(predictions) == pytest.approx(0.25)


def test_brier_score_penalizes_confident_wrong_predictions_more() -> None:
    """Errar com 90% de confiança deve custar mais caro (Brier maior) do que errar com 55%."""
    confident_wrong = brier_score([(0.90, False)])
    unsure_wrong = brier_score([(0.55, False)])
    assert confident_wrong > unsure_wrong


def test_overall_hit_rate_computes_simple_win_fraction() -> None:
    predictions = [(0.7, True), (0.7, True), (0.7, False), (0.7, False)]
    assert overall_hit_rate(predictions) == pytest.approx(0.5)


def test_overall_hit_rate_is_none_for_empty_predictions() -> None:
    assert overall_hit_rate([]) is None


def test_reliability_table_groups_by_bucket() -> None:
    predictions = [
        (0.62, True), (0.63, False),  # bucket 60-65%
        (0.72, True), (0.74, True), (0.73, False),  # bucket 70-75%
    ]
    table = reliability_table(predictions, bucket_edges=(0.60, 0.65, 0.70, 0.75, 1.01))

    labels = {bucket.bucket_label: bucket for bucket in table}
    assert "60%-65%" in labels
    assert labels["60%-65%"].count == 2
    assert labels["60%-65%"].actual_win_rate == pytest.approx(0.5)

    assert "70%-75%" in labels
    assert labels["70%-75%"].count == 3
    assert labels["70%-75%"].actual_win_rate == pytest.approx(2 / 3, abs=1e-3)


def test_reliability_table_omits_empty_buckets() -> None:
    predictions = [(0.62, True)]
    table = reliability_table(predictions, bucket_edges=(0.60, 0.65, 0.70, 0.75, 1.01))
    assert len(table) == 1
    assert table[0].bucket_label == "60%-65%"


def test_reliability_table_is_empty_for_no_predictions() -> None:
    assert reliability_table([]) == []


def test_overconfidence_is_positive_when_model_beats_reality_less_than_predicted() -> None:
    predictions = [(0.90, False), (0.90, False), (0.90, True)]  # previu 90%, acertou 33%
    table = reliability_table(predictions, bucket_edges=(0.85, 0.95))
    assert len(table) == 1
    assert table[0].overconfidence > 0  # modelo confiante demais


def test_overconfidence_is_negative_when_model_is_too_conservative() -> None:
    predictions = [(0.65, True), (0.65, True), (0.65, True)]  # previu 65%, acertou 100%
    table = reliability_table(predictions, bucket_edges=(0.60, 0.70))
    assert len(table) == 1
    assert table[0].overconfidence < 0
