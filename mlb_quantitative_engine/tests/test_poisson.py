from __future__ import annotations

import pytest
from scipy.stats import nbinom as scipy_nbinom
from scipy.stats import poisson as scipy_poisson

from mlb_quantitative_engine.analytics.poisson import (
    DEFAULT_OVERDISPERSION,
    confidence_interval,
    probability_over,
    probability_under,
    score_distribution,
)


def test_probability_over_matches_scipy_negative_binomial_directly() -> None:
    mean = 8.6
    line = 8.5
    r = mean / (DEFAULT_OVERDISPERSION - 1.0)
    p = 1.0 / DEFAULT_OVERDISPERSION
    expected = 1.0 - scipy_nbinom.cdf(8, r, p)
    assert probability_over(mean, line) == pytest.approx(expected)


def test_probability_over_and_under_sum_to_one_for_half_integer_line() -> None:
    mean = 8.6
    line = 8.5
    assert probability_over(mean, line) + probability_under(mean, line) == pytest.approx(1.0)


def test_probability_under_excludes_the_line_itself_for_integer_lines() -> None:
    """Para linha inteira N, under = P(X < N), não P(X <= N)."""
    mean = 9.0
    line = 9.0
    r = mean / (DEFAULT_OVERDISPERSION - 1.0)
    p = 1.0 / DEFAULT_OVERDISPERSION
    expected_under = scipy_nbinom.cdf(8, r, p)
    assert probability_under(mean, line) == pytest.approx(expected_under)


def test_higher_projected_mean_increases_probability_of_over() -> None:
    line = 8.5
    low_mean_prob = probability_over(6.0, line)
    high_mean_prob = probability_over(11.0, line)
    assert high_mean_prob > low_mean_prob


def test_probability_over_is_zero_when_mean_is_non_positive() -> None:
    assert probability_over(0.0, 8.5) == 0.0
    assert probability_under(0.0, 8.5) == 1.0


def test_score_distribution_sums_close_to_one() -> None:
    distribution = score_distribution(projected_mean=8.6, max_runs=60)
    assert sum(distribution.values()) == pytest.approx(1.0, abs=1e-4)


def test_score_distribution_peaks_near_projected_mean() -> None:
    mean = 8.6
    distribution = score_distribution(projected_mean=mean, max_runs=40)
    most_likely_score = max(distribution, key=distribution.get)
    assert abs(most_likely_score - mean) <= 2


def test_confidence_interval_covers_at_least_the_requested_probability() -> None:
    mean = 8.6
    lower, upper = confidence_interval(mean, confidence=0.95)
    r = mean / (DEFAULT_OVERDISPERSION - 1.0)
    p = 1.0 / DEFAULT_OVERDISPERSION
    covered = scipy_nbinom.cdf(upper, r, p) - scipy_nbinom.cdf(lower - 1, r, p)
    assert covered >= 0.94  # tolerância pequena por causa da discretização


def test_confidence_interval_is_degenerate_for_zero_mean() -> None:
    assert confidence_interval(0.0) == (0, 0)


# --- overdispersion=1.0 deve recuperar a Poisson pura exatamente (caso-limite) ---


def test_overdispersion_one_recovers_poisson_probability_over() -> None:
    mean = 8.6
    line = 8.5
    expected = 1.0 - scipy_poisson.cdf(8, mean)
    assert probability_over(mean, line, overdispersion=1.0) == pytest.approx(expected)


def test_overdispersion_one_recovers_poisson_probability_under() -> None:
    mean = 9.0
    line = 9.0
    expected = scipy_poisson.cdf(8, mean)
    assert probability_under(mean, line, overdispersion=1.0) == pytest.approx(expected)


def test_overdispersion_one_recovers_poisson_score_distribution() -> None:
    mean = 8.6
    poisson_dist = {k: float(scipy_poisson.pmf(k, mean)) for k in range(31)}
    nb_dist = score_distribution(mean, max_runs=30, overdispersion=1.0)
    for k in poisson_dist:
        assert nb_dist[k] == pytest.approx(poisson_dist[k])


# --- overdispersion > 1.0 deve produzir probabilidades menos extremas que a Poisson ---


def test_higher_overdispersion_pulls_extreme_probabilities_toward_fifty_percent() -> None:
    """Uma NB com mais overdispersão que a Poisson (φ>1) deve dar menos certeza (probabilidade
    mais perto de 50%) para um evento que a Poisson já considerava bem provável."""
    mean = 11.0
    line = 8.5  # média bem acima da linha -> Poisson já dá probabilidade alta de Over
    poisson_prob = probability_over(mean, line, overdispersion=1.0)
    nb_prob = probability_over(mean, line, overdispersion=DEFAULT_OVERDISPERSION)
    assert poisson_prob > 0.5
    assert nb_prob < poisson_prob
