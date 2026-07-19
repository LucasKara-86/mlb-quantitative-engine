from __future__ import annotations

import pytest

from mlb_quantitative_engine.analytics.monte_carlo import (
    DEFAULT_MEAN_UNCERTAINTY_PCT,
    DEFAULT_RANDOM_SEED,
    simulate_total_probability,
)
from mlb_quantitative_engine.analytics.poisson import probability_over as analytical_probability_over


def test_zero_mean_is_degenerate() -> None:
    result = simulate_total_probability(0.0, 8.5)
    assert result.probability_over == 0.0
    assert result.probability_under == 1.0
    assert result.probability_push == 0.0


def test_probabilities_sum_to_one_for_half_integer_line() -> None:
    result = simulate_total_probability(8.6, 8.5)
    assert result.probability_over + result.probability_under == pytest.approx(1.0)
    assert result.probability_push == 0.0


def test_probabilities_sum_to_one_including_push_for_integer_line() -> None:
    result = simulate_total_probability(9.0, 9.0)
    total = result.probability_over + result.probability_under + result.probability_push
    assert total == pytest.approx(1.0)
    assert result.probability_push > 0.0  # linha inteira -> push deve ser um resultado possível


def test_same_inputs_are_deterministic_across_calls() -> None:
    """Semente fixa por padrão -> mesma entrada sempre produz o mesmo resultado."""
    result_a = simulate_total_probability(8.6, 8.5)
    result_b = simulate_total_probability(8.6, 8.5)
    assert result_a == result_b


def test_different_seed_still_gives_a_close_but_not_identical_result() -> None:
    result_default = simulate_total_probability(8.6, 8.5, random_seed=DEFAULT_RANDOM_SEED)
    result_other = simulate_total_probability(8.6, 8.5, random_seed=123)
    assert result_default.probability_over == pytest.approx(result_other.probability_over, abs=0.02)


def test_higher_projected_mean_increases_probability_of_over() -> None:
    low = simulate_total_probability(6.0, 8.5)
    high = simulate_total_probability(11.0, 8.5)
    assert high.probability_over > low.probability_over


def test_monte_carlo_probability_is_less_extreme_than_analytical_probability() -> None:
    """A camada de incerteza de parâmetro deve puxar a probabilidade pra mais perto de
    50% do que a versão puramente analítica (Binomial Negativa sem incerteza de média),
    para um cenário onde a média está bem acima da linha."""
    mean = 11.0
    line = 8.5
    analytical = analytical_probability_over(mean, line)
    mc = simulate_total_probability(mean, line)
    assert analytical > 0.5
    assert mc.probability_over < analytical


def test_zero_mean_uncertainty_converges_to_analytical_probability() -> None:
    """Sem incerteza de parâmetro (mean_uncertainty_pct ~ 0), o Monte Carlo deve convergir
    para a mesma probabilidade da Binomial Negativa analítica (mesma overdispersão)."""
    mean = 9.2
    line = 8.5
    analytical = analytical_probability_over(mean, line)
    mc = simulate_total_probability(mean, line, mean_uncertainty_pct=1e-6, n_simulations=200_000)
    assert mc.probability_over == pytest.approx(analytical, abs=0.01)


def test_more_mean_uncertainty_pulls_probability_further_toward_fifty_percent() -> None:
    mean = 11.0
    line = 8.5
    low_uncertainty = simulate_total_probability(mean, line, mean_uncertainty_pct=0.05)
    high_uncertainty = simulate_total_probability(mean, line, mean_uncertainty_pct=0.30)
    assert abs(high_uncertainty.probability_over - 0.5) < abs(low_uncertainty.probability_over - 0.5)


def test_simulation_error_is_small_with_default_sample_size() -> None:
    result = simulate_total_probability(8.6, 8.5)
    assert result.simulation_error < 0.01


def test_default_mean_uncertainty_is_a_reasonable_positive_fraction() -> None:
    assert 0.0 < DEFAULT_MEAN_UNCERTAINTY_PCT < 0.5
