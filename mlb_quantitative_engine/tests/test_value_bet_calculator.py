from __future__ import annotations

import pytest

from mlb_quantitative_engine.analytics.value_bet_calculator import (
    MAX_STAKE_FRACTION,
    MIN_CONFIDENCE,
    MIN_EDGE,
    MIN_EXPECTED_VALUE,
    PRICE_TOLERANCE,
    cap_stake_fraction,
    calculate_edge,
    evaluate_game_total_value_bets,
    evaluate_team_total_value_bets,
    expected_value,
    kelly_fraction,
    minimum_acceptable_price,
    remove_vig,
)


def test_remove_vig_sums_to_one() -> None:
    """Identidade do de-vig: as probabilidades justas sempre somam exatamente 1.0."""
    fair_over, fair_under = remove_vig(1.90, 1.90)
    assert fair_over + fair_under == pytest.approx(1.0)


def test_remove_vig_symmetric_prices_split_evenly() -> None:
    fair_over, fair_under = remove_vig(1.90, 1.90)
    assert fair_over == pytest.approx(0.5)
    assert fair_under == pytest.approx(0.5)


def test_remove_vig_asymmetric_prices() -> None:
    raw_over = 1.0 / 1.50
    raw_under = 1.0 / 2.80
    overround = raw_over + raw_under
    fair_over, fair_under = remove_vig(1.50, 2.80)
    assert fair_over == pytest.approx(raw_over / overround)
    assert fair_under == pytest.approx(raw_under / overround)


def test_remove_vig_fair_probability_is_lower_than_raw_due_to_vig() -> None:
    """A probabilidade justa deve ser menor que a bruta (1/odd), porque o vig infla a bruta."""
    fair_over, _ = remove_vig(1.90, 1.90)
    raw_over = 1.0 / 1.90
    assert fair_over < raw_over


def test_expected_value_positive_edge() -> None:
    assert expected_value(0.55, 1.91) == pytest.approx(0.55 * 1.91 - 1.0)


def test_expected_value_negative_when_probability_too_low() -> None:
    assert expected_value(0.40, 1.91) < 0


def test_calculate_edge() -> None:
    assert calculate_edge(0.55, 0.50) == pytest.approx(0.05)


def test_kelly_fraction_positive_edge() -> None:
    expected = (0.55 * 1.91 - 1.0) / (1.91 - 1.0)
    assert kelly_fraction(0.55, 1.91) == pytest.approx(expected)


def test_kelly_fraction_is_zero_for_negative_edge() -> None:
    assert kelly_fraction(0.40, 1.91) == 0.0


def test_kelly_fraction_is_capped_at_one() -> None:
    assert kelly_fraction(0.99, 50.0) <= 1.0


def test_evaluate_game_total_value_bets_flags_clear_value_as_meeting_criteria() -> None:
    """Projeção de 10.5 corridas contra uma linha de mercado de 8.5 é um edge grande e óbvio."""
    over_bet, under_bet = evaluate_game_total_value_bets(
        game_pk=1,
        home_team="Home",
        away_team="Away",
        projected_total_runs=10.5,
        point=8.5,
        over_price=1.90,
        over_bookmaker="FanDuel",
        under_price=1.90,
        under_bookmaker="DraftKings",
        confidence_score=90.0,
    )

    assert over_bet.meets_criteria is True
    assert over_bet.edge > 0
    assert over_bet.expected_value > 0
    assert under_bet.meets_criteria is False  # o lado oposto não deve ter edge


def test_evaluate_game_total_value_bets_respects_confidence_threshold() -> None:
    """Mesmo com edge claro, confiança abaixo do limiar (70%) não deve qualificar a aposta."""
    over_bet, _ = evaluate_game_total_value_bets(
        game_pk=1,
        home_team="Home",
        away_team="Away",
        projected_total_runs=10.5,
        point=8.5,
        over_price=1.90,
        over_bookmaker="FanDuel",
        under_price=1.90,
        under_bookmaker="DraftKings",
        confidence_score=50.0,
    )

    assert over_bet.confidence_score < MIN_CONFIDENCE
    assert over_bet.meets_criteria is False


def test_evaluate_game_total_value_bets_no_edge_when_projection_matches_market() -> None:
    """Se nossa projeção bate exatamente com a linha do mercado, nenhum lado deve ter valor."""
    over_bet, under_bet = evaluate_game_total_value_bets(
        game_pk=1,
        home_team="Home",
        away_team="Away",
        projected_total_runs=8.5,
        point=8.5,
        over_price=1.90,
        over_bookmaker="FanDuel",
        under_price=1.90,
        under_bookmaker="DraftKings",
        confidence_score=90.0,
    )

    assert over_bet.meets_criteria is False
    assert under_bet.meets_criteria is False


def test_evaluate_game_total_value_bets_uses_correct_market_labels() -> None:
    over_bet, under_bet = evaluate_game_total_value_bets(
        game_pk=42,
        home_team="Yankees",
        away_team="Red Sox",
        projected_total_runs=9.0,
        point=8.5,
        over_price=1.90,
        over_bookmaker="FanDuel",
        under_price=1.90,
        under_bookmaker="DraftKings",
        confidence_score=90.0,
    )

    assert over_bet.market == "game_total_over"
    assert under_bet.market == "game_total_under"
    assert over_bet.game_pk == 42
    assert over_bet.home_team == "Yankees"


def test_evaluate_team_total_value_bets_uses_team_label_as_market_prefix() -> None:
    over_bet, under_bet = evaluate_team_total_value_bets(
        game_pk=1,
        home_team="Yankees",
        away_team="Red Sox",
        team_label="home_team_total",
        projected_team_runs=5.5,
        point=4.5,
        over_price=1.90,
        over_bookmaker="FanDuel",
        under_price=1.90,
        under_bookmaker="DraftKings",
        confidence_score=90.0,
    )

    assert over_bet.market == "home_team_total_over"
    assert under_bet.market == "home_team_total_under"


def test_evaluate_team_total_value_bets_computes_edge_from_team_projection() -> None:
    """Projeção de time bem acima da linha de time deve gerar edge e EV positivos no Over."""
    over_bet, _ = evaluate_team_total_value_bets(
        game_pk=1,
        home_team="Yankees",
        away_team="Red Sox",
        team_label="away_team_total",
        projected_team_runs=6.5,
        point=4.5,
        over_price=1.90,
        over_bookmaker="FanDuel",
        under_price=1.90,
        under_bookmaker="DraftKings",
        confidence_score=90.0,
    )

    assert over_bet.edge > 0.04
    assert over_bet.expected_value > 0.05
    assert over_bet.meets_criteria is True


def test_moderate_probability_still_qualifies_when_edge_ev_and_confidence_pass() -> None:
    """O piso fixo de probabilidade projetada (>= 64%) foi removido -- uma probabilidade
    mais modesta (~59-60%) não deve mais bloquear a recomendação por si só, desde que
    edge/EV/confiança passem nos próprios limiares (ver docstring do módulo)."""
    over_bet, _ = evaluate_game_total_value_bets(
        game_pk=1, home_team="Home", away_team="Away",
        projected_total_runs=9.4, point=8.5,
        over_price=1.90, over_bookmaker="FanDuel",
        under_price=1.90, under_bookmaker="DraftKings",
        confidence_score=90.0,
    )

    assert over_bet.edge > MIN_EDGE
    assert over_bet.expected_value > MIN_EXPECTED_VALUE
    assert over_bet.confidence_score > MIN_CONFIDENCE
    assert over_bet.projected_probability < 0.64  # a antiga linha de corte
    assert over_bet.meets_criteria is True


def test_cap_stake_fraction_caps_at_max() -> None:
    assert cap_stake_fraction(0.30, max_stake_fraction=0.02) == pytest.approx(0.02)


def test_cap_stake_fraction_leaves_smaller_values_untouched() -> None:
    assert cap_stake_fraction(0.01, max_stake_fraction=0.02) == pytest.approx(0.01)


def test_minimum_acceptable_price_default_tolerance() -> None:
    assert minimum_acceptable_price(2.07) == pytest.approx(2.07 * 0.95)


def test_minimum_acceptable_price_custom_tolerance() -> None:
    assert minimum_acceptable_price(2.00, tolerance=0.20) == pytest.approx(1.60)


def test_evaluate_side_caps_suggested_stake_even_with_huge_edge() -> None:
    """Um edge enorme geraria um Kelly Fracionado bem acima de 2% -- a recomendação
    deve ficar limitada ao teto de 2%, mesmo que o Kelly "puro" seja maior."""
    over_bet, _ = evaluate_game_total_value_bets(
        game_pk=1, home_team="Home", away_team="Away",
        projected_total_runs=15.0, point=5.5,
        over_price=2.50, over_bookmaker="FanDuel",
        under_price=1.90, under_bookmaker="DraftKings",
        confidence_score=90.0,
    )

    assert over_bet.kelly_fraction_quarter > MAX_STAKE_FRACTION
    assert over_bet.suggested_stake_fraction == pytest.approx(MAX_STAKE_FRACTION)


def test_evaluate_side_computes_minimum_acceptable_price_from_best_price() -> None:
    over_bet, under_bet = evaluate_game_total_value_bets(
        game_pk=1, home_team="Home", away_team="Away",
        projected_total_runs=9.0, point=8.5,
        over_price=2.07, over_bookmaker="BetRivers",
        under_price=1.90, under_bookmaker="DraftKings",
        confidence_score=90.0,
    )

    assert over_bet.minimum_acceptable_price == pytest.approx(2.07 * (1 - PRICE_TOLERANCE))
    assert under_bet.minimum_acceptable_price == pytest.approx(1.90 * (1 - PRICE_TOLERANCE))
