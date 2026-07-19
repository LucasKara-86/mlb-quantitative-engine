from __future__ import annotations

from dataclasses import replace

import pytest

from mlb_quantitative_engine.analytics.sabermetrics import (
    BattingStatLine,
    LeagueConstants,
    PitchingStatLine,
    aggregate_batting_stat_lines,
    aggregate_pitching_stat_lines,
    calculate_avg,
    calculate_babip,
    calculate_bb_percent,
    calculate_era,
    calculate_fb_percent,
    calculate_fip,
    calculate_gb_percent,
    calculate_hr_per_9,
    calculate_iso,
    calculate_k_percent,
    calculate_lob_percent,
    calculate_obp,
    calculate_ops,
    calculate_runs_created,
    calculate_slg,
    calculate_whip,
    calculate_woba,
    calculate_wrc_plus,
    calculate_xfip,
    compute_batting_metrics,
    compute_pitching_metrics,
    shrink_toward_league_average,
)


# ---------------------------------------------------------------------------
# Batting — valores calculados manualmente com números redondos
# ---------------------------------------------------------------------------

_BATTER = BattingStatLine(ab=500, h=150, doubles=30, triples=5, hr=20, bb=60, ibb=5, hbp=5, sf=5, k=100)


def test_singles_and_total_bases_derived_correctly() -> None:
    assert _BATTER.singles == 95  # 150 - 30 - 5 - 20
    assert _BATTER.total_bases == 250  # 95 + 60 + 15 + 80


def test_avg() -> None:
    assert calculate_avg(_BATTER) == pytest.approx(150 / 500)


def test_obp() -> None:
    assert calculate_obp(_BATTER) == pytest.approx((150 + 60 + 5) / (500 + 60 + 5 + 5))


def test_slg() -> None:
    assert calculate_slg(_BATTER) == pytest.approx(250 / 500)


def test_ops_equals_obp_plus_slg() -> None:
    assert calculate_ops(_BATTER) == pytest.approx(calculate_obp(_BATTER) + calculate_slg(_BATTER))


def test_iso_equals_slg_minus_avg() -> None:
    assert calculate_iso(_BATTER) == pytest.approx(calculate_slg(_BATTER) - calculate_avg(_BATTER))


def test_babip() -> None:
    assert calculate_babip(_BATTER) == pytest.approx((150 - 20) / (500 - 100 - 20 + 5))


def test_woba_matches_weighted_formula() -> None:
    constants = LeagueConstants()
    expected = (
        constants.w_bb * (60 - 5)
        + constants.w_hbp * 5
        + constants.w_1b * 95
        + constants.w_2b * 30
        + constants.w_3b * 5
        + constants.w_hr * 20
    ) / (500 + 60 - 5 + 5 + 5)
    assert calculate_woba(_BATTER, constants) == pytest.approx(expected)


def test_wrc_plus_is_100_when_woba_equals_league_average() -> None:
    """Identidade sabermétrica: se wOBA do jogador == wOBA da liga, wRC+ deve ser exatamente 100."""
    base_constants = LeagueConstants()
    woba = calculate_woba(_BATTER, base_constants)
    constants = replace(base_constants, league_woba=woba)
    assert calculate_wrc_plus(_BATTER, constants) == pytest.approx(100.0)


def test_wrc_plus_above_100_when_woba_above_league_average() -> None:
    base_constants = LeagueConstants()
    woba = calculate_woba(_BATTER, base_constants)
    constants = replace(base_constants, league_woba=woba - 0.020)
    assert calculate_wrc_plus(_BATTER, constants) > 100.0


def test_runs_created() -> None:
    assert calculate_runs_created(_BATTER) == pytest.approx((150 + 60) * 250 / (500 + 60))


def test_compute_batting_metrics_returns_all_fields_rounded() -> None:
    metrics = compute_batting_metrics(_BATTER)
    assert metrics.avg == round(150 / 500, 4)
    assert metrics.ops == round(metrics.obp + metrics.slg, 4)


def test_aggregate_batting_stat_lines_sums_counting_stats() -> None:
    player_a = BattingStatLine(ab=300, h=90, hr=10, bb=30, k=60)
    player_b = BattingStatLine(ab=400, h=100, hr=15, bb=40, k=90)

    aggregated = aggregate_batting_stat_lines([player_a, player_b])

    assert aggregated.ab == 700
    assert aggregated.h == 190
    assert aggregated.hr == 25
    assert aggregated.bb == 70
    assert aggregated.k == 150


def test_aggregate_batting_stat_lines_avg_is_not_average_of_individual_avgs() -> None:
    """AVG do agregado deve vir de somar AB/H, não da média simples dos AVGs individuais."""
    high_volume_low_avg = BattingStatLine(ab=500, h=100)  # .200
    low_volume_high_avg = BattingStatLine(ab=10, h=5)  # .500

    aggregated = aggregate_batting_stat_lines([high_volume_low_avg, low_volume_high_avg])
    aggregate_avg = calculate_avg(aggregated)

    naive_average_of_avgs = (0.200 + 0.500) / 2
    assert aggregate_avg != pytest.approx(naive_average_of_avgs)
    assert aggregate_avg == pytest.approx(105 / 510)


def test_batting_metrics_handle_zero_at_bats_without_crashing() -> None:
    empty = BattingStatLine(ab=0, h=0)
    metrics = compute_batting_metrics(empty)
    assert metrics.avg == 0.0
    assert metrics.slg == 0.0
    assert metrics.babip == 0.0


# ---------------------------------------------------------------------------
# Pitching — valores calculados manualmente com números redondos
# ---------------------------------------------------------------------------

_PITCHER = PitchingStatLine(
    outs=540,  # 180 innings
    h=150,
    er=80,
    r=85,
    hr=10,
    bb=30,
    hbp=5,
    k=100,
    batters_faced=750,
    ground_balls=300,
    fly_balls=150,
    line_drives=100,
)


def test_innings_pitched_from_outs() -> None:
    assert _PITCHER.innings_pitched == pytest.approx(180.0)


def test_era() -> None:
    assert calculate_era(_PITCHER) == pytest.approx(80 / 180 * 9)


def test_whip() -> None:
    assert calculate_whip(_PITCHER) == pytest.approx((30 + 150) / 180)


def test_k_percent_and_bb_percent() -> None:
    assert calculate_k_percent(_PITCHER) == pytest.approx(100 / 750)
    assert calculate_bb_percent(_PITCHER) == pytest.approx(30 / 750)


def test_hr_per_9() -> None:
    assert calculate_hr_per_9(_PITCHER) == pytest.approx(10 / 180 * 9)


def test_fip() -> None:
    constants = LeagueConstants()
    expected = (13 * 10 + 3 * (30 + 5) - 2 * 100) / 180 + constants.fip_constant
    assert calculate_fip(_PITCHER, constants) == pytest.approx(expected)


def test_xfip_uses_league_average_hr_per_fb_instead_of_actual_hr() -> None:
    constants = LeagueConstants()
    expected_hr = 150 * constants.league_hr_per_fb
    expected = (13 * expected_hr + 3 * (30 + 5) - 2 * 100) / 180 + constants.fip_constant
    assert calculate_xfip(_PITCHER, constants) == pytest.approx(expected)


def test_lob_percent() -> None:
    expected = (150 + 30 + 5 - 85) / (150 + 30 + 5 - 1.4 * 10)
    assert calculate_lob_percent(_PITCHER) == pytest.approx(expected)


def test_gb_percent_and_fb_percent() -> None:
    total = 300 + 150 + 100
    assert calculate_gb_percent(_PITCHER) == pytest.approx(300 / total)
    assert calculate_fb_percent(_PITCHER) == pytest.approx(150 / total)


def test_batted_ball_percentages_are_none_without_batted_ball_data() -> None:
    pitcher = PitchingStatLine(outs=180, h=50, er=20, batters_faced=250)
    assert calculate_gb_percent(pitcher) is None
    assert calculate_fb_percent(pitcher) is None


def test_xfip_is_none_without_batted_ball_data() -> None:
    """Sem fly_balls, xFIP não deve virar um número (falsamente baixo) — deve ser None."""
    pitcher = PitchingStatLine(outs=180, h=50, er=20, hr=5, bb=10, k=60, batters_faced=250)
    assert calculate_xfip(pitcher) is None


def test_compute_pitching_metrics_returns_all_fields_rounded() -> None:
    metrics = compute_pitching_metrics(_PITCHER)
    assert metrics.era == round(80 / 180 * 9, 2)
    assert metrics.k_minus_bb_percent == pytest.approx(metrics.k_percent - metrics.bb_percent)


def test_pitching_metrics_handle_zero_innings_without_crashing() -> None:
    empty = PitchingStatLine(outs=0, h=0, er=0, batters_faced=0)
    metrics = compute_pitching_metrics(empty)
    assert metrics.era == 0.0
    assert metrics.whip == 0.0
    assert metrics.gb_percent is None
    assert metrics.xfip is None


def test_aggregate_pitching_stat_lines_sums_counting_stats() -> None:
    reliever_a = PitchingStatLine(outs=60, h=15, er=6, r=6, hr=2, bb=8, k=25, batters_faced=90)
    reliever_b = PitchingStatLine(outs=45, h=12, er=5, r=5, hr=1, bb=6, k=20, batters_faced=70)

    aggregated = aggregate_pitching_stat_lines([reliever_a, reliever_b])

    assert aggregated.outs == 105
    assert aggregated.h == 27
    assert aggregated.er == 11
    assert aggregated.k == 45
    assert aggregated.batters_faced == 160


def test_aggregate_pitching_stat_lines_era_is_not_average_of_individual_eras() -> None:
    """ERA do bullpen agregado deve vir de ER/IP somados, não da média das ERAs individuais."""
    high_volume_low_era = PitchingStatLine(outs=180, h=100, er=30, batters_faced=700)  # ERA 1.50
    low_volume_high_era = PitchingStatLine(outs=6, h=5, er=5, batters_faced=15)  # ERA 22.50

    aggregated = aggregate_pitching_stat_lines([high_volume_low_era, low_volume_high_era])
    aggregate_era = calculate_era(aggregated)

    naive_average_of_eras = (1.50 + 22.50) / 2
    assert aggregate_era != pytest.approx(naive_average_of_eras)
    assert aggregate_era == pytest.approx(35 / (186 / 3) * 9)


def test_compute_batting_metrics_exposes_plate_appearances() -> None:
    metrics = compute_batting_metrics(_BATTER)
    assert metrics.plate_appearances == _BATTER.plate_appearances


def test_compute_pitching_metrics_exposes_innings_pitched() -> None:
    metrics = compute_pitching_metrics(_PITCHER)
    assert metrics.innings_pitched == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# Encolhimento Bayesiano (shrinkage) em direção à média da liga
# ---------------------------------------------------------------------------


def test_shrink_toward_league_average_large_sample_stays_close_to_observed() -> None:
    """Amostra muito maior que o ponto de estabilização -> quase nenhum encolhimento."""
    shrunk = shrink_toward_league_average(observed=150.0, league_average=100.0, sample_size=5000, stabilization_point=250)
    assert shrunk == pytest.approx(150.0, abs=3.0)


def test_shrink_toward_league_average_small_sample_pulls_hard_toward_average() -> None:
    """Amostra muito menor que o ponto de estabilização -> quase todo encolhido pra média."""
    shrunk = shrink_toward_league_average(observed=150.0, league_average=100.0, sample_size=5, stabilization_point=250)
    assert shrunk == pytest.approx(100.0, abs=3.0)


def test_shrink_toward_league_average_equals_midpoint_when_sample_equals_stabilization_point() -> None:
    shrunk = shrink_toward_league_average(observed=150.0, league_average=100.0, sample_size=250, stabilization_point=250)
    assert shrunk == pytest.approx(125.0)


def test_shrink_toward_league_average_zero_sample_returns_league_average() -> None:
    assert shrink_toward_league_average(observed=200.0, league_average=100.0, sample_size=0, stabilization_point=250) == 100.0


def test_shrink_toward_league_average_never_overshoots_past_the_observed_value() -> None:
    """O encolhimento é sempre um ponto ENTRE observado e média da liga, nunca além de nenhum dos dois."""
    shrunk = shrink_toward_league_average(observed=150.0, league_average=100.0, sample_size=80, stabilization_point=250)
    assert 100.0 < shrunk < 150.0
