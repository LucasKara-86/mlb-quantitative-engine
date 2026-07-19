from __future__ import annotations

import pytest

from mlb_quantitative_engine.analytics.projections import (
    OpposingPitcherInput,
    ProjectionEngine,
    TeamOffenseInput,
)
from mlb_quantitative_engine.analytics.sabermetrics import LeagueConstants
from mlb_quantitative_engine.models.game_projection import GameProjection


def test_average_matchup_projects_league_average_runs_for_both_teams() -> None:
    """wRC+ 100 (ofensa média) vs FIP == ERA média da liga (pitcher médio) -> corridas médias."""
    constants = LeagueConstants()
    engine = ProjectionEngine(constants=constants)
    average_offense = TeamOffenseInput(wrc_plus=100.0)
    average_pitcher = OpposingPitcherInput(starter_fip=constants.league_avg_era)

    projection = engine.project_game(
        home_team="Team A",
        away_team="Team B",
        home_offense=average_offense,
        away_offense=average_offense,
        home_starting_pitcher=average_pitcher,
        away_starting_pitcher=average_pitcher,
    )

    assert isinstance(projection, GameProjection)
    assert projection.projected_home_runs == pytest.approx(constants.league_avg_runs_per_game, abs=0.01)
    assert projection.projected_away_runs == pytest.approx(constants.league_avg_runs_per_game, abs=0.01)
    assert projection.projected_total_runs == pytest.approx(
        constants.league_avg_runs_per_game * 2, abs=0.02
    )


def test_elite_offense_against_weak_pitcher_projects_above_average() -> None:
    constants = LeagueConstants()
    engine = ProjectionEngine(constants=constants)
    elite_offense = TeamOffenseInput(wrc_plus=130.0)
    weak_pitcher = OpposingPitcherInput(starter_fip=5.50)
    average_offense = TeamOffenseInput(wrc_plus=100.0)
    average_pitcher = OpposingPitcherInput(starter_fip=constants.league_avg_era)

    projection = engine.project_game(
        home_team="Elite Offense Team",
        away_team="League Average Team",
        home_offense=elite_offense,
        away_offense=average_offense,
        home_starting_pitcher=average_pitcher,
        away_starting_pitcher=weak_pitcher,
    )

    assert projection.projected_home_runs > constants.league_avg_runs_per_game


def test_weak_offense_against_ace_pitcher_projects_below_average() -> None:
    constants = LeagueConstants()
    engine = ProjectionEngine(constants=constants)
    weak_offense = TeamOffenseInput(wrc_plus=70.0)
    ace_pitcher = OpposingPitcherInput(starter_fip=2.50)
    average_offense = TeamOffenseInput(wrc_plus=100.0)
    average_pitcher = OpposingPitcherInput(starter_fip=constants.league_avg_era)

    projection = engine.project_game(
        home_team="Weak Offense Team",
        away_team="League Average Team",
        home_offense=weak_offense,
        away_offense=average_offense,
        home_starting_pitcher=average_pitcher,
        away_starting_pitcher=ace_pitcher,
    )

    assert projection.projected_home_runs < constants.league_avg_runs_per_game


def test_park_factor_scales_projection_linearly() -> None:
    constants = LeagueConstants()
    engine = ProjectionEngine(constants=constants)
    offense = TeamOffenseInput(wrc_plus=100.0)
    pitcher = OpposingPitcherInput(starter_fip=constants.league_avg_era)

    neutral_park = engine.project_game(
        "Home", "Away", offense, offense, pitcher, pitcher, park_factor=1.0
    )
    hitter_friendly_park = engine.project_game(
        "Home", "Away", offense, offense, pitcher, pitcher, park_factor=1.20
    )

    assert hitter_friendly_park.projected_total_runs > neutral_park.projected_total_runs
    assert hitter_friendly_park.projected_total_runs == pytest.approx(
        neutral_park.projected_total_runs * 1.20, rel=1e-2
    )


def test_weather_factor_scales_projection_linearly() -> None:
    constants = LeagueConstants()
    engine = ProjectionEngine(constants=constants)
    offense = TeamOffenseInput(wrc_plus=100.0)
    pitcher = OpposingPitcherInput(starter_fip=constants.league_avg_era)

    calm_weather = engine.project_game(
        "Home", "Away", offense, offense, pitcher, pitcher, weather_factor=1.0
    )
    windy_out_weather = engine.project_game(
        "Home", "Away", offense, offense, pitcher, pitcher, weather_factor=1.10
    )

    assert windy_out_weather.projected_total_runs > calm_weather.projected_total_runs


def test_projected_runs_never_go_below_minimum_floor() -> None:
    """Mesmo em um matchup extremo (ofensa péssima vs. pitcher de elite), corridas > 0."""
    engine = ProjectionEngine()
    terrible_offense = TeamOffenseInput(wrc_plus=1.0)
    dominant_pitcher = OpposingPitcherInput(starter_fip=0.50)

    projection = engine.project_game(
        "Home", "Away", terrible_offense, terrible_offense, dominant_pitcher, dominant_pitcher
    )

    assert projection.projected_home_runs >= ProjectionEngine.MIN_PROJECTED_RUNS
    assert projection.projected_away_runs >= ProjectionEngine.MIN_PROJECTED_RUNS


def test_home_pitcher_faces_away_offense_and_vice_versa() -> None:
    """Confirma que os confrontos estão emparelhados corretamente (não invertidos)."""
    constants = LeagueConstants()
    engine = ProjectionEngine(constants=constants)
    strong_offense = TeamOffenseInput(wrc_plus=130.0)
    weak_offense = TeamOffenseInput(wrc_plus=70.0)
    average_pitcher = OpposingPitcherInput(starter_fip=constants.league_avg_era)

    projection = engine.project_game(
        home_team="Home",
        away_team="Away",
        home_offense=strong_offense,
        away_offense=weak_offense,
        home_starting_pitcher=average_pitcher,
        away_starting_pitcher=average_pitcher,
    )

    # o time da casa tem a ofensa forte -> deve projetar mais corridas que o visitante (ofensa fraca)
    assert projection.projected_home_runs > projection.projected_away_runs


def test_without_bullpen_data_uses_starter_fip_only() -> None:
    """Comportamento anterior à integração de bullpen deve ser preservado quando bullpen_fip=None."""
    engine = ProjectionEngine()
    pitcher_without_bullpen = OpposingPitcherInput(starter_fip=3.50)

    assert engine._effective_pitching_fip(pitcher_without_bullpen) == pytest.approx(3.50)


def test_effective_fip_blends_starter_and_bullpen_by_innings_share() -> None:
    engine = ProjectionEngine()
    pitcher = OpposingPitcherInput(starter_fip=3.00, bullpen_fip=5.00)

    expected = (
        3.00 * ProjectionEngine.STARTER_EXPECTED_INNINGS
        + 5.00 * (ProjectionEngine.GAME_INNINGS - ProjectionEngine.STARTER_EXPECTED_INNINGS)
    ) / ProjectionEngine.GAME_INNINGS
    assert engine._effective_pitching_fip(pitcher) == pytest.approx(expected)


def test_fatigued_bullpen_worsens_effective_fip() -> None:
    engine = ProjectionEngine()
    fresh_bullpen = OpposingPitcherInput(starter_fip=3.50, bullpen_fip=4.00, bullpen_fatigue_index=0.0)
    tired_bullpen = OpposingPitcherInput(starter_fip=3.50, bullpen_fip=4.00, bullpen_fatigue_index=100.0)

    assert engine._effective_pitching_fip(tired_bullpen) > engine._effective_pitching_fip(fresh_bullpen)


def test_unavailable_relievers_worsen_effective_fip() -> None:
    engine = ProjectionEngine()
    full_bullpen = OpposingPitcherInput(starter_fip=3.50, bullpen_fip=4.00, bullpen_unavailable_count=0)
    depleted_bullpen = OpposingPitcherInput(starter_fip=3.50, bullpen_fip=4.00, bullpen_unavailable_count=3)

    assert engine._effective_pitching_fip(depleted_bullpen) > engine._effective_pitching_fip(full_bullpen)


def test_tired_bullpen_increases_projected_runs_allowed() -> None:
    """Um bullpen fatigado deve, no fim, se traduzir em mais corridas projetadas para o adversário."""
    constants = LeagueConstants()
    engine = ProjectionEngine(constants=constants)
    offense = TeamOffenseInput(wrc_plus=100.0)
    fresh = OpposingPitcherInput(starter_fip=3.50, bullpen_fip=3.50, bullpen_fatigue_index=0.0)
    tired = OpposingPitcherInput(starter_fip=3.50, bullpen_fip=3.50, bullpen_fatigue_index=100.0)

    projection_fresh = engine.project_game("Home", "Away", offense, offense, fresh, fresh)
    projection_tired = engine.project_game("Home", "Away", offense, offense, tired, tired)

    assert projection_tired.projected_total_runs > projection_fresh.projected_total_runs


# ---------------------------------------------------------------------------
# Encolhimento bayesiano (shrinkage) de amostra pequena
# ---------------------------------------------------------------------------


def test_offense_without_plate_appearances_skips_shrinkage() -> None:
    """Compatibilidade: sem plate_appearances informado, o wrc_plus bruto é usado (comportamento anterior)."""
    engine = ProjectionEngine()
    offense = TeamOffenseInput(wrc_plus=150.0)
    assert engine._shrunk_wrc_plus(offense) == pytest.approx(150.0)


def test_offense_with_small_sample_shrinks_toward_league_average() -> None:
    engine = ProjectionEngine()
    small_sample_hot_streak = TeamOffenseInput(wrc_plus=180.0, plate_appearances=20)
    shrunk = engine._shrunk_wrc_plus(small_sample_hot_streak)
    assert 100.0 < shrunk < 180.0
    assert shrunk < 130.0  # amostra bem pequena -> puxado forte pra média


def test_offense_with_large_sample_barely_shrinks() -> None:
    engine = ProjectionEngine()
    full_season_sample = TeamOffenseInput(wrc_plus=130.0, plate_appearances=5000)
    shrunk = engine._shrunk_wrc_plus(full_season_sample)
    assert shrunk == pytest.approx(130.0, abs=2.0)


def test_starter_without_innings_pitched_skips_shrinkage() -> None:
    engine = ProjectionEngine()
    pitcher = OpposingPitcherInput(starter_fip=2.00)
    assert engine._shrunk_starter_fip(pitcher) == pytest.approx(2.00)


def test_starter_with_small_sample_shrinks_toward_league_average_era() -> None:
    constants = LeagueConstants()
    engine = ProjectionEngine(constants=constants)
    small_sample_ace = OpposingPitcherInput(starter_fip=1.50, starter_innings_pitched=15.0)
    shrunk = engine._shrunk_starter_fip(small_sample_ace)
    assert 1.50 < shrunk < constants.league_avg_era


def test_shrunk_wrc_plus_used_when_projecting_a_full_game() -> None:
    """Ponta a ponta: uma ofensa com wRC+ altíssimo mas amostra pequena deve projetar bem
    menos corridas do que a mesma ofensa com amostra grande (não deve ser tratada como
    igualmente confiável)."""
    constants = LeagueConstants()
    engine = ProjectionEngine(constants=constants)
    average_pitcher = OpposingPitcherInput(starter_fip=constants.league_avg_era)

    small_sample = TeamOffenseInput(wrc_plus=180.0, plate_appearances=15)
    large_sample = TeamOffenseInput(wrc_plus=180.0, plate_appearances=5000)

    projection_small = engine.project_game("Home", "Away", small_sample, small_sample, average_pitcher, average_pitcher)
    projection_large = engine.project_game("Home", "Away", large_sample, large_sample, average_pitcher, average_pitcher)

    assert projection_small.projected_home_runs < projection_large.projected_home_runs
