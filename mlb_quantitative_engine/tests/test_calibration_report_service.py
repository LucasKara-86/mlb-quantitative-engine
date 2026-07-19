from __future__ import annotations

from pathlib import Path

import pytest

from mlb_quantitative_engine.database.repository import Repository
from mlb_quantitative_engine.services.calibration_report_service import (
    build_calibration_report,
    format_calibration_report,
)


@pytest.fixture()
def repository(tmp_path: Path) -> Repository:
    return Repository(db_path=str(tmp_path / "calibration.db"))


def _seed_resolved_bet(
    repository: Repository, game_pk: int, projection_probability: float, outcome: str
) -> None:
    game = repository.upsert_game(game_pk=game_pk, game_date="2026-07-17", home_team="A", away_team="B")
    projection = repository.save_projection(
        game_id=game.id, projected_home_runs=5.0, projected_away_runs=4.0,
        projected_total_runs=9.0, probability_over=0.6, probability_under=0.4,
    )
    bet = repository.save_value_bet(
        projection_id=projection.id, market="game_total_over", bookmaker="DraftKings",
        price=1.95, point=8.5, projection_probability=projection_probability,
        implied_probability_raw=0.5, implied_probability_fair=0.5, edge=0.1, expected_value=0.15,
        kelly_fraction=0.1, kelly_fraction_quarter=0.025, suggested_stake_fraction=0.02,
        minimum_acceptable_price=1.7, confidence_score=90.0, meets_criteria=True,
    )
    repository.mark_alert_sent(bet.id)
    repository.mark_bet_outcome(bet.id, outcome)


def test_report_is_empty_when_no_bets_resolved(repository: Repository) -> None:
    report = build_calibration_report(repository)
    assert report.total_resolved == 0
    assert report.brier_score is None
    assert report.overall_hit_rate is None
    assert report.reliability == []


def test_report_counts_wins_losses_and_pushes(repository: Repository) -> None:
    _seed_resolved_bet(repository, game_pk=1, projection_probability=0.75, outcome="win")
    _seed_resolved_bet(repository, game_pk=2, projection_probability=0.75, outcome="loss")
    _seed_resolved_bet(repository, game_pk=3, projection_probability=0.75, outcome="push")

    report = build_calibration_report(repository)

    assert report.total_resolved == 3
    assert report.total_wins == 1
    assert report.total_losses == 1
    assert report.total_pushes == 1


def test_report_hit_rate_ignores_pushes(repository: Repository) -> None:
    _seed_resolved_bet(repository, game_pk=1, projection_probability=0.75, outcome="win")
    _seed_resolved_bet(repository, game_pk=2, projection_probability=0.75, outcome="push")

    report = build_calibration_report(repository)

    assert report.overall_hit_rate == pytest.approx(1.0)  # só a "win" conta, o push é ignorado


def test_report_builds_reliability_buckets(repository: Repository) -> None:
    _seed_resolved_bet(repository, game_pk=1, projection_probability=0.72, outcome="win")
    _seed_resolved_bet(repository, game_pk=2, projection_probability=0.73, outcome="loss")

    report = build_calibration_report(repository)

    assert len(report.reliability) == 1
    assert report.reliability[0].count == 2
    assert report.reliability[0].actual_win_rate == pytest.approx(0.5)


def test_format_calibration_report_handles_empty_report(repository: Repository) -> None:
    report = build_calibration_report(repository)
    text = format_calibration_report(report)
    assert "0" in text
    assert "nada para calibrar" in text.lower()


def test_format_calibration_report_renders_table_without_crashing(repository: Repository) -> None:
    _seed_resolved_bet(repository, game_pk=1, projection_probability=0.72, outcome="win")
    _seed_resolved_bet(repository, game_pk=2, projection_probability=0.73, outcome="loss")

    report = build_calibration_report(repository)
    text = format_calibration_report(report)

    assert "Brier Score" in text
    assert "70%" in text
