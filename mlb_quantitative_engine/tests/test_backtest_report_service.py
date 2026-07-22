from __future__ import annotations

from pathlib import Path

import pytest

from mlb_quantitative_engine.database.repository import Repository
from mlb_quantitative_engine.services.backtest_report_service import (
    build_backtest_report,
    format_backtest_report,
)


@pytest.fixture()
def repository(tmp_path: Path) -> Repository:
    return Repository(db_path=str(tmp_path / "backtest.db"))


def _seed_resolved_bet(
    repository: Repository,
    game_pk: int,
    market: str,
    price: float,
    stake: float,
    confidence_score: float,
    outcome: str,
    game_date: str = "2026-07-21",
) -> None:
    game = repository.upsert_game(game_pk=game_pk, game_date=game_date, home_team="A", away_team="B")
    projection = repository.save_projection(
        game_id=game.id, projected_home_runs=5.0, projected_away_runs=4.0,
        projected_total_runs=9.0, probability_over=0.6, probability_under=0.4,
    )
    bet = repository.save_value_bet(
        projection_id=projection.id, market=market, bookmaker="DraftKings",
        price=price, point=8.5, projection_probability=0.6,
        implied_probability_raw=0.5, implied_probability_fair=0.5, edge=0.1, expected_value=0.15,
        kelly_fraction=0.1, kelly_fraction_quarter=0.025, suggested_stake_fraction=stake,
        minimum_acceptable_price=price * 0.95, confidence_score=confidence_score, meets_criteria=True,
    )
    repository.mark_alert_sent(bet.id)
    repository.mark_bet_outcome(bet.id, outcome)


def test_report_is_empty_when_no_bets_resolved(repository: Repository) -> None:
    report = build_backtest_report(repository)
    assert report.overall.total_bets == 0
    assert report.by_market == {}
    assert report.by_confidence_bucket == {}


def test_report_computes_overall_metrics_from_resolved_bets(repository: Repository) -> None:
    _seed_resolved_bet(repository, game_pk=1, market="game_total_over", price=2.0, stake=0.02,
                        confidence_score=90.0, outcome="win")
    _seed_resolved_bet(repository, game_pk=2, market="game_total_under", price=2.0, stake=0.02,
                        confidence_score=90.0, outcome="loss")

    report = build_backtest_report(repository)

    assert report.overall.total_bets == 2
    assert report.overall.wins == 1
    assert report.overall.losses == 1
    assert report.overall.hit_rate == pytest.approx(0.5)


def test_report_segments_by_market_family_ignoring_over_under(repository: Repository) -> None:
    _seed_resolved_bet(repository, game_pk=1, market="home_team_total_over", price=2.0, stake=0.02,
                        confidence_score=90.0, outcome="win")
    _seed_resolved_bet(repository, game_pk=2, market="home_team_total_under", price=2.0, stake=0.02,
                        confidence_score=90.0, outcome="win")
    _seed_resolved_bet(repository, game_pk=3, market="game_total_over", price=2.0, stake=0.02,
                        confidence_score=90.0, outcome="loss")

    report = build_backtest_report(repository)

    assert report.by_market["home_team_total"].total_bets == 2
    assert report.by_market["game_total"].total_bets == 1


def test_report_segments_by_confidence_bucket(repository: Repository) -> None:
    _seed_resolved_bet(repository, game_pk=1, market="game_total_over", price=2.0, stake=0.02,
                        confidence_score=72.0, outcome="win")
    _seed_resolved_bet(repository, game_pk=2, market="game_total_over", price=2.0, stake=0.02,
                        confidence_score=91.0, outcome="loss")

    report = build_backtest_report(repository)

    assert report.by_confidence_bucket["70-75"].total_bets == 1
    assert report.by_confidence_bucket["90-95"].total_bets == 1


def test_report_only_covers_a_specific_date_when_rows_are_scoped(repository: Repository) -> None:
    _seed_resolved_bet(repository, game_pk=1, market="game_total_over", price=2.0, stake=0.02,
                        confidence_score=90.0, outcome="win", game_date="2026-07-20")
    _seed_resolved_bet(repository, game_pk=2, market="game_total_over", price=2.0, stake=0.02,
                        confidence_score=90.0, outcome="loss", game_date="2026-07-21")

    yesterday_rows = repository.list_resolved_value_bets_for_date("2026-07-21")
    # list_resolved_value_bets_for_date devolve só ValueBet; monta os pares manualmente
    # com o Game correspondente, do mesmo jeito que reports/daily_analysis_runner.py fará.
    all_rows = repository.list_resolved_value_bets_with_game()
    scoped_rows = [(vb, game) for vb, game in all_rows if game.game_date == "2026-07-21"]

    assert len(yesterday_rows) == 1
    report = build_backtest_report(resolved_rows=scoped_rows)
    assert report.overall.total_bets == 1
    assert report.overall.losses == 1


def test_format_backtest_report_handles_empty_report(repository: Repository) -> None:
    report = build_backtest_report(repository)
    text = format_backtest_report(report)
    assert "nada para calcular" in text.lower()


def test_format_backtest_report_renders_without_crashing(repository: Repository) -> None:
    _seed_resolved_bet(repository, game_pk=1, market="game_total_over", price=2.0, stake=0.02,
                        confidence_score=90.0, outcome="win")

    report = build_backtest_report(repository)
    text = format_backtest_report(report)

    assert "Hit rate" in text
    assert "ROI" in text
