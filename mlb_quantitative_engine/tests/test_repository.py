from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from mlb_quantitative_engine.database.repository import Repository


@pytest.fixture()
def repository(tmp_path: Path) -> Repository:
    return Repository(db_path=str(tmp_path / "test.db"))


def test_upsert_game_creates_new_game(repository: Repository) -> None:
    game = repository.upsert_game(
        game_pk=123456,
        game_date="2026-07-17",
        home_team="Yankees",
        away_team="Red Sox",
        venue="Yankee Stadium",
        status="Scheduled",
        home_probable_pitcher="Gerrit Cole",
        away_probable_pitcher="Brayan Bello",
    )

    assert game.id is not None
    assert game.game_pk == 123456
    assert game.home_team == "Yankees"


def test_upsert_game_updates_existing_game(repository: Repository) -> None:
    repository.upsert_game(game_pk=1, game_date="2026-07-17", home_team="Mets", away_team="Braves")
    updated = repository.upsert_game(
        game_pk=1, game_date="2026-07-17", home_team="Mets", away_team="Braves", status="Final"
    )

    all_games = repository.list_games_by_date("2026-07-17")
    assert len(all_games) == 1
    assert updated.status == "Final"


def test_upsert_game_stores_game_datetime(repository: Repository) -> None:
    game = repository.upsert_game(
        game_pk=1, game_date="2026-07-18", home_team="Guardians", away_team="Pirates",
        game_datetime="2026-07-18T17:11:00Z",
    )
    assert game.game_datetime == "2026-07-18T17:11:00Z"


def test_list_games_by_date_filters_correctly(repository: Repository) -> None:
    repository.upsert_game(game_pk=1, game_date="2026-07-17", home_team="Cubs", away_team="Brewers")
    repository.upsert_game(game_pk=2, game_date="2026-07-18", home_team="Giants", away_team="Padres")

    games_today = repository.list_games_by_date("2026-07-17")
    assert len(games_today) == 1
    assert games_today[0].home_team == "Cubs"


def test_save_projection_links_to_game(repository: Repository) -> None:
    game = repository.upsert_game(game_pk=1, game_date="2026-07-17", home_team="Astros", away_team="Rangers")
    projection = repository.save_projection(
        game_id=game.id,
        projected_home_runs=4.8,
        projected_away_runs=4.1,
        projected_total_runs=8.9,
        probability_over=0.55,
        probability_under=0.45,
        model_version="v0-hybrid",
    )

    assert projection.id is not None
    assert projection.game_id == game.id

    projections = repository.list_projections_for_game(game.id)
    assert len(projections) == 1
    assert projections[0].model_version == "v0-hybrid"


def test_save_value_bet_links_to_projection(repository: Repository) -> None:
    game = repository.upsert_game(game_pk=1, game_date="2026-07-17", home_team="Dodgers", away_team="Giants")
    projection = repository.save_projection(
        game_id=game.id,
        projected_home_runs=5.0,
        projected_away_runs=4.0,
        projected_total_runs=9.0,
        probability_over=0.60,
        probability_under=0.40,
    )
    value_bet = repository.save_value_bet(
        projection_id=projection.id,
        market="game_total_over",
        bookmaker="DraftKings",
        price=1.95,
        point=8.5,
        projection_probability=0.60,
        implied_probability_raw=0.5128,
        implied_probability_fair=0.4950,
        edge=0.1050,
        expected_value=0.17,
        kelly_fraction=0.15,
        kelly_fraction_quarter=0.0375, suggested_stake_fraction=0.0375, minimum_acceptable_price=1.71,
        confidence_score=90.0,
        meets_criteria=True,
    )

    assert value_bet.id is not None
    all_bets = repository.list_value_bets()
    assert len(all_bets) == 1
    assert all_bets[0].bookmaker == "DraftKings"


def test_list_value_bets_can_filter_by_meets_criteria(repository: Repository) -> None:
    game = repository.upsert_game(game_pk=1, game_date="2026-07-17", home_team="Dodgers", away_team="Giants")
    projection = repository.save_projection(
        game_id=game.id, projected_home_runs=5.0, projected_away_runs=4.0,
        projected_total_runs=9.0, probability_over=0.60, probability_under=0.40,
    )
    repository.save_value_bet(
        projection_id=projection.id, market="game_total_over", bookmaker="DraftKings",
        price=1.95, point=8.5, projection_probability=0.60, implied_probability_raw=0.5128,
        implied_probability_fair=0.4950, edge=0.1050, expected_value=0.17,
        kelly_fraction=0.15, kelly_fraction_quarter=0.0375, suggested_stake_fraction=0.0375, minimum_acceptable_price=1.71, confidence_score=90.0, meets_criteria=True,
    )
    repository.save_value_bet(
        projection_id=projection.id, market="game_total_under", bookmaker="FanDuel",
        price=1.85, point=8.5, projection_probability=0.40, implied_probability_raw=0.4878,
        implied_probability_fair=0.5050, edge=-0.1050, expected_value=-0.26,
        kelly_fraction=0.0, kelly_fraction_quarter=0.0, suggested_stake_fraction=0.0, minimum_acceptable_price=1.71, confidence_score=90.0, meets_criteria=False,
    )

    assert len(repository.list_value_bets()) == 2
    qualifying = repository.list_value_bets(meets_criteria_only=True)
    assert len(qualifying) == 1
    assert qualifying[0].market == "game_total_over"


def test_list_value_bets_for_projection_filters_by_projection_id(repository: Repository) -> None:
    game = repository.upsert_game(game_pk=1, game_date="2026-07-18", home_team="Dodgers", away_team="Giants")
    projection_a = repository.save_projection(
        game_id=game.id, projected_home_runs=5.0, projected_away_runs=4.0,
        projected_total_runs=9.0, probability_over=0.6, probability_under=0.4,
    )
    projection_b = repository.save_projection(
        game_id=game.id, projected_home_runs=5.5, projected_away_runs=4.2,
        projected_total_runs=9.7, probability_over=0.62, probability_under=0.38,
    )
    repository.save_value_bet(
        projection_id=projection_a.id, market="game_total_over", bookmaker="DraftKings",
        price=1.95, point=8.5, projection_probability=0.60, implied_probability_raw=0.5128,
        implied_probability_fair=0.4950, edge=0.1050, expected_value=0.17,
        kelly_fraction=0.15, kelly_fraction_quarter=0.0375, suggested_stake_fraction=0.0375, minimum_acceptable_price=1.71, confidence_score=90.0, meets_criteria=True,
    )
    repository.save_value_bet(
        projection_id=projection_b.id, market="game_total_under", bookmaker="FanDuel",
        price=1.85, point=9.5, projection_probability=0.40, implied_probability_raw=0.4878,
        implied_probability_fair=0.5050, edge=-0.1050, expected_value=-0.26,
        kelly_fraction=0.0, kelly_fraction_quarter=0.0, suggested_stake_fraction=0.0, minimum_acceptable_price=1.71, confidence_score=90.0, meets_criteria=False,
    )

    bets_for_a = repository.list_value_bets_for_projection(projection_a.id)
    assert len(bets_for_a) == 1
    assert bets_for_a[0].market == "game_total_over"


def _seed_value_bet(repository: Repository, game_pk: int = 1, market: str = "game_total_over") -> int:
    game = repository.upsert_game(game_pk=game_pk, game_date="2026-07-17", home_team="Dodgers", away_team="Giants")
    projection = repository.save_projection(
        game_id=game.id, projected_home_runs=5.0, projected_away_runs=4.0,
        projected_total_runs=9.0, probability_over=0.60, probability_under=0.40,
    )
    bet = repository.save_value_bet(
        projection_id=projection.id, market=market, bookmaker="DraftKings",
        price=1.95, point=8.5, projection_probability=0.60, implied_probability_raw=0.5128,
        implied_probability_fair=0.4950, edge=0.1050, expected_value=0.17,
        kelly_fraction=0.15, kelly_fraction_quarter=0.0375, suggested_stake_fraction=0.0375,
        minimum_acceptable_price=1.71, confidence_score=90.0, meets_criteria=True,
    )
    return bet.id


def test_new_value_bet_defaults_alert_sent_and_result_notified_to_false(repository: Repository) -> None:
    bet_id = _seed_value_bet(repository)
    assert repository.list_bets_pending_result_check() == []

    repository.mark_alert_sent(bet_id)
    pending = repository.list_bets_pending_result_check()
    assert len(pending) == 1
    bet, game = pending[0]
    assert bet.id == bet_id
    assert game.home_team == "Dodgers"


def test_mark_result_notified_removes_bet_from_pending_list(repository: Repository) -> None:
    bet_id = _seed_value_bet(repository)
    repository.mark_alert_sent(bet_id)
    assert len(repository.list_bets_pending_result_check()) == 1

    repository.mark_result_notified(bet_id)
    assert repository.list_bets_pending_result_check() == []


def test_bets_without_alert_sent_never_appear_as_pending(repository: Repository) -> None:
    _seed_value_bet(repository)  # alert_sent nunca marcado
    assert repository.list_bets_pending_result_check() == []


def test_new_value_bet_has_no_outcome_by_default(repository: Repository) -> None:
    bet_id = _seed_value_bet(repository)
    assert repository.list_resolved_value_bets() == []
    repository.mark_alert_sent(bet_id)
    assert repository.list_resolved_value_bets() == []  # alert_sent sozinho não é "resolvido"


def test_has_alert_been_sent_is_false_by_default(repository: Repository) -> None:
    _seed_value_bet(repository, game_pk=1, market="game_total_over")
    assert repository.has_alert_been_sent(1, "game_total_over") is False


def test_has_alert_been_sent_is_true_after_marking_alert_sent(repository: Repository) -> None:
    bet_id = _seed_value_bet(repository, game_pk=1, market="game_total_over")
    repository.mark_alert_sent(bet_id)
    assert repository.has_alert_been_sent(1, "game_total_over") is True


def test_has_alert_been_sent_is_market_specific(repository: Repository) -> None:
    bet_id = _seed_value_bet(repository, game_pk=1, market="game_total_over")
    repository.mark_alert_sent(bet_id)
    assert repository.has_alert_been_sent(1, "game_total_under") is False


def test_has_alert_been_sent_is_game_specific(repository: Repository) -> None:
    bet_id = _seed_value_bet(repository, game_pk=1, market="game_total_over")
    repository.mark_alert_sent(bet_id)
    assert repository.has_alert_been_sent(2, "game_total_over") is False


def test_has_any_alert_been_sent_for_game_is_false_by_default(repository: Repository) -> None:
    _seed_value_bet(repository, game_pk=1, market="game_total_over")
    assert repository.has_any_alert_been_sent_for_game(1) is False


def test_has_any_alert_been_sent_for_game_is_true_regardless_of_market(repository: Repository) -> None:
    bet_id = _seed_value_bet(repository, game_pk=1, market="game_total_over")
    repository.mark_alert_sent(bet_id)
    # qualquer mercado do jogo já enviado -> True (dedup por jogo, não por mercado)
    assert repository.has_any_alert_been_sent_for_game(1) is True


def test_has_any_alert_been_sent_for_game_is_game_specific(repository: Repository) -> None:
    bet_id = _seed_value_bet(repository, game_pk=1, market="game_total_over")
    repository.mark_alert_sent(bet_id)
    assert repository.has_any_alert_been_sent_for_game(2) is False


def test_mark_bet_outcome_sets_outcome_and_result_notified(repository: Repository) -> None:
    bet_id = _seed_value_bet(repository)
    repository.mark_alert_sent(bet_id)
    repository.mark_bet_outcome(bet_id, "win")

    resolved = repository.list_resolved_value_bets()
    assert len(resolved) == 1
    assert resolved[0].outcome == "win"
    assert repository.list_bets_pending_result_check() == []  # outcome marcado -> resolvido, some da fila


def test_batch_is_not_processed_by_default(repository: Repository) -> None:
    anchor = datetime(2026, 7, 18, 14, 10, tzinfo=timezone.utc)
    assert repository.is_batch_processed("2026-07-18", anchor) is False


def test_mark_batch_processed_then_is_batch_processed_returns_true(repository: Repository) -> None:
    anchor = datetime(2026, 7, 18, 14, 10, tzinfo=timezone.utc)
    repository.mark_batch_processed("2026-07-18", anchor)
    assert repository.is_batch_processed("2026-07-18", anchor) is True


def test_different_anchor_times_are_tracked_independently(repository: Repository) -> None:
    anchor1 = datetime(2026, 7, 18, 14, 10, tzinfo=timezone.utc)
    anchor2 = datetime(2026, 7, 18, 17, 10, tzinfo=timezone.utc)
    repository.mark_batch_processed("2026-07-18", anchor1)

    assert repository.is_batch_processed("2026-07-18", anchor1) is True
    assert repository.is_batch_processed("2026-07-18", anchor2) is False


def test_repository_creates_database_file(tmp_path: Path) -> None:
    db_file = tmp_path / "nested" / "database.db"
    Repository(db_path=str(db_file))
    assert db_file.exists()


def test_list_due_lineup_retries_is_empty_by_default(repository: Repository) -> None:
    now = datetime(2026, 7, 18, 14, 0, tzinfo=timezone.utc)
    assert repository.list_due_lineup_retries(now) == []


def test_upsert_pending_lineup_retry_appears_when_due(repository: Repository) -> None:
    retry_at = datetime(2026, 7, 18, 13, 50, tzinfo=timezone.utc)
    repository.upsert_pending_lineup_retry(game_pk=1, game_date="2026-07-18", retry_at=retry_at)

    before = datetime(2026, 7, 18, 13, 40, tzinfo=timezone.utc)
    after = datetime(2026, 7, 18, 14, 0, tzinfo=timezone.utc)

    assert repository.list_due_lineup_retries(before) == []
    due = repository.list_due_lineup_retries(after)
    assert len(due) == 1
    assert due[0].game_pk == 1


def test_upsert_pending_lineup_retry_reschedules_existing_unresolved_entry(repository: Repository) -> None:
    """Chamar de novo para o mesmo jogo (ainda não resolvido) deve atualizar o horário,
    não criar uma segunda retentativa."""
    repository.upsert_pending_lineup_retry(
        game_pk=1, game_date="2026-07-18", retry_at=datetime(2026, 7, 18, 13, 50, tzinfo=timezone.utc)
    )
    repository.upsert_pending_lineup_retry(
        game_pk=1, game_date="2026-07-18", retry_at=datetime(2026, 7, 18, 14, 20, tzinfo=timezone.utc)
    )

    due = repository.list_due_lineup_retries(datetime(2026, 7, 18, 14, 30, tzinfo=timezone.utc))
    assert len(due) == 1
    assert due[0].retry_at.replace(tzinfo=timezone.utc) == datetime(2026, 7, 18, 14, 20, tzinfo=timezone.utc)


def test_mark_lineup_retry_resolved_removes_it_from_due_list(repository: Repository) -> None:
    repository.upsert_pending_lineup_retry(
        game_pk=1, game_date="2026-07-18", retry_at=datetime(2026, 7, 18, 13, 50, tzinfo=timezone.utc)
    )
    repository.mark_lineup_retry_resolved(game_pk=1)

    due = repository.list_due_lineup_retries(datetime(2026, 7, 18, 14, 0, tzinfo=timezone.utc))
    assert due == []


def test_resolved_retry_can_be_rescheduled_as_a_new_entry(repository: Repository) -> None:
    """Depois de resolvida, um novo upsert para o mesmo jogo deve criar uma nova
    retentativa (não reaproveitar a resolvida)."""
    repository.upsert_pending_lineup_retry(
        game_pk=1, game_date="2026-07-18", retry_at=datetime(2026, 7, 18, 13, 50, tzinfo=timezone.utc)
    )
    repository.mark_lineup_retry_resolved(game_pk=1)
    repository.upsert_pending_lineup_retry(
        game_pk=1, game_date="2026-07-18", retry_at=datetime(2026, 7, 18, 15, 0, tzinfo=timezone.utc)
    )

    due = repository.list_due_lineup_retries(datetime(2026, 7, 18, 15, 30, tzinfo=timezone.utc))
    assert len(due) == 1
    assert due[0].retry_at.replace(tzinfo=timezone.utc) == datetime(2026, 7, 18, 15, 0, tzinfo=timezone.utc)

