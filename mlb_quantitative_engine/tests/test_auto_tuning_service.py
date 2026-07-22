from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlb_quantitative_engine.database.repository import Repository
from mlb_quantitative_engine.services.auto_tuning_service import run_auto_tuning


@pytest.fixture()
def repository(tmp_path: Path) -> Repository:
    return Repository(db_path=str(tmp_path / "auto_tuning.db"))


def _seed_losing_bets(repository: Repository, count: int) -> None:
    """Amostra deliberadamente abaixo de 60% de probabilidade projetada, para acionar só a
    regra de ROI geral (R4) e manter o teste do gate de segurança isolado da regra de
    calibração (R1/R2, já coberta em tests/test_auto_tuning.py)."""
    for i in range(count):
        game = repository.upsert_game(game_pk=i, game_date="2026-07-21", home_team="A", away_team="B")
        projection = repository.save_projection(
            game_id=game.id, projected_home_runs=5.0, projected_away_runs=4.0,
            projected_total_runs=9.0, probability_over=0.55, probability_under=0.45,
        )
        bet = repository.save_value_bet(
            projection_id=projection.id, market="game_total_over", bookmaker="DraftKings",
            price=1.90, point=8.5, projection_probability=0.55,
            implied_probability_raw=0.5, implied_probability_fair=0.5, edge=0.1, expected_value=0.15,
            kelly_fraction=0.1, kelly_fraction_quarter=0.025, suggested_stake_fraction=0.02,
            minimum_acceptable_price=1.8, confidence_score=90.0, meets_criteria=True,
        )
        repository.mark_alert_sent(bet.id)
        repository.mark_bet_outcome(bet.id, "loss")


def _fake_params_path(tmp_path: Path) -> Path:
    path = tmp_path / "tunable_params.json"
    path.write_text(
        json.dumps({
            "model_version": "v1", "min_edge": 0.04, "min_confidence": 70.0,
            "overdispersion": 1.4, "mean_uncertainty_pct": 0.12,
        }),
        encoding="utf-8",
    )
    return path


def test_no_changes_when_sample_insufficient(repository: Repository, tmp_path: Path) -> None:
    result = run_auto_tuning(
        repository=repository, run_tests_fn=lambda: True, git_is_clean_fn=lambda: True,
        commit_fn=lambda adjustment: "deadbeef", tunable_params_path=_fake_params_path(tmp_path),
    )
    assert result.changes == []
    assert result.skipped_reason is None


def test_applies_change_when_tests_pass(repository: Repository, tmp_path: Path) -> None:
    _seed_losing_bets(repository, 25)
    params_path = _fake_params_path(tmp_path)

    result = run_auto_tuning(
        repository=repository, run_tests_fn=lambda: True, git_is_clean_fn=lambda: True,
        commit_fn=lambda adjustment: "deadbeef", tunable_params_path=params_path,
    )

    assert len(result.changes) == 1
    change = result.changes[0]
    assert change.applied is True
    assert change.git_commit_sha == "deadbeef"
    assert change.adjustment.parameter_name == "min_edge"
    assert change.adjustment.new_value > change.adjustment.old_value

    written = json.loads(params_path.read_text(encoding="utf-8"))
    assert written["min_edge"] == pytest.approx(change.adjustment.new_value)
    assert written["model_version"] == "v2"

    logged = repository.last_applied_parameter_change("min_edge")
    assert logged is not None
    assert logged.applied is True
    assert logged.git_commit_sha == "deadbeef"


def test_reverts_when_tests_fail(repository: Repository, tmp_path: Path) -> None:
    _seed_losing_bets(repository, 25)
    params_path = _fake_params_path(tmp_path)
    original = json.loads(params_path.read_text(encoding="utf-8"))
    commit_calls = []

    result = run_auto_tuning(
        repository=repository, run_tests_fn=lambda: False, git_is_clean_fn=lambda: True,
        commit_fn=lambda adjustment: commit_calls.append(adjustment) or "should-not-happen",
        tunable_params_path=params_path,
    )

    assert len(result.changes) == 1
    change = result.changes[0]
    assert change.applied is False
    assert change.git_commit_sha is None
    assert commit_calls == []  # gate nunca chama commit_fn quando os testes falham
    assert json.loads(params_path.read_text(encoding="utf-8")) == original  # valor anterior restaurado

    logged = repository.last_applied_parameter_change("min_edge")
    assert logged is None  # só entra como "aplicado" quando applied=True


def test_skips_everything_when_git_tree_is_dirty(repository: Repository, tmp_path: Path) -> None:
    _seed_losing_bets(repository, 25)
    params_path = _fake_params_path(tmp_path)
    original = params_path.read_text(encoding="utf-8")

    result = run_auto_tuning(
        repository=repository, run_tests_fn=lambda: True, git_is_clean_fn=lambda: False,
        commit_fn=lambda adjustment: "deadbeef", tunable_params_path=params_path,
    )

    assert result.changes == []
    assert result.skipped_reason is not None
    assert params_path.read_text(encoding="utf-8") == original


def test_cooldown_prevents_reapplying_recently_changed_parameter(repository: Repository, tmp_path: Path) -> None:
    _seed_losing_bets(repository, 25)
    repository.record_parameter_change(
        parameter_name="min_edge", old_value=0.035, new_value=0.04, rationale="ajuste anterior",
        sample_size=25, applied=True, git_commit_sha="prev-sha",
    )

    result = run_auto_tuning(
        repository=repository, run_tests_fn=lambda: True, git_is_clean_fn=lambda: True,
        commit_fn=lambda adjustment: "deadbeef", tunable_params_path=_fake_params_path(tmp_path),
    )

    assert result.changes == []
    assert len(result.deferred) == 1
    assert result.deferred[0].parameter_name == "min_edge"


def test_negative_roi_findings_are_reported_even_without_auto_action(repository: Repository, tmp_path: Path) -> None:
    for i in range(25):
        game = repository.upsert_game(game_pk=100 + i, game_date="2026-07-21", home_team="A", away_team="B")
        projection = repository.save_projection(
            game_id=game.id, projected_home_runs=5.0, projected_away_runs=4.0,
            projected_total_runs=9.0, probability_over=0.55, probability_under=0.45,
        )
        bet = repository.save_value_bet(
            projection_id=projection.id, market="home_team_total_under", bookmaker="DraftKings",
            price=1.90, point=4.5, projection_probability=0.55,
            implied_probability_raw=0.5, implied_probability_fair=0.5, edge=0.1, expected_value=0.15,
            kelly_fraction=0.1, kelly_fraction_quarter=0.025, suggested_stake_fraction=0.02,
            minimum_acceptable_price=1.8, confidence_score=90.0, meets_criteria=True,
        )
        repository.mark_alert_sent(bet.id)
        repository.mark_bet_outcome(bet.id, "loss")

    result = run_auto_tuning(
        repository=repository, run_tests_fn=lambda: True, git_is_clean_fn=lambda: True,
        commit_fn=lambda adjustment: "deadbeef", tunable_params_path=_fake_params_path(tmp_path),
    )

    assert any("home_team_total" in finding for finding in result.negative_roi_findings)
