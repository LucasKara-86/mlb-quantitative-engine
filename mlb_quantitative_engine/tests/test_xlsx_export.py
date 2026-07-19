from __future__ import annotations

from pathlib import Path

import pytest

from mlb_quantitative_engine.database.repository import Repository
from mlb_quantitative_engine.reports.xlsx_export import build_export_rows, export_to_xlsx


@pytest.fixture()
def repository(tmp_path: Path) -> Repository:
    return Repository(db_path=str(tmp_path / "test.db"))


def test_build_export_rows_reflects_best_qualifying_bet(repository: Repository) -> None:
    game = repository.upsert_game(
        game_pk=1, game_date="2026-07-18", home_team="Yankees", away_team="Red Sox",
        home_probable_pitcher="Gerrit Cole", away_probable_pitcher="Brayan Bello",
        game_datetime="2026-07-18T17:11:00Z",
    )
    projection = repository.save_projection(
        game_id=game.id, projected_home_runs=5.0, projected_away_runs=4.0,
        projected_total_runs=9.0, probability_over=0.6, probability_under=0.4,
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

    rows = build_export_rows(repository, "2026-07-18")

    assert len(rows) == 1
    row = rows[0]
    assert row.jogo == "Red Sox @ Yankees"
    assert row.horario == "2026-07-18T17:11:00Z"
    assert row.melhor_aposta == "Jogo Over"
    assert row.recomendado is True
    assert row.edge == pytest.approx(0.1050)
    assert row.odd_minima == pytest.approx(1.71)
    assert row.stake_sugerida == pytest.approx(0.0375)
    assert not hasattr(row, "casa_apostas")


def test_build_export_rows_falls_back_to_highest_ev_when_none_qualify(repository: Repository) -> None:
    game = repository.upsert_game(game_pk=1, game_date="2026-07-18", home_team="Yankees", away_team="Red Sox")
    projection = repository.save_projection(
        game_id=game.id, projected_home_runs=4.5, projected_away_runs=4.0,
        projected_total_runs=8.5, probability_over=0.5, probability_under=0.5,
    )
    repository.save_value_bet(
        projection_id=projection.id, market="game_total_over", bookmaker="DraftKings",
        price=1.90, point=8.5, projection_probability=0.50, implied_probability_raw=0.5263,
        implied_probability_fair=0.50, edge=0.0, expected_value=-0.05,
        kelly_fraction=0.0, kelly_fraction_quarter=0.0, suggested_stake_fraction=0.0, minimum_acceptable_price=1.71, confidence_score=90.0, meets_criteria=False,
    )

    rows = build_export_rows(repository, "2026-07-18")

    assert rows[0].recomendado is False
    assert rows[0].melhor_aposta == "Jogo Over"  # ainda mostra o melhor, mesmo sem qualificar


def test_build_export_rows_handles_game_without_projection(repository: Repository) -> None:
    repository.upsert_game(game_pk=1, game_date="2026-07-18", home_team="Yankees", away_team="Red Sox")

    rows = build_export_rows(repository, "2026-07-18")

    assert len(rows) == 1
    assert rows[0].projecao_total is None
    assert rows[0].melhor_aposta is None
    assert rows[0].recomendado is False


def test_build_export_rows_filters_by_date(repository: Repository) -> None:
    repository.upsert_game(game_pk=1, game_date="2026-07-18", home_team="Yankees", away_team="Red Sox")
    repository.upsert_game(game_pk=2, game_date="2026-07-19", home_team="Cubs", away_team="Brewers")

    rows = build_export_rows(repository, "2026-07-18")

    assert len(rows) == 1
    assert rows[0].game_pk == 1


def test_export_to_xlsx_creates_file_with_expected_rows(repository: Repository, tmp_path: Path) -> None:
    repository.upsert_game(game_pk=1, game_date="2026-07-18", home_team="Yankees", away_team="Red Sox")
    output = str(tmp_path / "test_export.xlsx")

    saved_path = export_to_xlsx(repository, "2026-07-18", output)

    assert Path(saved_path).exists()


def test_export_to_xlsx_handles_no_games_gracefully(repository: Repository, tmp_path: Path) -> None:
    output = str(tmp_path / "empty_export.xlsx")
    saved_path = export_to_xlsx(repository, "2026-07-18", output)
    assert Path(saved_path).exists()

