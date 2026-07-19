from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pytest

from mlb_quantitative_engine.api.mlb_api import GameSummary
from mlb_quantitative_engine.database.repository import Repository
from mlb_quantitative_engine.reports.incremental_runner import run_due_batches


def _t(hour: int, minute: int) -> datetime:
    return datetime(2026, 7, 18, hour, minute, tzinfo=timezone.utc)


def _game(game_pk: int, hour: int, minute: int) -> GameSummary:
    return GameSummary(
        game_pk=game_pk,
        game_date="2026-07-18",
        game_datetime=_t(hour, minute).isoformat().replace("+00:00", "Z"),
        home_team=f"Home{game_pk}",
        away_team=f"Away{game_pk}",
        venue="Some Park",
        status="Scheduled",
        home_probable_pitcher="Pitcher A",
        away_probable_pitcher="Pitcher B",
        home_probable_pitcher_id=100 + game_pk,
        away_probable_pitcher_id=200 + game_pk,
        home_team_id=1000 + game_pk,
        away_team_id=2000 + game_pk,
    )


class _FakeApiClient:
    def __init__(self, games: List[GameSummary]) -> None:
        self.games = games

    def get_games_for_date(self, date: Optional[str] = None) -> List[GameSummary]:
        return self.games


class _FakeReportGenerator:
    """Registra quais jogos foram processados, sem fazer nenhuma chamada real."""

    def __init__(self) -> None:
        self.built_game_pks: List[int] = []
        self.fetch_all_odds_calls = 0

    def fetch_all_odds(self):
        self.fetch_all_odds_calls += 1
        return []

    def build_row(self, game: GameSummary, all_odds):
        self.built_game_pks.append(game.game_pk)


@pytest.fixture()
def repository(tmp_path: Path) -> Repository:
    return Repository(db_path=str(tmp_path / "test.db"))


def test_no_batches_due_processes_nothing(repository: Repository, tmp_path: Path) -> None:
    games = [_game(1, 14, 10), _game(2, 17, 10)]
    generator = _FakeReportGenerator()

    result = run_due_batches(
        date="2026-07-18",
        api_client=_FakeApiClient(games),
        repository=repository,
        report_generator=generator,
        xlsx_output_path=str(tmp_path / "out.xlsx"),
        now=_t(10, 0),  # bem antes de qualquer disparo (13:50 e 16:50)
    )

    assert result == []
    assert generator.built_game_pks == []


def test_due_batch_processes_only_its_games(repository: Repository, tmp_path: Path) -> None:
    """Cada jogo tem seu próprio gatilho (30 min antes do seu horário) -- só o jogo 1
    (gatilho 13:40) está devido às 14:00; o jogo 2 (gatilho 14:50) e o jogo 3
    (gatilho 16:40) ainda não."""
    games = [_game(1, 14, 10), _game(2, 15, 20), _game(3, 17, 10)]
    generator = _FakeReportGenerator()

    result = run_due_batches(
        date="2026-07-18",
        api_client=_FakeApiClient(games),
        repository=repository,
        report_generator=generator,
        xlsx_output_path=str(tmp_path / "out.xlsx"),
        now=_t(14, 0),
    )

    assert len(result) == 1
    assert generator.built_game_pks == [1]  # não inclui os jogos 2 e 3 (gatilhos futuros)
    assert repository.is_batch_processed("2026-07-18", _t(14, 10)) is True
    assert repository.is_batch_processed("2026-07-18", _t(15, 20)) is False
    assert repository.is_batch_processed("2026-07-18", _t(17, 10)) is False


def test_already_processed_batch_is_not_reprocessed(repository: Repository, tmp_path: Path) -> None:
    games = [_game(1, 14, 10)]
    generator = _FakeReportGenerator()
    xlsx_path = str(tmp_path / "out.xlsx")

    run_due_batches(
        date="2026-07-18", api_client=_FakeApiClient(games), repository=repository,
        report_generator=generator, xlsx_output_path=xlsx_path, now=_t(14, 0),
    )
    assert generator.built_game_pks == [1]

    # roda de novo, mais tarde -- lote já processado não deve reprocessar
    result = run_due_batches(
        date="2026-07-18", api_client=_FakeApiClient(games), repository=repository,
        report_generator=generator, xlsx_output_path=xlsx_path, now=_t(18, 0),
    )
    assert result == []
    assert generator.built_game_pks == [1]  # não duplicou


def test_xlsx_is_exported_after_processing_a_batch(repository: Repository, tmp_path: Path) -> None:
    games = [_game(1, 14, 10)]
    generator = _FakeReportGenerator()
    xlsx_path = str(tmp_path / "out.xlsx")

    run_due_batches(
        date="2026-07-18", api_client=_FakeApiClient(games), repository=repository,
        report_generator=generator, xlsx_output_path=xlsx_path, now=_t(14, 0),
    )

    assert Path(xlsx_path).exists()


def test_games_without_game_datetime_are_ignored_in_scheduling(repository: Repository, tmp_path: Path) -> None:
    game_no_time = _game(1, 14, 10)
    game_no_time = GameSummary(**{**game_no_time.__dict__, "game_datetime": None})
    generator = _FakeReportGenerator()

    result = run_due_batches(
        date="2026-07-18", api_client=_FakeApiClient([game_no_time]), repository=repository,
        report_generator=generator, xlsx_output_path=str(tmp_path / "out.xlsx"), now=_t(14, 0),
    )

    assert result == []
    assert generator.built_game_pks == []
