from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pytest

from mlb_quantitative_engine.api.mlb_api import GameSummary
from mlb_quantitative_engine.database.repository import Repository
from mlb_quantitative_engine.services.bet_result_checker import check_pending_bet_results
from mlb_quantitative_engine.services.game_result_service import GameResultService


def _game_summary(game_pk: int, status: str, game_date: str = "2026-07-17") -> GameSummary:
    return GameSummary(
        game_pk=game_pk, game_date=game_date, game_datetime=f"{game_date}T23:05:00Z",
        home_team="Dodgers", away_team="Giants", venue="Some Park", status=status,
        home_probable_pitcher="A", away_probable_pitcher="B",
        home_probable_pitcher_id=1, away_probable_pitcher_id=2,
        home_team_id=100, away_team_id=200,
    )


class _FakeApiClient:
    def __init__(self, games_by_date: Dict[str, List[GameSummary]]) -> None:
        self.games_by_date = games_by_date
        self.schedule_calls = 0

    def get_games_for_date(self, date: Optional[str] = None) -> List[GameSummary]:
        self.schedule_calls += 1
        return self.games_by_date.get(date, [])


class _FakeGameResultService:
    def __init__(self, boxscore_by_pk: Dict[int, tuple]) -> None:
        self.boxscore_by_pk = boxscore_by_pk

    def get_final_score(self, game_pk: int):
        from mlb_quantitative_engine.services.game_result_service import GameResult

        if game_pk not in self.boxscore_by_pk:
            return None
        home_runs, away_runs = self.boxscore_by_pk[game_pk]
        return GameResult(game_pk=game_pk, home_runs=home_runs, away_runs=away_runs)


class _FakeTelegramNotifier:
    def __init__(self, raise_error: bool = False) -> None:
        self.sent = []
        self.raise_error = raise_error

    def send_bet_result_alert(self, **kwargs) -> dict:
        if self.raise_error:
            raise RuntimeError("Telegram indisponível (simulado)")
        self.sent.append(kwargs)
        return {"ok": True}


@pytest.fixture()
def repository(tmp_path: Path) -> Repository:
    return Repository(db_path=str(tmp_path / "checker.db"))


def _seed_sent_bet(repository: Repository, game_pk: int, market: str, point: float) -> int:
    game = repository.upsert_game(game_pk=game_pk, game_date="2026-07-17", home_team="Dodgers", away_team="Giants")
    projection = repository.save_projection(
        game_id=game.id, projected_home_runs=5.0, projected_away_runs=4.0,
        projected_total_runs=9.0, probability_over=0.60, probability_under=0.40,
    )
    bet = repository.save_value_bet(
        projection_id=projection.id, market=market, bookmaker="DraftKings",
        price=1.95, point=point, projection_probability=0.60, implied_probability_raw=0.5128,
        implied_probability_fair=0.4950, edge=0.1050, expected_value=0.17,
        kelly_fraction=0.15, kelly_fraction_quarter=0.0375, suggested_stake_fraction=0.0375,
        minimum_acceptable_price=1.71, confidence_score=90.0, meets_criteria=True,
    )
    repository.mark_alert_sent(bet.id)
    return bet.id


def test_returns_zero_and_makes_no_calls_when_nothing_pending(repository: Repository) -> None:
    client = _FakeApiClient({})
    notified = check_pending_bet_results(
        api_client=client, repository=repository,
        game_result_service=_FakeGameResultService({}), telegram_notifier=_FakeTelegramNotifier(),
    )
    assert notified == 0
    assert client.schedule_calls == 0


def test_skips_bet_when_game_not_yet_final(repository: Repository) -> None:
    _seed_sent_bet(repository, game_pk=1, market="game_total_over", point=8.5)
    client = _FakeApiClient({"2026-07-17": [_game_summary(1, status="In Progress")]})
    notifier = _FakeTelegramNotifier()

    notified = check_pending_bet_results(
        api_client=client, repository=repository,
        game_result_service=_FakeGameResultService({1: (5, 4)}), telegram_notifier=notifier,
    )

    assert notified == 0
    assert notifier.sent == []
    assert len(repository.list_bets_pending_result_check()) == 1


def test_sends_green_when_game_total_over_hits(repository: Repository) -> None:
    _seed_sent_bet(repository, game_pk=1, market="game_total_over", point=8.5)
    client = _FakeApiClient({"2026-07-17": [_game_summary(1, status="Final")]})
    notifier = _FakeTelegramNotifier()

    notified = check_pending_bet_results(
        api_client=client, repository=repository,
        game_result_service=_FakeGameResultService({1: (6, 4)}), telegram_notifier=notifier,
    )

    assert notified == 1
    assert notifier.sent[0]["outcome_label"] == "GREEN"
    assert repository.list_bets_pending_result_check() == []
    resolved = repository.list_resolved_value_bets()
    assert len(resolved) == 1
    assert resolved[0].outcome == "win"


def test_sends_red_when_game_total_over_misses(repository: Repository) -> None:
    _seed_sent_bet(repository, game_pk=1, market="game_total_over", point=8.5)
    client = _FakeApiClient({"2026-07-17": [_game_summary(1, status="Final")]})
    notifier = _FakeTelegramNotifier()

    notified = check_pending_bet_results(
        api_client=client, repository=repository,
        game_result_service=_FakeGameResultService({1: (2, 1)}), telegram_notifier=notifier,
    )

    assert notified == 1
    assert notifier.sent[0]["outcome_label"] == "RED"
    assert repository.list_resolved_value_bets()[0].outcome == "loss"


def test_sends_push_when_total_ties_the_line(repository: Repository) -> None:
    _seed_sent_bet(repository, game_pk=1, market="game_total_under", point=9.0)
    client = _FakeApiClient({"2026-07-17": [_game_summary(1, status="Game Over")]})
    notifier = _FakeTelegramNotifier()

    notified = check_pending_bet_results(
        api_client=client, repository=repository,
        game_result_service=_FakeGameResultService({1: (5, 4)}), telegram_notifier=notifier,
    )

    assert notified == 1
    assert notifier.sent[0]["outcome_label"] == "PUSH"
    assert repository.list_resolved_value_bets()[0].outcome == "push"


def test_grades_home_team_total_using_home_runs_only(repository: Repository) -> None:
    _seed_sent_bet(repository, game_pk=1, market="home_team_total_over", point=3.5)
    client = _FakeApiClient({"2026-07-17": [_game_summary(1, status="Final")]})
    notifier = _FakeTelegramNotifier()

    notified = check_pending_bet_results(
        api_client=client, repository=repository,
        game_result_service=_FakeGameResultService({1: (5, 1)}), telegram_notifier=notifier,
    )

    assert notified == 1
    assert notifier.sent[0]["outcome_label"] == "GREEN"


def test_does_not_mark_notified_when_telegram_send_fails(repository: Repository) -> None:
    _seed_sent_bet(repository, game_pk=1, market="game_total_over", point=8.5)
    client = _FakeApiClient({"2026-07-17": [_game_summary(1, status="Final")]})
    notifier = _FakeTelegramNotifier(raise_error=True)

    notified = check_pending_bet_results(
        api_client=client, repository=repository,
        game_result_service=_FakeGameResultService({1: (6, 4)}), telegram_notifier=notifier,
    )

    assert notified == 0
    assert len(repository.list_bets_pending_result_check()) == 1  # continua pendente p/ tentar de novo


def test_only_queries_schedule_once_per_distinct_game_date(repository: Repository) -> None:
    _seed_sent_bet(repository, game_pk=1, market="game_total_over", point=8.5)
    _seed_sent_bet(repository, game_pk=2, market="game_total_over", point=8.5)
    client = _FakeApiClient(
        {"2026-07-17": [_game_summary(1, status="Final"), _game_summary(2, status="Final")]}
    )
    notifier = _FakeTelegramNotifier()

    check_pending_bet_results(
        api_client=client, repository=repository,
        game_result_service=_FakeGameResultService({1: (6, 4), 2: (6, 4)}), telegram_notifier=notifier,
    )

    assert client.schedule_calls == 1
