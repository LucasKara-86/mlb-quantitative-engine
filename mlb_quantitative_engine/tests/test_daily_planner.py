from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from mlb_quantitative_engine.api.mlb_api import GameSummary
from mlb_quantitative_engine.reports.daily_planner import (
    _cron_line,
    _replace_block,
    plan_today,
)


def _t(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 7, 20, hour, minute, tzinfo=timezone.utc)


def _game(game_pk: int, hour: int, minute: int) -> GameSummary:
    dt = _t(hour, minute).isoformat().replace("+00:00", "Z")
    return GameSummary(
        game_pk=game_pk, game_date="2026-07-20", game_datetime=dt,
        home_team=f"Home{game_pk}", away_team=f"Away{game_pk}", venue="Park", status="Scheduled",
        home_probable_pitcher="A", away_probable_pitcher="B",
        home_probable_pitcher_id=1, away_probable_pitcher_id=2, home_team_id=10, away_team_id=20,
    )


class _FakeApiClient:
    def __init__(self, games: List[GameSummary]) -> None:
        self.games = games

    def get_games_for_date(self, date: Optional[str] = None) -> List[GameSummary]:
        return self.games


class _RecordingApply:
    def __init__(self) -> None:
        self.triggers = None
        self.window = None
        self.calls = 0

    def __call__(self, triggers, window) -> None:
        self.calls += 1
        self.triggers = triggers
        self.window = window


def test_plan_today_computes_two_triggers_per_game_and_a_window() -> None:
    apply = _RecordingApply()
    plan_today(
        date="2026-07-20",
        api_client=_FakeApiClient([_game(1, 22, 40), _game(2, 23, 10)]),
        now=_t(9),
        apply_fn=apply,
    )

    assert apply.calls == 1
    # -30/-15 de cada jogo: 22:10, 22:25, 22:40, 22:55
    assert apply.triggers == [_t(22, 10), _t(22, 25), _t(22, 40), _t(22, 55)]
    # janela: primeiro jogo até último + 4h
    assert apply.window == (_t(22, 40), _t(23, 10) + timedelta(hours=4))


def test_plan_today_with_no_games_applies_empty_schedule() -> None:
    apply = _RecordingApply()
    plan_today(date="2026-07-20", api_client=_FakeApiClient([]), now=_t(9), apply_fn=apply)

    assert apply.triggers == []
    assert apply.window is None


def test_plan_today_ignores_games_without_datetime() -> None:
    game_no_time = _game(1, 22, 40)
    game_no_time = GameSummary(**{**game_no_time.__dict__, "game_datetime": None})
    apply = _RecordingApply()

    plan_today(date="2026-07-20", api_client=_FakeApiClient([game_no_time]), now=_t(9), apply_fn=apply)

    assert apply.triggers == []
    assert apply.window is None


def test_plan_today_drops_triggers_already_in_the_past() -> None:
    apply = _RecordingApply()
    # agora 22:20: o gatilho -30 (22:10) já passou; sobra só o -15 (22:25)
    plan_today(
        date="2026-07-20", api_client=_FakeApiClient([_game(1, 22, 40)]), now=_t(22, 20), apply_fn=apply
    )

    assert apply.triggers == [_t(22, 25)]


def test_plan_today_returns_computed_values() -> None:
    triggers, window = plan_today(
        date="2026-07-20",
        api_client=_FakeApiClient([_game(1, 22, 40)]),
        now=_t(9),
        apply_fn=_RecordingApply(),
    )
    assert triggers == [_t(22, 10), _t(22, 25)]
    assert window == (_t(22, 40), _t(22, 40) + timedelta(hours=4))


def test_cron_line_uses_day_and_month_as_one_shot_trigger() -> None:
    trigger = _t(22, 40)
    line = _cron_line(trigger, "echo oi")
    minute, hour, day, month, weekday, *command = line.split()
    local = trigger.astimezone()
    assert (minute, hour, day, month, weekday) == (
        str(local.minute), str(local.hour), str(local.day), str(local.month), "*",
    )
    assert " ".join(command) == "echo oi"


def test_replace_block_inserts_new_block_when_crontab_is_empty() -> None:
    result = _replace_block("", "# BEGIN x", "# END x", ["1 2 3 4 * echo a"])
    assert result == "# BEGIN x\n1 2 3 4 * echo a\n# END x\n"


def test_replace_block_preserves_unrelated_lines() -> None:
    existing = "0 9 * * * echo outro-job\n# BEGIN x\nold line\n# END x\n"
    result = _replace_block(existing, "# BEGIN x", "# END x", ["new line"])
    assert result == "0 9 * * * echo outro-job\n# BEGIN x\nnew line\n# END x\n"


def test_replace_block_with_empty_new_lines_removes_the_block() -> None:
    existing = "0 9 * * * echo outro-job\n# BEGIN x\nold line\n# END x\n"
    result = _replace_block(existing, "# BEGIN x", "# END x", [])
    assert result == "0 9 * * * echo outro-job\n"
