from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mlb_quantitative_engine.analytics.schedule_planning import (
    compute_incremental_trigger_times,
    compute_result_check_window,
)


def _t(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 7, 20, hour, minute, tzinfo=timezone.utc)


# --- compute_incremental_trigger_times ---


def test_no_games_produces_no_triggers() -> None:
    assert compute_incremental_trigger_times([], now=_t(9)) == []


def test_single_game_produces_two_triggers_by_default() -> None:
    triggers = compute_incremental_trigger_times([_t(22, 40)], now=_t(9))
    assert triggers == [_t(22, 10), _t(22, 25)]  # -30 e -15


def test_triggers_are_sorted_and_deduplicated_across_games() -> None:
    # dois jogos: 22:40 e 22:55. -30/-15 de cada = 22:10, 22:25, 22:25, 22:40
    # o 22:25 coincide (é o -15 do 22:40 e o -30 do 22:55) e deve aparecer uma vez só
    triggers = compute_incremental_trigger_times([_t(22, 40), _t(22, 55)], now=_t(9))
    assert triggers == [_t(22, 10), _t(22, 25), _t(22, 40)]


def test_past_triggers_are_dropped() -> None:
    # jogo às 22:40; se agora já são 22:20, o gatilho -30 (22:10) já passou e sai;
    # só o -15 (22:25) permanece
    triggers = compute_incremental_trigger_times([_t(22, 40)], now=_t(22, 20))
    assert triggers == [_t(22, 25)]


def test_all_triggers_past_yields_empty() -> None:
    triggers = compute_incremental_trigger_times([_t(22, 40)], now=_t(23, 0))
    assert triggers == []


def test_custom_offsets_are_respected() -> None:
    triggers = compute_incremental_trigger_times(
        [_t(20, 0)], now=_t(9), offsets=(timedelta(minutes=45),)
    )
    assert triggers == [_t(19, 15)]


# --- compute_result_check_window ---


def test_result_window_is_none_without_games() -> None:
    assert compute_result_check_window([], now=_t(9)) is None


def test_result_window_spans_first_game_to_last_plus_tail() -> None:
    window = compute_result_check_window([_t(19, 40), _t(22, 10)], now=_t(9))
    assert window == (_t(19, 40), _t(22, 10) + timedelta(hours=4))


def test_result_window_start_is_never_in_the_past() -> None:
    # se agora (20:00) já passou do primeiro jogo (19:40), a janela começa em 'now'
    window = compute_result_check_window([_t(19, 40), _t(22, 10)], now=_t(20, 0))
    assert window is not None
    assert window[0] == _t(20, 0)


def test_result_window_is_none_when_entirely_in_the_past() -> None:
    # último jogo 19:40 + 4h = 23:40; se agora são 23:50, a janela já acabou
    assert compute_result_check_window([_t(19, 40)], now=_t(23, 50)) is None


def test_result_window_custom_tail() -> None:
    window = compute_result_check_window([_t(19, 0)], now=_t(9), tail=timedelta(hours=3))
    assert window == (_t(19, 0), _t(22, 0))
