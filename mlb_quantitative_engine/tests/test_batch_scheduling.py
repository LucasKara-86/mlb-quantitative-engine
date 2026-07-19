from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mlb_quantitative_engine.analytics.batch_scheduling import (
    Batch,
    GameSchedule,
    compute_batches,
    due_batches,
)


def _t(hour: int, minute: int) -> datetime:
    return datetime(2026, 7, 18, hour, minute, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Comportamento padrão atual: cada jogo é seu próprio lote, disparado 30min antes
# ---------------------------------------------------------------------------


def test_each_game_becomes_its_own_batch_by_default() -> None:
    """Regra atual: sem agrupamento por janela -- cada jogo dispara individualmente
    30 minutos antes do seu próprio horário de início."""
    games = [
        GameSchedule(game_pk=1, start_time=_t(13, 15)),
        GameSchedule(game_pk=2, start_time=_t(14, 10)),
        GameSchedule(game_pk=3, start_time=_t(15, 20)),
    ]

    batches = compute_batches(games)

    assert len(batches) == 3
    assert batches[0].game_pks == [1]
    assert batches[0].anchor_time == _t(13, 15)
    assert batches[0].trigger_time == _t(12, 45)  # 30 min antes
    assert batches[1].game_pks == [2]
    assert batches[1].trigger_time == _t(13, 40)
    assert batches[2].game_pks == [3]
    assert batches[2].trigger_time == _t(14, 50)


def test_games_at_the_exact_same_start_time_share_a_batch() -> None:
    """Doubleheader com os dois jogos marcados para o mesmo instante -> mesmo lote
    (window=0 só agrupa quando o horário é IDÊNTICO, não só próximo)."""
    games = [
        GameSchedule(game_pk=1, start_time=_t(14, 10)),
        GameSchedule(game_pk=2, start_time=_t(14, 10)),
    ]
    batches = compute_batches(games)
    assert len(batches) == 1
    assert batches[0].game_pks == [1, 2]


def test_games_one_minute_apart_do_not_share_a_batch() -> None:
    games = [
        GameSchedule(game_pk=1, start_time=_t(14, 10)),
        GameSchedule(game_pk=2, start_time=_t(14, 11)),
    ]
    batches = compute_batches(games)
    assert len(batches) == 2


def test_empty_game_list_produces_no_batches() -> None:
    assert compute_batches([]) == []


def test_single_game_produces_single_batch() -> None:
    games = [GameSchedule(game_pk=1, start_time=_t(14, 10))]
    batches = compute_batches(games)
    assert len(batches) == 1
    assert batches[0].game_pks == [1]
    assert batches[0].trigger_time == _t(13, 40)


def test_games_out_of_order_are_sorted_before_batching() -> None:
    games = [
        GameSchedule(game_pk=2, start_time=_t(15, 20)),
        GameSchedule(game_pk=1, start_time=_t(14, 10)),
    ]
    batches = compute_batches(games)
    assert [b.game_pks for b in batches] == [[1], [2]]  # ordem cronológica


# ---------------------------------------------------------------------------
# Agrupamento por janela: ainda funcional via parâmetro explícito (não é mais o padrão)
# ---------------------------------------------------------------------------


def test_explicit_window_still_groups_nearby_games() -> None:
    """O agrupamento por janela continua funcionando se alguém passar `window`
    explicitamente -- só deixou de ser o comportamento padrão do app."""
    games = [
        GameSchedule(game_pk=1, start_time=_t(14, 10)),
        GameSchedule(game_pk=2, start_time=_t(15, 20)),
        GameSchedule(game_pk=3, start_time=_t(16, 7)),
        GameSchedule(game_pk=4, start_time=_t(16, 10)),
        GameSchedule(game_pk=5, start_time=_t(17, 10)),
    ]

    batches = compute_batches(games, window=timedelta(hours=2), lead_time=timedelta(minutes=20))

    assert len(batches) == 2
    assert batches[0].anchor_time == _t(14, 10)
    assert batches[0].trigger_time == _t(13, 50)
    assert batches[0].game_pks == [1, 2, 3, 4]

    assert batches[1].anchor_time == _t(17, 10)
    assert batches[1].trigger_time == _t(16, 50)
    assert batches[1].game_pks == [5]


def test_explicit_window_boundary_is_inclusive() -> None:
    games = [
        GameSchedule(game_pk=1, start_time=_t(14, 10)),
        GameSchedule(game_pk=2, start_time=_t(16, 10)),  # exatamente +2h
    ]
    batches = compute_batches(games, window=timedelta(hours=2))
    assert len(batches) == 1
    assert batches[0].game_pks == [1, 2]


def test_explicit_window_is_anchored_to_first_game_not_chained_transitively() -> None:
    """Um jogo a 1h50 do anterior mas > 2h da âncora do lote deve iniciar novo lote --
    a janela é sempre contra a âncora, não contra o jogo anterior na cadeia."""
    games = [
        GameSchedule(game_pk=1, start_time=_t(10, 0)),   # âncora
        GameSchedule(game_pk=2, start_time=_t(11, 55)),  # +1h55, ok no lote 1
        GameSchedule(game_pk=3, start_time=_t(13, 40)),  # +3h40 da âncora (>2h) -> novo lote
    ]
    batches = compute_batches(games, window=timedelta(hours=2))
    assert len(batches) == 2
    assert batches[0].game_pks == [1, 2]
    assert batches[1].game_pks == [3]
    assert batches[1].anchor_time == _t(13, 40)


# ---------------------------------------------------------------------------
# due_batches
# ---------------------------------------------------------------------------


def test_due_batches_returns_only_batches_past_trigger_and_unprocessed() -> None:
    batches = [
        Batch(anchor_time=_t(14, 10), trigger_time=_t(13, 40), games=[GameSchedule(1, _t(14, 10))]),
        Batch(anchor_time=_t(17, 10), trigger_time=_t(16, 40), games=[GameSchedule(2, _t(17, 10))]),
    ]

    now = _t(14, 0)  # depois do primeiro disparo (13:40), antes do segundo (16:40)
    result = due_batches(batches, now=now, is_processed=lambda anchor: False)

    assert len(result) == 1
    assert result[0].anchor_time == _t(14, 10)


def test_due_batches_excludes_already_processed_batches() -> None:
    batches = [
        Batch(anchor_time=_t(14, 10), trigger_time=_t(13, 40), games=[GameSchedule(1, _t(14, 10))]),
    ]
    now = _t(14, 0)

    result = due_batches(batches, now=now, is_processed=lambda anchor: anchor == _t(14, 10))
    assert result == []


def test_due_batches_handles_coarse_cron_cadence_by_catching_up() -> None:
    """Com cadência de cron grosseira, um lote cujo disparo já passou há um tempo
    (mas ainda não processado) deve continuar aparecendo como devido."""
    batches = [
        Batch(anchor_time=_t(14, 10), trigger_time=_t(13, 40), games=[GameSchedule(1, _t(14, 10))]),
    ]
    now = _t(15, 45)  # bem depois do disparo -- cron rodou tarde, mas ainda não processou

    result = due_batches(batches, now=now, is_processed=lambda anchor: False)
    assert len(result) == 1
