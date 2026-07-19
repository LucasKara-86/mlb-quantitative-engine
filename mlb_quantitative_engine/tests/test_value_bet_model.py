from __future__ import annotations

from mlb_quantitative_engine.models.value_bet import describe_market


def test_describe_market_game_total() -> None:
    assert describe_market("game_total_over", "Yankees", "Red Sox") == "Jogo Over"
    assert describe_market("game_total_under", "Yankees", "Red Sox") == "Jogo Under"


def test_describe_market_team_total_uses_real_team_name() -> None:
    assert describe_market("home_team_total_over", "Yankees", "Red Sox") == "Yankees Over"
    assert describe_market("away_team_total_under", "Yankees", "Red Sox") == "Red Sox Under"
