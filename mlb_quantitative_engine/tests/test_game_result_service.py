from __future__ import annotations

from typing import Any, Dict

from mlb_quantitative_engine.services.game_result_service import GameResultService


class _FakeApiClient:
    def __init__(self, boxscore: Dict[str, Any]) -> None:
        self.boxscore = boxscore

    def get_boxscore(self, game_pk: int) -> Dict[str, Any]:
        return self.boxscore


def _boxscore(home_runs, away_runs) -> Dict[str, Any]:
    return {
        "teams": {
            "home": {"teamStats": {"batting": {"runs": home_runs}}},
            "away": {"teamStats": {"batting": {"runs": away_runs}}},
        }
    }


def test_get_final_score_returns_result_when_available() -> None:
    client = _FakeApiClient(_boxscore(10, 0))
    service = GameResultService(api_client=client)

    result = service.get_final_score(824766)

    assert result is not None
    assert result.home_runs == 10
    assert result.away_runs == 0
    assert result.total_runs == 10


def test_get_final_score_returns_none_when_runs_missing() -> None:
    client = _FakeApiClient({"teams": {"home": {"teamStats": {"batting": {}}}, "away": {"teamStats": {"batting": {}}}}})
    service = GameResultService(api_client=client)

    assert service.get_final_score(1) is None


def test_determine_totals_outcome_over() -> None:
    client = _FakeApiClient(_boxscore(6, 4))  # total 10
    service = GameResultService(api_client=client)

    assert service.determine_totals_outcome(1, point=8.5) == "over"


def test_determine_totals_outcome_under() -> None:
    client = _FakeApiClient(_boxscore(2, 1))  # total 3
    service = GameResultService(api_client=client)

    assert service.determine_totals_outcome(1, point=8.5) == "under"


def test_determine_totals_outcome_push_on_integer_line() -> None:
    client = _FakeApiClient(_boxscore(5, 4))  # total 9
    service = GameResultService(api_client=client)

    assert service.determine_totals_outcome(1, point=9.0) == "push"


def test_determine_totals_outcome_none_when_game_not_finished() -> None:
    client = _FakeApiClient({"teams": {"home": {"teamStats": {"batting": {}}}, "away": {"teamStats": {"batting": {}}}}})
    service = GameResultService(api_client=client)

    assert service.determine_totals_outcome(1, point=8.5) is None


def test_classify_total_over_under_push() -> None:
    assert GameResultService.classify_total(10, 8.5) == "over"
    assert GameResultService.classify_total(3, 8.5) == "under"
    assert GameResultService.classify_total(9, 9.0) == "push"


def test_determine_team_total_outcome_home_side() -> None:
    client = _FakeApiClient(_boxscore(home_runs=6, away_runs=2))
    service = GameResultService(api_client=client)

    assert service.determine_team_total_outcome(1, "home", point=3.5) == "over"
    assert service.determine_team_total_outcome(1, "away", point=3.5) == "under"


def test_determine_team_total_outcome_none_when_game_not_finished() -> None:
    client = _FakeApiClient({"teams": {"home": {"teamStats": {"batting": {}}}, "away": {"teamStats": {"batting": {}}}}})
    service = GameResultService(api_client=client)

    assert service.determine_team_total_outcome(1, "home", point=3.5) is None
