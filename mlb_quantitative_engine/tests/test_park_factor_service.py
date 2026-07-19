from __future__ import annotations

from mlb_quantitative_engine.services.park_factor_service import (
    NEUTRAL_PARK_FACTORS,
    ParkFactorService,
)


def test_known_venue_returns_specific_factors() -> None:
    service = ParkFactorService()
    factors = service.get_park_factors("Coors Field")

    assert factors.venue == "Coors Field"
    assert factors.run_factor > 1.10  # Coors é notoriamente o parque mais hitter-friendly da MLB


def test_pitcher_friendly_park_has_run_factor_below_one() -> None:
    service = ParkFactorService()
    factors = service.get_park_factors("Oracle Park")

    assert factors.run_factor < 1.00


def test_unknown_venue_falls_back_to_neutral() -> None:
    service = ParkFactorService()
    factors = service.get_park_factors("Estádio Inexistente")

    assert factors == NEUTRAL_PARK_FACTORS
    assert factors.run_factor == 1.00


def test_none_venue_falls_back_to_neutral() -> None:
    service = ParkFactorService()
    assert service.get_park_factors(None) == NEUTRAL_PARK_FACTORS


def test_all_30_mlb_venues_are_registered() -> None:
    service = ParkFactorService()
    venues = [
        "Chase Field", "Sutter Health Park", "Truist Park", "Oriole Park at Camden Yards",
        "Fenway Park", "Wrigley Field", "Rate Field", "Great American Ball Park",
        "Progressive Field", "Coors Field", "Comerica Park", "Daikin Park",
        "Kauffman Stadium", "Angel Stadium", "UNIQLO Field at Dodger Stadium",
        "loanDepot park", "American Family Field", "Target Field", "Citi Field",
        "Yankee Stadium", "Citizens Bank Park", "PNC Park", "Petco Park",
        "Oracle Park", "T-Mobile Park", "Busch Stadium", "Tropicana Field",
        "Globe Life Field", "Rogers Centre", "Nationals Park",
    ]
    assert len(venues) == 30

    for venue in venues:
        factors = service.get_park_factors(venue)
        assert factors.venue == venue
        assert factors != NEUTRAL_PARK_FACTORS


def test_yankee_stadium_strongly_favors_left_handed_home_runs() -> None:
    """O right field curto do Yankee Stadium é uma característica bem documentada."""
    service = ParkFactorService()
    factors = service.get_park_factors("Yankee Stadium")

    assert factors.lhb_factor > factors.rhb_factor
    assert factors.right_field_ft < factors.left_field_ft
