import pytest

from app.tools.compare_flights import (
    NotEnoughFlightsError,
    compare_flights,
    compare_flights_tool,
    filter_flights_for_comparison,
)

FLIGHTS = [
    {
        "airline": "Uzbekistan Airways",
        "flight_number": "HY701",
        "departure_airport": "TAS",
        "arrival_airport": "IST",
        "departure_time": "2026-08-15T07:20:00",
        "arrival_time": "2026-08-15T09:30:00",
        "duration": "4h 10m",
        "stops": 0,
        "cabin_class": "economy",
        "price": 180.0,
        "currency": "USD",
    },
    {
        "airline": "Turkish Airlines",
        "flight_number": "TK372",
        "departure_airport": "TAS",
        "arrival_airport": "IST",
        "departure_time": "2026-08-15T22:10:00",
        "arrival_time": "2026-08-16T04:55:00",
        "duration": "6h 45m",
        "stops": 1,
        "cabin_class": "economy",
        "price": 150.0,
        "currency": "USD",
    },
]


def test_compare_flights_by_price_recommends_cheapest():
    result = compare_flights(FLIGHTS, metric="price")

    assert result["best_flight"]["flight_number"] == "TK372"
    assert "cheapest" in result["recommendation"]
    assert len(result["table"]) == 2


def test_compare_flights_by_duration_recommends_shortest():
    result = compare_flights(FLIGHTS, metric="duration")

    assert result["best_flight"]["flight_number"] == "HY701"
    assert "shortest" in result["recommendation"]


def test_compare_flights_by_stops_recommends_fewest():
    result = compare_flights(FLIGHTS, metric="stops")
    assert result["best_flight"]["flight_number"] == "HY701"


def test_compare_flights_requires_at_least_two():
    with pytest.raises(NotEnoughFlightsError):
        compare_flights(FLIGHTS[:1])


def test_filter_flights_by_airline():
    filtered = filter_flights_for_comparison(FLIGHTS, airlines=["Turkish"])
    assert len(filtered) == 1
    assert filtered[0]["flight_number"] == "TK372"


def test_filter_flights_by_flight_number():
    filtered = filter_flights_for_comparison(FLIGHTS, flight_numbers=["hy701"])
    assert len(filtered) == 1
    assert filtered[0]["airline"] == "Uzbekistan Airways"


def test_filter_flights_no_filters_returns_all():
    filtered = filter_flights_for_comparison(FLIGHTS)
    assert filtered == FLIGHTS


def test_compare_flights_tool_is_not_directly_invokable():
    """compare_flights_tool requires state injection by the agent's tool-execution node;
    calling it directly (as the LLM's raw tool-call would) must fail loudly, not silently
    fabricate a comparison."""
    with pytest.raises(RuntimeError):
        compare_flights_tool.invoke({"metric": "price"})
