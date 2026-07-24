from datetime import date, timedelta

import pytest

from app.tools.flight_search import (
    FlightSearchInput,
    NoFlightsFoundError,
    flight_search_tool,
    search_flights,
)
from app.tools.travelpayouts_client import TravelpayoutsRateLimitError, TravelpayoutsRequestError

PRICES_URL = "https://api.travelpayouts.com/v1/prices/cheap"
AIRLINES_URL = "https://api.travelpayouts.com/data/en/airlines.json"
FUTURE_DATE = (date.today() + timedelta(days=30)).isoformat()


def _sample_payload():
    """Shape per Travelpayouts docs: data keyed by destination/date, values are fare dicts."""
    return {
        "success": True,
        "data": {
            "IST": {
                "price": 150.0,
                "airline": "TK",
                "flight_number": "372",
                "origin_airport": "TAS",
                "destination_airport": "IST",
                "departure_at": f"{FUTURE_DATE}T22:10:00+00:00",
                "transfers": 1,
                "duration": 405,
            }
        },
    }


def _multi_fare_payload():
    return {
        "success": True,
        "data": {
            "IST": {
                "0": {
                    "price": 150.0,
                    "airline": "TK",
                    "flight_number": "372",
                    "origin_airport": "TAS",
                    "destination_airport": "IST",
                    "departure_at": f"{FUTURE_DATE}T22:10:00+00:00",
                    "transfers": 1,
                    "duration": 405,
                },
                "1": {
                    "price": 180.0,
                    "airline": "HY",
                    "flight_number": "701",
                    "origin_airport": "TAS",
                    "destination_airport": "IST",
                    "departure_at": f"{FUTURE_DATE}T07:20:00+00:00",
                    "transfers": 0,
                    "duration": 250,
                },
            }
        },
    }


def test_search_flights_parses_and_sorts_by_price(responses, travelpayouts_test_client):
    responses.add(responses.GET, PRICES_URL, json=_multi_fare_payload(), status=200)

    params = FlightSearchInput(origin="tas", destination="ist", date=FUTURE_DATE)
    flights = search_flights(params, client=travelpayouts_test_client)

    assert len(flights) == 2
    assert flights[0]["airline"] == "TK"
    assert flights[0]["price"] == 150.0
    assert flights[0]["stops"] == 1
    assert flights[1]["airline"] == "HY"
    assert flights[1]["stops"] == 0
    assert flights[1]["flight_number"] == "HY701"
    assert flights[1]["duration"] == "4h 10m"
    assert flights[1]["currency"] == "USD"


def test_search_flights_falls_back_to_airline_code_when_reference_lookup_unmocked(
    responses, travelpayouts_test_client
):
    """No mock registered for the airlines.json reference endpoint -> the lookup fails and we
    fall back to the raw 2-letter code rather than crashing or leaving the field blank."""
    responses.add(responses.GET, PRICES_URL, json=_multi_fare_payload(), status=200)

    params = FlightSearchInput(origin="TAS", destination="IST", date=FUTURE_DATE)
    flights = search_flights(params, client=travelpayouts_test_client)

    assert {f["airline"] for f in flights} == {"TK", "HY"}


def test_search_flights_resolves_airline_code_to_real_name(responses, travelpayouts_test_client):
    responses.add(responses.GET, PRICES_URL, json=_multi_fare_payload(), status=200)
    responses.add(
        responses.GET,
        AIRLINES_URL,
        json=[
            {"code": "TK", "name": "Turkish Airlines"},
            {"code": "HY", "name": "Uzbekistan Airways"},
        ],
        status=200,
    )

    params = FlightSearchInput(origin="TAS", destination="IST", date=FUTURE_DATE)
    flights = search_flights(params, client=travelpayouts_test_client)

    names = {f["airline_code"]: f["airline"] for f in flights}
    assert names == {"TK": "Turkish Airlines", "HY": "Uzbekistan Airways"}


def test_search_flights_handles_nested_numeric_keys(responses, travelpayouts_test_client):
    """Some Travelpayouts response variants nest multiple fares one level deeper under numeric
    keys (destination -> {"0": {...}, "1": {...}}) instead of a single flat fare dict."""
    responses.add(responses.GET, PRICES_URL, json=_multi_fare_payload(), status=200)

    params = FlightSearchInput(origin="TAS", destination="IST", date=FUTURE_DATE)
    flights = search_flights(params, client=travelpayouts_test_client)

    assert len(flights) == 2


def test_search_flights_direct_only_filters_out_multi_stop(responses, travelpayouts_test_client):
    responses.add(responses.GET, PRICES_URL, json=_multi_fare_payload(), status=200)

    params = FlightSearchInput(origin="TAS", destination="IST", date=FUTURE_DATE, direct_only=True)
    flights = search_flights(params, client=travelpayouts_test_client)

    assert len(flights) == 1
    assert flights[0]["airline"] == "HY"


def test_search_flights_max_stops_filter(responses, travelpayouts_test_client):
    responses.add(responses.GET, PRICES_URL, json=_multi_fare_payload(), status=200)

    params = FlightSearchInput(origin="TAS", destination="IST", date=FUTURE_DATE, max_stops=0)
    flights = search_flights(params, client=travelpayouts_test_client)

    assert len(flights) == 1
    assert flights[0]["airline"] == "HY"


def test_search_flights_time_of_day_filter(responses, travelpayouts_test_client):
    responses.add(responses.GET, PRICES_URL, json=_multi_fare_payload(), status=200)

    params = FlightSearchInput(origin="TAS", destination="IST", date=FUTURE_DATE, time_of_day="evening")
    flights = search_flights(params, client=travelpayouts_test_client)

    assert len(flights) == 1
    assert flights[0]["airline"] == "TK"


def test_search_flights_no_results_raises(responses, travelpayouts_test_client):
    responses.add(responses.GET, PRICES_URL, json={"success": True, "data": {}}, status=200)

    params = FlightSearchInput(origin="TAS", destination="IST", date=FUTURE_DATE)
    with pytest.raises(NoFlightsFoundError):
        search_flights(params, client=travelpayouts_test_client)


def test_search_flights_rate_limit_raises(responses, travelpayouts_test_client):
    responses.add(responses.GET, PRICES_URL, json={"error": "rate limited"}, status=429)

    params = FlightSearchInput(origin="TAS", destination="IST", date=FUTURE_DATE)
    with pytest.raises(TravelpayoutsRateLimitError):
        search_flights(params, client=travelpayouts_test_client)


def test_search_flights_server_error_raises(responses, travelpayouts_test_client):
    responses.add(responses.GET, PRICES_URL, json={"error": "boom"}, status=500)

    params = FlightSearchInput(origin="TAS", destination="IST", date=FUTURE_DATE)
    with pytest.raises(TravelpayoutsRequestError):
        search_flights(params, client=travelpayouts_test_client)


def test_search_flights_api_success_false_raises(responses, travelpayouts_test_client):
    responses.add(responses.GET, PRICES_URL, json={"success": False, "error": "bad token"}, status=200)

    params = FlightSearchInput(origin="TAS", destination="IST", date=FUTURE_DATE)
    with pytest.raises(TravelpayoutsRequestError):
        search_flights(params, client=travelpayouts_test_client)


def test_flight_search_input_rejects_past_date():
    with pytest.raises(ValueError):
        FlightSearchInput(origin="TAS", destination="IST", date="2020-01-01")


def test_flight_search_input_rejects_bad_date_format():
    with pytest.raises(ValueError):
        FlightSearchInput(origin="TAS", destination="IST", date="15-08-2026")


def test_flight_search_tool_returns_ok_status(responses, travelpayouts_test_client, mocker):
    responses.add(responses.GET, PRICES_URL, json=_multi_fare_payload(), status=200)
    mocker.patch("app.tools.flight_search.get_travelpayouts_client", return_value=travelpayouts_test_client)

    result = flight_search_tool.invoke({"origin": "TAS", "destination": "IST", "date": FUTURE_DATE})

    assert result["status"] == "ok"
    assert result["count"] == 2
    assert "note" in result  # unconditional stops-unknown caveat for this data source
    assert "stop count" in result["note"]


def test_flight_search_tool_adds_passenger_caveat_for_children(responses, travelpayouts_test_client, mocker):
    responses.add(responses.GET, PRICES_URL, json=_sample_payload(), status=200)
    mocker.patch("app.tools.flight_search.get_travelpayouts_client", return_value=travelpayouts_test_client)

    result = flight_search_tool.invoke(
        {"origin": "TAS", "destination": "IST", "date": FUTURE_DATE, "children": 1}
    )

    assert result["status"] == "ok"
    assert "note" in result


def test_flight_search_tool_returns_no_results_status(responses, travelpayouts_test_client, mocker):
    responses.add(responses.GET, PRICES_URL, json={"success": True, "data": {}}, status=200)
    mocker.patch("app.tools.flight_search.get_travelpayouts_client", return_value=travelpayouts_test_client)

    result = flight_search_tool.invoke({"origin": "TAS", "destination": "IST", "date": FUTURE_DATE})

    assert result["status"] == "no_results"
    assert result["flights"] == []


def test_flight_search_tool_returns_error_status_on_api_failure(responses, travelpayouts_test_client, mocker):
    responses.add(responses.GET, PRICES_URL, json={"error": "boom"}, status=500)
    mocker.patch("app.tools.flight_search.get_travelpayouts_client", return_value=travelpayouts_test_client)

    result = flight_search_tool.invoke({"origin": "TAS", "destination": "IST", "date": FUTURE_DATE})

    assert result["status"] == "error"
    assert "message" in result
