import pytest

from app.tools.airport_search import (
    AirportSearchInput,
    NoLocationsFoundError,
    airport_search_tool,
    search_airports,
)
from app.tools.travelpayouts_client import TravelpayoutsRequestError

AUTOCOMPLETE_URL = "https://autocomplete.travelpayouts.com/places2"


def _tas_payload():
    return [
        {
            "type": "city",
            "code": "TAS",
            "name": "Tashkent",
            "country_code": "UZ",
            "country_name": "Uzbekistan",
        }
    ]


def _istanbul_payload():
    return [
        {
            "type": "airport",
            "code": "IST",
            "name": "Istanbul New Airport",
            "country_code": "TR",
            "country_name": "Turkiye",
            "city_code": "IST",
            "city_name": "Istanbul",
        },
        {
            "type": "airport",
            "code": "SAW",
            "name": "Sabiha Gokcen International Airport",
            "country_code": "TR",
            "country_name": "Turkiye",
            "city_code": "IST",
            "city_name": "Istanbul",
        },
    ]


def _ist_code_query_payload():
    """Real API behavior when querying the code 'IST' directly: returns the city entry plus
    both of its airports (one sharing the code, one — Sabiha Gokcen — only sharing city_code)."""
    return [
        {
            "type": "city",
            "code": "IST",
            "name": "Istanbul",
            "country_code": "TR",
            "country_name": "Turkiye",
        },
        {
            "type": "airport",
            "code": "IST",
            "name": "Istanbul New Airport",
            "country_code": "TR",
            "country_name": "Turkiye",
            "city_code": "IST",
            "city_name": "Istanbul",
        },
        {
            "type": "airport",
            "code": "SAW",
            "name": "Sabiha Gokcen International Airport",
            "country_code": "TR",
            "country_name": "Turkiye",
            "city_code": "IST",
            "city_name": "Istanbul",
        },
    ]


def _tas_fuzzy_noise_payload():
    """Real API behavior: a 3-letter code query also returns fuzzy name-prefix matches that
    aren't the actual code being asked about."""
    return _tas_payload() + [
        {"type": "city", "code": "AGM", "name": "Tasiilaq", "country_code": "GL", "country_name": "Greenland"}
    ]


def test_search_airports_code_query_includes_same_city_airports(responses, travelpayouts_test_client):
    """Regression test: querying the city code 'IST' must also surface Sabiha Gokcen (SAW),
    which shares city_code 'IST' but not the code itself — a real bug caught via live testing
    where the exact-code filter silently dropped it."""
    responses.add(responses.GET, AUTOCOMPLETE_URL, json=_ist_code_query_payload(), status=200)

    params = AirportSearchInput(query="IST")
    locations = search_airports(params, client=travelpayouts_test_client)

    codes = {loc["iata_code"] for loc in locations}
    assert codes == {"IST", "SAW"}
    assert len(locations) == 3


def test_search_airports_resolves_iata_code(responses, travelpayouts_test_client):
    responses.add(responses.GET, AUTOCOMPLETE_URL, json=_tas_payload(), status=200)

    params = AirportSearchInput(query="TAS")
    locations = search_airports(params, client=travelpayouts_test_client)

    assert len(locations) == 1
    assert locations[0]["city_name"] == "Tashkent"
    assert locations[0]["iata_code"] == "TAS"


def test_search_airports_filters_out_fuzzy_noise_for_code_query(responses, travelpayouts_test_client):
    responses.add(responses.GET, AUTOCOMPLETE_URL, json=_tas_fuzzy_noise_payload(), status=200)

    params = AirportSearchInput(query="TAS")
    locations = search_airports(params, client=travelpayouts_test_client)

    assert len(locations) == 1
    assert locations[0]["iata_code"] == "TAS"


def test_search_airports_resolves_city_to_multiple_airports(responses, travelpayouts_test_client):
    responses.add(responses.GET, AUTOCOMPLETE_URL, json=_istanbul_payload(), status=200)

    params = AirportSearchInput(query="Istanbul")
    locations = search_airports(params, client=travelpayouts_test_client)

    assert len(locations) == 2
    codes = {loc["iata_code"] for loc in locations}
    assert codes == {"IST", "SAW"}


def test_search_airports_no_results_raises(responses, travelpayouts_test_client):
    responses.add(responses.GET, AUTOCOMPLETE_URL, json=[], status=200)

    params = AirportSearchInput(query="Nowhereland")
    with pytest.raises(NoLocationsFoundError):
        search_airports(params, client=travelpayouts_test_client)


def test_search_airports_api_error_raises(responses, travelpayouts_test_client):
    responses.add(responses.GET, AUTOCOMPLETE_URL, json={"error": "boom"}, status=500)

    params = AirportSearchInput(query="TAS")
    with pytest.raises(TravelpayoutsRequestError):
        search_airports(params, client=travelpayouts_test_client)


def test_airport_search_tool_returns_ok_status(responses, travelpayouts_test_client, mocker):
    responses.add(responses.GET, AUTOCOMPLETE_URL, json=_tas_payload(), status=200)
    mocker.patch("app.tools.airport_search.get_travelpayouts_client", return_value=travelpayouts_test_client)

    result = airport_search_tool.invoke({"query": "TAS"})

    assert result["status"] == "ok"
    assert result["locations"][0]["iata_code"] == "TAS"


def test_airport_search_tool_returns_no_results_status(responses, travelpayouts_test_client, mocker):
    responses.add(responses.GET, AUTOCOMPLETE_URL, json=[], status=200)
    mocker.patch("app.tools.airport_search.get_travelpayouts_client", return_value=travelpayouts_test_client)

    result = airport_search_tool.invoke({"query": "Nowhereland"})

    assert result["status"] == "no_results"
