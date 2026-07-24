"""Airport Search Tool: resolves airport codes <-> city/airport names via Travelpayouts'
autocomplete API. This endpoint needs no API token at all.
"""
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field

from app.tools.travelpayouts_client import (
    TravelpayoutsClient,
    TravelpayoutsError,
    get_travelpayouts_client,
)

LocationSubtype = Literal["AIRPORT", "CITY", "AIRPORT,CITY"]

_SUBTYPE_TO_TYPES = {
    "AIRPORT": ["airport"],
    "CITY": ["city"],
    "AIRPORT,CITY": ["airport", "city"],
}


class AirportSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ..., description="A city/airport name (e.g. 'Istanbul') or an IATA code (e.g. 'TAS') to resolve"
    )
    subtype: LocationSubtype = Field(
        "AIRPORT,CITY", description="Restrict results to airports only, cities only, or both"
    )


class NoLocationsFoundError(Exception):
    """Raised when the autocomplete search returns no matches."""


def search_airports(params: AirportSearchInput, client: Optional[TravelpayoutsClient] = None) -> list[dict]:
    """Pure function: calls the Travelpayouts autocomplete endpoint, returns parsed location dicts."""
    client = client or get_travelpayouts_client()

    query_params = {"term": params.query.strip(), "locale": "en"}
    entries = client.get_autocomplete(query_params)

    wanted_types = set(_SUBTYPE_TO_TYPES[params.subtype])
    is_code_query = len(query_params["term"]) == 3 and query_params["term"].isalpha()

    term_upper = query_params["term"].upper()
    locations = []
    for entry in entries:
        if entry.get("type") not in wanted_types:
            continue
        if is_code_query:
            # Keep exact code matches AND same-city entries (e.g. querying city code "IST"
            # should also surface Sabiha Gokcen/SAW, whose city_code is "IST" even though its
            # own code differs) — otherwise a code query for a multi-airport city silently
            # drops its other airports.
            code_matches = entry.get("code", "").upper() == term_upper
            same_city = entry.get("city_code", "").upper() == term_upper
            if not (code_matches or same_city):
                continue
        is_airport = entry.get("type") == "airport"
        locations.append(
            {
                "name": entry.get("name"),
                "iata_code": entry.get("code"),
                "type": entry.get("type", "").upper(),
                "city_name": entry.get("city_name") if is_airport else entry.get("name"),
                "country_name": entry.get("country_name"),
            }
        )

    if not locations:
        raise NoLocationsFoundError(f"No airports/cities found matching '{params.query}'.")
    return locations


@tool("airport_search_tool", args_schema=AirportSearchInput)
def airport_search_tool(query: str, subtype: str = "AIRPORT,CITY") -> dict:
    """Resolve an IATA airport/city code to its name, or a city/airport name to its IATA code(s),
    via the real Travelpayouts autocomplete API. Never guesses codes from memory."""
    params = AirportSearchInput(query=query, subtype=subtype)
    try:
        locations = search_airports(params)
        return {"status": "ok", "locations": locations, "count": len(locations)}
    except NoLocationsFoundError as exc:
        return {"status": "no_results", "message": str(exc), "locations": []}
    except TravelpayoutsError as exc:
        return {"status": "error", "message": str(exc), "locations": []}
